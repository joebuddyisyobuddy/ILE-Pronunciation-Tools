#!/usr/bin/env python3
"""
ILE Pronunciation Coach — Native Messaging Host

Chrome launches this process automatically via Native Messaging.
Receives audio over stdin, runs wav2vec2 ONNX inference, returns IPA phonemes over stdout.

On first run, downloads the ONNX model from HuggingFace (~318 MB).
Sends progress messages back to the extension during download.
"""

import sys
import os
import struct
import json
import base64
import io
from pathlib import Path

# ─── Paths (always relative to this script) ──────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent
MODELS_DIR = SCRIPT_DIR / "models"
VOCAB_PATH = MODELS_DIR / "vocab.json"

MODEL_FILENAME = "model_int8.onnx"
MODEL_PATH = MODELS_DIR / MODEL_FILENAME
MODEL_URL = "https://huggingface.co/onnx-community/wav2vec2-lv-60-espeak-cv-ft-ONNX/resolve/main/onnx/model_int8.onnx"
MODEL_MIN_SIZE = 50 * 1024 * 1024  # 50 MB minimum

# ─── Native Messaging Protocol ────────────────────────────────────────────────

def read_message():
    """Read a length-prefixed JSON message from stdin."""
    raw_length = sys.stdin.buffer.read(4)
    if not raw_length:
        return None
    length = struct.unpack("=I", raw_length)[0]
    data = sys.stdin.buffer.read(length)
    return json.loads(data.decode("utf-8"))


def send_message(msg):
    """Send a length-prefixed JSON message to stdout."""
    encoded = json.dumps(msg, ensure_ascii=False).encode("utf-8")
    sys.stdout.buffer.write(struct.pack("=I", len(encoded)))
    sys.stdout.buffer.write(encoded)
    sys.stdout.buffer.flush()

# ─── Model Self-Provisioning ─────────────────────────────────────────────────

def model_ready():
    """Check if the ONNX model exists and looks valid."""
    return MODEL_PATH.exists() and MODEL_PATH.stat().st_size > MODEL_MIN_SIZE


def download_model():
    """Download the ONNX model, sending progress messages to the extension."""
    import urllib.request

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    tmp_path = MODEL_PATH.with_suffix(".tmp")

    send_message({"status": "downloading", "message": "Downloading phoneme model (318 MB)...", "progress": 0})

    try:
        req = urllib.request.Request(MODEL_URL)
        with urllib.request.urlopen(req) as response:
            total = int(response.headers.get("Content-Length", 0))
            downloaded = 0
            last_pct = -1

            with open(tmp_path, "wb") as f:
                while True:
                    chunk = response.read(1024 * 256)  # 256 KB chunks
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)

                    if total > 0:
                        pct = int(downloaded * 100 / total)
                        if pct != last_pct and pct % 5 == 0:
                            last_pct = pct
                            send_message({
                                "status": "downloading",
                                "message": f"Downloading phoneme model... {pct}%",
                                "progress": pct / 100
                            })

        # Verify size
        if tmp_path.stat().st_size < MODEL_MIN_SIZE:
            tmp_path.unlink()
            send_message({"status": "error", "error": "Download incomplete — file too small. Please try again."})
            return False

        # Atomic rename
        tmp_path.rename(MODEL_PATH)
        send_message({"status": "downloading", "message": "Model downloaded successfully", "progress": 1})
        return True

    except Exception as e:
        if tmp_path.exists():
            tmp_path.unlink()
        send_message({"status": "error", "error": f"Model download failed: {str(e)}"})
        return False

# ─── ONNX Inference ──────────────────────────────────────────────────────────

_session = None
_vocab = None


def load_model():
    """Load ONNX model and vocab. Caches in globals for subsequent calls."""
    global _session, _vocab

    if _session is not None:
        return True

    try:
        import onnxruntime as ort
        _session = ort.InferenceSession(
            str(MODEL_PATH),
            providers=["CPUExecutionProvider"]
        )
    except Exception as e:
        send_message({"status": "error", "error": f"Failed to load model: {str(e)}"})
        return False

    try:
        with open(VOCAB_PATH, "r", encoding="utf-8") as f:
            _vocab = json.load(f)
    except Exception as e:
        send_message({"status": "error", "error": f"Failed to load vocab: {str(e)}"})
        return False

    return True


def analyze_audio(audio_b64, sample_rate, audio_format="float32"):
    """Run phoneme recognition on base64-encoded audio."""
    import numpy as np
    import librosa

    # Decode audio
    audio_bytes = base64.b64decode(audio_b64)

    if audio_format == "float32":
        audio = np.frombuffer(audio_bytes, dtype=np.float32)
    else:
        audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

    # Resample to 16kHz if needed
    if sample_rate != 16000:
        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)

    # Normalize: zero-mean, unit-variance
    audio = (audio - audio.mean()) / (audio.std() + 1e-7)

    # Run inference
    input_values = audio.reshape(1, -1).astype(np.float32)
    input_name = _session.get_inputs()[0].name
    logits = _session.run(None, {input_name: input_values})[0]

    # CTC greedy decode
    predicted_ids = np.argmax(logits[0], axis=-1)

    # Softmax for confidence scores
    exp_logits = np.exp(logits[0] - np.max(logits[0], axis=-1, keepdims=True))
    probs = exp_logits / exp_logits.sum(axis=-1, keepdims=True)
    confidences = np.max(probs, axis=-1)

    # Collapse repeats, remove blanks, build phoneme list
    frame_duration = 320 / 16000  # 20ms per frame
    phonemes = []
    raw_ipa = []
    prev_id = -1
    blank_id = 0  # CTC blank is typically 0

    current_phoneme = None
    current_start = 0
    current_confidences = []

    for i, token_id in enumerate(predicted_ids):
        token_id = int(token_id)

        if token_id == prev_id:
            if current_phoneme is not None:
                current_confidences.append(float(confidences[i]))
            prev_id = token_id
            continue

        # Emit previous phoneme
        if current_phoneme is not None and current_phoneme != blank_id:
            phoneme_str = _vocab.get(str(current_phoneme), "")
            if phoneme_str and phoneme_str not in ("<pad>", "<s>", "</s>", "<unk>"):
                avg_conf = sum(current_confidences) / len(current_confidences) if current_confidences else 0
                phonemes.append({
                    "phoneme": phoneme_str,
                    "start": round(current_start * frame_duration, 3),
                    "end": round(i * frame_duration, 3),
                    "confidence": round(avg_conf, 3)
                })
                raw_ipa.append(phoneme_str)

        # Start new phoneme
        current_phoneme = token_id
        current_start = i
        current_confidences = [float(confidences[i])]
        prev_id = token_id

    # Emit last phoneme
    if current_phoneme is not None and current_phoneme != blank_id:
        phoneme_str = _vocab.get(str(current_phoneme), "")
        if phoneme_str and phoneme_str not in ("<pad>", "<s>", "</s>", "<unk>"):
            avg_conf = sum(current_confidences) / len(current_confidences) if current_confidences else 0
            phonemes.append({
                "phoneme": phoneme_str,
                "start": round(current_start * frame_duration, 3),
                "end": round(len(predicted_ids) * frame_duration, 3),
                "confidence": round(avg_conf, 3)
            })
            raw_ipa.append(phoneme_str)

    return {
        "phonemes": phonemes,
        "rawIPA": "".join(raw_ipa)
    }

# ─── Main Loop ────────────────────────────────────────────────────────────────

def main():
    while True:
        msg = read_message()
        if msg is None:
            break

        action = msg.get("action", "")

        if action == "ping":
            send_message({"status": "ok", "version": "2.0", "modelReady": model_ready()})
            continue

        if action == "analyze":
            # Ensure model is downloaded
            if not model_ready():
                if not download_model():
                    continue

            # Ensure model is loaded
            if not load_model():
                continue

            try:
                result = analyze_audio(
                    msg["audio"],
                    msg.get("sampleRate", 44100),
                    msg.get("format", "float32")
                )
                send_message({"status": "ok", **result})
            except Exception as e:
                send_message({"status": "error", "error": str(e)})
            continue

        send_message({"status": "error", "error": f"Unknown action: {action}"})


if __name__ == "__main__":
    main()