#!/usr/bin/env python3
"""
Export wav2vec2 phoneme model from HuggingFace to ONNX.

Downloads facebook/wav2vec2-lv-60-espeak-cv-ft and exports it as an ONNX model
for use with ONNX Runtime (CPU). Called automatically by setup.py when the model
doesn't exist yet.

Can also be run standalone:
    python export_model.py
    python export_model.py --model-id facebook/wav2vec2-lv-60-espeak-cv-ft
    python export_model.py --output-dir ./models
"""

import sys
import os
import json
import argparse
from pathlib import Path

DEFAULT_MODEL_ID = "facebook/wav2vec2-lv-60-espeak-cv-ft"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "models"
ONNX_FILENAME = "wav2vec2-phoneme.onnx"
VOCAB_FILENAME = "vocab.json"
OPSET_VERSION = 14


def check_build_dependencies():
    """Verify that torch and transformers are available."""
    missing = []
    try:
        import torch
    except ImportError:
        missing.append("torch")
    try:
        import transformers
    except ImportError:
        missing.append("transformers")
    try:
        import onnx
    except ImportError:
        missing.append("onnx")

    if missing:
        print(f"Missing build dependencies: {', '.join(missing)}")
        print(f"Install with: pip install {' '.join(missing)}")
        sys.exit(1)


def export(model_id=DEFAULT_MODEL_ID, output_dir=DEFAULT_OUTPUT_DIR, verbose=True):
    """Download model from HuggingFace and export to ONNX."""
    check_build_dependencies()

    import torch
    import onnx
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = output_dir / ONNX_FILENAME
    vocab_path = output_dir / VOCAB_FILENAME

    # ── Step 1: Download and load model ──
    if verbose:
        print(f"  Downloading model: {model_id}")
        print(f"  This downloads ~1.2 GB of weights from HuggingFace...")

    processor = Wav2Vec2Processor.from_pretrained(model_id)
    model = Wav2Vec2ForCTC.from_pretrained(model_id)
    model.eval()

    if verbose:
        print(f"  Model loaded successfully")

    # ── Step 2: Export vocab.json ──
    # Map token IDs to IPA phoneme strings
    vocab = processor.tokenizer.get_vocab()
    # Invert: id → phoneme
    id_to_phoneme = {v: k for k, v in vocab.items()}
    # Sort by ID for clean output
    vocab_list = {str(i): id_to_phoneme.get(i, "") for i in range(len(id_to_phoneme))}

    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab_list, f, ensure_ascii=False, indent=2)

    if verbose:
        print(f"  Vocab saved: {vocab_path} ({len(vocab_list)} tokens)")

    # ── Step 3: Export to ONNX ──
    if verbose:
        print(f"  Exporting to ONNX (opset {OPSET_VERSION})...")

    # Create dummy input: 1 second of audio at 16kHz
    dummy_input = torch.randn(1, 16000, dtype=torch.float32)

    # Dynamic axes: batch size and sequence length can vary
    dynamic_axes = {
        "input_values": {0: "batch_size", 1: "sequence_length"},
        "logits": {0: "batch_size", 1: "time_steps"},
    }

    with torch.no_grad():
        torch.onnx.export(
            model,
            (dummy_input,),
            str(onnx_path),
            export_params=True,
            opset_version=OPSET_VERSION,
            do_constant_folding=True,
            input_names=["input_values"],
            output_names=["logits"],
            dynamic_axes=dynamic_axes,
        )

    if verbose:
        size_mb = onnx_path.stat().st_size / (1024 * 1024)
        print(f"  ONNX model saved: {onnx_path} ({size_mb:.0f} MB)")

    # ── Step 4: Validate ──
    if verbose:
        print(f"  Validating ONNX model...")

    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)

    if verbose:
        print(f"  Validation passed")

    # ── Step 5: Quick inference test with ONNX Runtime ──
    try:
        import onnxruntime as ort
        import numpy as np

        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        dummy_np = np.random.randn(1, 16000).astype(np.float32)
        result = sess.run(None, {"input_values": dummy_np})

        if verbose:
            print(f"  Inference test passed — output shape: {result[0].shape}")

    except ImportError:
        if verbose:
            print(f"  Skipping inference test (onnxruntime not installed yet)")

    if verbose:
        print(f"  Export complete!")

    return onnx_path, vocab_path


def main():
    parser = argparse.ArgumentParser(description="Export wav2vec2 phoneme model to ONNX")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID,
                        help=f"HuggingFace model ID (default: {DEFAULT_MODEL_ID})")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR),
                        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})")
    args = parser.parse_args()

    export(model_id=args.model_id, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
