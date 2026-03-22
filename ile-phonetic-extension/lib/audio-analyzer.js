/**
 * AudioAnalyzer
 * 
 * Handles microphone recording, audio decoding, and multi-dimensional
 * feature extraction for phonetic comparison.
 * 
 * Features extracted:
 *   - Pitch contour (via YIN in pitch-detector.js)
 *   - Energy envelope (RMS per frame)
 *   - Rhythm / segment durations (onset detection)
 *   - Spectral centroid (brightness)
 */

const AudioAnalyzer = (() => {

  const FRAME_SIZE = 2048;
  const HOP_SIZE = 512;

  let audioContext = null;

  function getAudioContext() {
    if (!audioContext || audioContext.state === 'closed') {
      audioContext = new (window.AudioContext || window.webkitAudioContext)();
    }
    return audioContext;
  }

  // ─── Recording ──────────────────────────────────────────────

  /**
   * Record audio from the user's microphone.
   * Returns a promise that resolves when recording stops.
   * 
   * @param {function} onStop - Callback: called with { audioBuffer, float32Data, sampleRate }
   * @returns {{ stop: function }} - Call .stop() to end recording
   */
  async function startRecording() {
    const ctx = getAudioContext();
    if (ctx.state === 'suspended') await ctx.resume();

    const stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
        sampleRate: 44100
      }
    });

    const mediaRecorder = new MediaRecorder(stream, {
      mimeType: MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
        ? 'audio/webm;codecs=opus'
        : 'audio/webm'
    });

    const chunks = [];

    return new Promise((resolve) => {
      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunks.push(e.data);
      };

      mediaRecorder.onstop = async () => {
        // Stop all tracks to release mic
        stream.getTracks().forEach(t => t.stop());

        const blob = new Blob(chunks, { type: mediaRecorder.mimeType });
        const arrayBuffer = await blob.arrayBuffer();
        const audioBuffer = await ctx.decodeAudioData(arrayBuffer);
        const float32Data = audioBuffer.getChannelData(0); // Mono

        resolve({
          audioBuffer,
          float32Data: new Float32Array(float32Data), // Copy — original may be neutered
          sampleRate: audioBuffer.sampleRate,
          duration: audioBuffer.duration,
          blob
        });
      };

      mediaRecorder.start();

      // Expose stop handle
      resolve.__stopHandle = {
        stop: () => mediaRecorder.stop(),
        mediaRecorder
      };
    });
  }

  /**
   * Wrapper: record with explicit start/stop control.
   */
  function createRecorder() {
    let stopFn = null;
    let resultPromise = null;

    return {
      async start() {
        const ctx = getAudioContext();
        if (ctx.state === 'suspended') await ctx.resume();

        const stream = await navigator.mediaDevices.getUserMedia({
          audio: {
            echoCancellation: true,
            noiseSuppression: true,
            autoGainControl: true
          }
        });

        const mediaRecorder = new MediaRecorder(stream, {
          mimeType: MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
            ? 'audio/webm;codecs=opus'
            : 'audio/webm'
        });

        const chunks = [];

        resultPromise = new Promise((resolve) => {
          mediaRecorder.ondataavailable = (e) => {
            if (e.data.size > 0) chunks.push(e.data);
          };

          mediaRecorder.onstop = async () => {
            stream.getTracks().forEach(t => t.stop());
            const blob = new Blob(chunks, { type: mediaRecorder.mimeType });
            const arrayBuffer = await blob.arrayBuffer();
            const audioBuffer = await ctx.decodeAudioData(arrayBuffer);
            const float32Data = new Float32Array(audioBuffer.getChannelData(0));

            resolve({
              audioBuffer,
              float32Data,
              sampleRate: audioBuffer.sampleRate,
              duration: audioBuffer.duration,
              blob
            });
          };
        });

        mediaRecorder.start();
        stopFn = () => mediaRecorder.stop();
      },

      stop() {
        if (stopFn) stopFn();
        return resultPromise;
      }
    };
  }

  // ─── Decode reference audio ─────────────────────────────────

  /**
   * Decode an audio file (from URL or Blob) into analyzable format.
   */
  async function decodeAudioFromUrl(url) {
    const ctx = getAudioContext();
    const response = await fetch(url);
    const arrayBuffer = await response.arrayBuffer();
    const audioBuffer = await ctx.decodeAudioData(arrayBuffer);
    const float32Data = new Float32Array(audioBuffer.getChannelData(0));

    return {
      audioBuffer,
      float32Data,
      sampleRate: audioBuffer.sampleRate,
      duration: audioBuffer.duration
    };
  }

  // ─── Feature extraction ─────────────────────────────────────

  /**
   * Compute RMS energy per frame.
   */
  function extractEnergy(float32Data, hopSize = HOP_SIZE, frameSize = FRAME_SIZE) {
    const energy = [];
    const numFrames = Math.floor((float32Data.length - frameSize) / hopSize);

    for (let i = 0; i < numFrames; i++) {
      const start = i * hopSize;
      let sum = 0;
      for (let j = start; j < start + frameSize; j++) {
        sum += float32Data[j] * float32Data[j];
      }
      energy.push({
        time: start / 44100, // Approximate — ideally pass sampleRate
        rms: Math.sqrt(sum / frameSize)
      });
    }

    return energy;
  }

  /**
   * Compute spectral centroid per frame (brightness indicator).
   */
  function extractSpectralCentroid(float32Data, sampleRate = 44100, hopSize = HOP_SIZE, frameSize = FRAME_SIZE) {
    const centroids = [];
    const numFrames = Math.floor((float32Data.length - frameSize) / hopSize);
    const fftSize = frameSize;
    const binFreqWidth = sampleRate / fftSize;

    // We'll use a simple DFT magnitude approach
    // For performance in production, use AnalyserNode or FFT library
    for (let i = 0; i < numFrames; i++) {
      const start = i * hopSize;
      const frame = float32Data.subarray(start, start + frameSize);

      // Apply Hann window
      const windowed = new Float32Array(frameSize);
      for (let j = 0; j < frameSize; j++) {
        windowed[j] = frame[j] * 0.5 * (1 - Math.cos(2 * Math.PI * j / (frameSize - 1)));
      }

      // Compute magnitude spectrum (only positive frequencies)
      let weightedSum = 0;
      let magnitudeSum = 0;
      const halfFFT = fftSize / 2;

      for (let k = 0; k < halfFFT; k++) {
        let real = 0, imag = 0;
        for (let n = 0; n < frameSize; n++) {
          const angle = -2 * Math.PI * k * n / fftSize;
          real += windowed[n] * Math.cos(angle);
          imag += windowed[n] * Math.sin(angle);
        }
        const magnitude = Math.sqrt(real * real + imag * imag);
        const freq = k * binFreqWidth;
        weightedSum += freq * magnitude;
        magnitudeSum += magnitude;
      }

      centroids.push({
        time: start / sampleRate,
        centroid: magnitudeSum > 0 ? weightedSum / magnitudeSum : 0
      });
    }

    return centroids;
  }

  /**
   * Simple onset detection based on energy increase.
   * Returns array of onset times in seconds.
   */
  function detectOnsets(energy, threshold = 0.02) {
    const onsets = [];
    const smoothed = [];

    // Smooth energy
    for (let i = 0; i < energy.length; i++) {
      const start = Math.max(0, i - 3);
      const end = Math.min(energy.length, i + 4);
      let sum = 0;
      for (let j = start; j < end; j++) sum += energy[j].rms;
      smoothed.push(sum / (end - start));
    }

    // Find positive-going threshold crossings
    let wasBelow = true;
    for (let i = 1; i < smoothed.length; i++) {
      const diff = smoothed[i] - smoothed[i - 1];
      if (wasBelow && smoothed[i] > threshold && diff > 0) {
        onsets.push(energy[i].time);
        wasBelow = false;
      }
      if (smoothed[i] < threshold * 0.5) {
        wasBelow = true;
      }
    }

    return onsets;
  }

  /**
   * Trim silence from the beginning and end of audio.
   * Returns { start, end } in samples.
   */
  function trimSilence(float32Data, sampleRate = 44100, threshold = 0.01) {
    const frameLen = Math.floor(sampleRate * 0.01); // 10ms frames
    let start = 0;
    let end = float32Data.length;

    // Find start
    for (let i = 0; i < float32Data.length - frameLen; i += frameLen) {
      let rms = 0;
      for (let j = i; j < i + frameLen; j++) {
        rms += float32Data[j] * float32Data[j];
      }
      rms = Math.sqrt(rms / frameLen);
      if (rms > threshold) {
        start = Math.max(0, i - frameLen * 2); // Keep a little pre-onset
        break;
      }
    }

    // Find end
    for (let i = float32Data.length - frameLen; i > start; i -= frameLen) {
      let rms = 0;
      for (let j = i; j < Math.min(i + frameLen, float32Data.length); j++) {
        rms += float32Data[j] * float32Data[j];
      }
      rms = Math.sqrt(rms / frameLen);
      if (rms > threshold) {
        end = Math.min(float32Data.length, i + frameLen * 3);
        break;
      }
    }

    return { start, end, trimmedData: float32Data.subarray(start, end) };
  }

  /**
   * Full feature extraction pipeline.
   * Takes raw audio, returns all features needed for comparison.
   */
  function extractFeatures(float32Data, sampleRate = 44100) {
    // Trim silence
    const { trimmedData, start: trimStart } = trimSilence(float32Data, sampleRate);
    const timeOffset = trimStart / sampleRate;

    // Pitch contour
    const rawContour = PitchDetector.extractContour(trimmedData, sampleRate, HOP_SIZE, FRAME_SIZE);
    const smoothedContour = PitchDetector.smoothContour(rawContour);
    const semitoneContour = PitchDetector.hzToSemitones(smoothedContour);

    // Energy
    const energy = extractEnergy(trimmedData, HOP_SIZE, FRAME_SIZE);

    // Onsets (rhythm)
    const onsets = detectOnsets(energy);

    // Duration
    const duration = trimmedData.length / sampleRate;

    // Voiced ratio
    const voicedFrames = rawContour.filter(p => p.frequency !== null).length;
    const voicedRatio = rawContour.length > 0 ? voicedFrames / rawContour.length : 0;

    // Mean pitch
    const voicedPitches = rawContour.filter(p => p.frequency !== null).map(p => p.frequency);
    const meanPitch = voicedPitches.length > 0
      ? voicedPitches.reduce((a, b) => a + b, 0) / voicedPitches.length
      : 0;

    // ── Spectral features (requires SpectralAnalysis module) ──

    let mfccs = [];
    let formants = [];
    let spectralFeatures = [];
    let zcr = [];
    let hnr = [];
    let frameClassification = [];

    if (typeof SpectralAnalysis !== 'undefined') {
      // MFCCs — the most important addition for phoneme identity
      mfccs = SpectralAnalysis.extractMFCCs(trimmedData, sampleRate, HOP_SIZE, FRAME_SIZE, 13);

      // Formants (F1/F2/F3) — vowel quality
      formants = SpectralAnalysis.extractFormants(trimmedData, sampleRate, HOP_SIZE, FRAME_SIZE);

      // Spectral shape: centroid, spread, rolloff, flatness, crest, entropy, flux
      spectralFeatures = SpectralAnalysis.extractSpectralFeatures(trimmedData, sampleRate, HOP_SIZE, FRAME_SIZE);

      // Zero-crossing rate — voiced/unvoiced distinction
      zcr = SpectralAnalysis.extractZCR(trimmedData, sampleRate, HOP_SIZE, FRAME_SIZE);

      // Harmonic-to-noise ratio — voice quality
      hnr = SpectralAnalysis.extractHNR(trimmedData, sampleRate, HOP_SIZE, FRAME_SIZE);

      // Frame classification (voiced / unvoiced / silent)
      frameClassification = SpectralAnalysis.classifyFrames(energy, zcr, rawContour);
    }

    return {
      pitchContour: semitoneContour,
      pitchContourHz: smoothedContour,
      energy,
      onsets,
      duration,
      voicedRatio,
      meanPitch,
      sampleRate,
      trimmedData,
      // New spectral features
      mfccs,
      formants,
      spectralFeatures,
      zcr,
      hnr,
      frameClassification
    };
  }

  return {
    createRecorder,
    decodeAudioFromUrl,
    extractEnergy,
    extractSpectralCentroid,
    detectOnsets,
    trimSilence,
    extractFeatures,
    getAudioContext
  };

})();

if (typeof window !== 'undefined') {
  window.AudioAnalyzer = AudioAnalyzer;
}
