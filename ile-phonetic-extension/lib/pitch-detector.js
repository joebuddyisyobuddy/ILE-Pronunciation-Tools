/**
 * YIN Pitch Detection Algorithm
 * 
 * Implements the YIN algorithm for fundamental frequency (F0) estimation.
 * Optimized for voice — works well in the 80–500 Hz range typical of speech.
 * 
 * Reference: de Cheveigné & Kawahara (2002), "YIN, a fundamental frequency
 * estimator for speech and music"
 */

const PitchDetector = (() => {

  const DEFAULT_THRESHOLD = 0.15;
  const DEFAULT_SAMPLE_RATE = 44100;

  /**
   * Compute the difference function d(tau) for a frame of audio.
   */
  function difference(float32Array, yinBuffer) {
    const halfLen = yinBuffer.length;
    let delta;
    for (let tau = 0; tau < halfLen; tau++) {
      yinBuffer[tau] = 0;
      for (let i = 0; i < halfLen; i++) {
        delta = float32Array[i] - float32Array[i + tau];
        yinBuffer[tau] += delta * delta;
      }
    }
  }

  /**
   * Cumulative mean normalized difference function d'(tau).
   */
  function cumulativeMeanNormalized(yinBuffer) {
    yinBuffer[0] = 1;
    let runningSum = 0;
    for (let tau = 1; tau < yinBuffer.length; tau++) {
      runningSum += yinBuffer[tau];
      yinBuffer[tau] *= tau / runningSum;
    }
  }

  /**
   * Absolute threshold step — find the first tau where d'(tau) < threshold.
   */
  function absoluteThreshold(yinBuffer, threshold) {
    for (let tau = 2; tau < yinBuffer.length; tau++) {
      if (yinBuffer[tau] < threshold) {
        // Walk forward to find the dip (local minimum)
        while (tau + 1 < yinBuffer.length && yinBuffer[tau + 1] < yinBuffer[tau]) {
          tau++;
        }
        return tau;
      }
    }
    return -1; // No pitch detected
  }

  /**
   * Parabolic interpolation around the detected tau for sub-sample accuracy.
   */
  function parabolicInterpolation(yinBuffer, tauEstimate) {
    const x0 = tauEstimate < 1 ? tauEstimate : tauEstimate - 1;
    const x2 = tauEstimate + 1 < yinBuffer.length ? tauEstimate + 1 : tauEstimate;

    if (x0 === tauEstimate) {
      return yinBuffer[tauEstimate] <= yinBuffer[x2] ? tauEstimate : x2;
    }
    if (x2 === tauEstimate) {
      return yinBuffer[tauEstimate] <= yinBuffer[x0] ? tauEstimate : x0;
    }

    const s0 = yinBuffer[x0];
    const s1 = yinBuffer[tauEstimate];
    const s2 = yinBuffer[x2];

    return tauEstimate + (s2 - s0) / (2 * (2 * s1 - s2 - s0));
  }

  /**
   * Detect pitch of a single audio frame.
   * 
   * @param {Float32Array} frame - Audio samples (mono, length should be >= 2 * max period)
   * @param {number} sampleRate
   * @param {number} threshold - YIN threshold (lower = stricter, default 0.15)
   * @returns {{ frequency: number, confidence: number } | null}
   */
  function detectPitch(frame, sampleRate = DEFAULT_SAMPLE_RATE, threshold = DEFAULT_THRESHOLD) {
    const halfLen = Math.floor(frame.length / 2);
    const yinBuffer = new Float32Array(halfLen);

    difference(frame, yinBuffer);
    cumulativeMeanNormalized(yinBuffer);

    const tauEstimate = absoluteThreshold(yinBuffer, threshold);

    if (tauEstimate === -1) return null;

    const betterTau = parabolicInterpolation(yinBuffer, tauEstimate);
    const frequency = sampleRate / betterTau;
    const confidence = 1 - yinBuffer[tauEstimate];

    // Sanity check: human voice range
    if (frequency < 50 || frequency > 600) return null;

    return { frequency, confidence };
  }

  /**
   * Extract a pitch contour from an entire audio buffer.
   * Returns an array of { time, frequency, confidence } objects.
   * 
   * @param {Float32Array} audioData - Full mono audio signal
   * @param {number} sampleRate
   * @param {number} hopSize - Samples between frames (default 512 ≈ 11.6ms at 44.1kHz)
   * @param {number} frameSize - Analysis window size (default 2048 ≈ 46ms)
   * @param {number} threshold
   * @returns {Array<{ time: number, frequency: number|null, confidence: number }>}
   */
  function extractContour(audioData, sampleRate = DEFAULT_SAMPLE_RATE, hopSize = 512, frameSize = 2048, threshold = DEFAULT_THRESHOLD) {
    const contour = [];
    const numFrames = Math.floor((audioData.length - frameSize) / hopSize);

    for (let i = 0; i < numFrames; i++) {
      const start = i * hopSize;
      const frame = audioData.subarray(start, start + frameSize);
      const result = detectPitch(frame, sampleRate, threshold);
      contour.push({
        time: start / sampleRate,
        frequency: result ? result.frequency : null,
        confidence: result ? result.confidence : 0
      });
    }

    return contour;
  }

  /**
   * Convert a pitch contour from Hz to semitones relative to a reference.
   * Semitones are perceptually uniform — 1 semitone is the same "size" anywhere.
   */
  function hzToSemitones(contour, referenceHz = 200) {
    return contour.map(point => ({
      ...point,
      semitones: point.frequency ? 12 * Math.log2(point.frequency / referenceHz) : null
    }));
  }

  /**
   * Smooth a contour by replacing nulls with interpolated values
   * and applying a moving average.
   */
  function smoothContour(contour, windowSize = 5) {
    // First: interpolate gaps
    const interpolated = [...contour];
    for (let i = 0; i < interpolated.length; i++) {
      if (interpolated[i].frequency === null) {
        // Find prev and next voiced frames
        let prev = i - 1;
        while (prev >= 0 && interpolated[prev].frequency === null) prev--;
        let next = i + 1;
        while (next < interpolated.length && interpolated[next].frequency === null) next++;

        if (prev >= 0 && next < interpolated.length) {
          const ratio = (i - prev) / (next - prev);
          interpolated[i] = {
            ...interpolated[i],
            frequency: interpolated[prev].frequency + ratio * (interpolated[next].frequency - interpolated[prev].frequency),
            confidence: 0.3 // Mark as interpolated
          };
        }
      }
    }

    // Second: moving average on voiced frames
    const smoothed = interpolated.map((point, i) => {
      if (point.frequency === null) return point;

      const half = Math.floor(windowSize / 2);
      let sum = 0, count = 0;
      for (let j = Math.max(0, i - half); j <= Math.min(interpolated.length - 1, i + half); j++) {
        if (interpolated[j].frequency !== null) {
          sum += interpolated[j].frequency;
          count++;
        }
      }
      return { ...point, frequency: count > 0 ? sum / count : null };
    });

    return smoothed;
  }

  return {
    detectPitch,
    extractContour,
    hzToSemitones,
    smoothContour
  };

})();

// Make available globally for content script
if (typeof window !== 'undefined') {
  window.PitchDetector = PitchDetector;
}
