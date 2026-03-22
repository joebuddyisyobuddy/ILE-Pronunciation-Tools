/**
 * SpectralAnalysis
 * 
 * Client-side extraction of spectral features that capture WHAT is being said,
 * not just the melody and rhythm.
 * 
 * Features:
 *   - MFCCs (Mel-Frequency Cepstral Coefficients) — the gold standard for
 *     encoding phoneme identity. Captures the spectral envelope that distinguishes
 *     vowels, consonants, and manner of articulation.
 * 
 *   - Formants (F1, F2, F3) via LPC (Linear Predictive Coding) — maps directly
 *     to tongue height (F1) and tongue front/back position (F2). Essential for
 *     vowel quality comparison.
 * 
 *   - Spectral shape descriptors — centroid, spread, rolloff, flatness, flux.
 *     Together these capture brightness, noisiness, and spectral change over time.
 * 
 *   - Zero-crossing rate — distinguishes voiced from unvoiced sounds (e.g. "z" vs "s").
 * 
 *   - Harmonic-to-noise ratio (HNR) — voice quality measure.
 */

const SpectralAnalysis = (() => {

  // ═══════════════════════════════════════════════════════════════
  //  FFT (Radix-2 Cooley-Tukey)
  // ═══════════════════════════════════════════════════════════════

  /**
   * In-place FFT. Input arrays are modified.
   * @param {Float64Array} real
   * @param {Float64Array} imag
   */
  function fft(real, imag) {
    const n = real.length;
    if (n <= 1) return;

    // Bit-reversal permutation
    let j = 0;
    for (let i = 0; i < n - 1; i++) {
      if (i < j) {
        [real[i], real[j]] = [real[j], real[i]];
        [imag[i], imag[j]] = [imag[j], imag[i]];
      }
      let k = n >> 1;
      while (k <= j) {
        j -= k;
        k >>= 1;
      }
      j += k;
    }

    // Cooley-Tukey butterfly
    for (let len = 2; len <= n; len <<= 1) {
      const halfLen = len >> 1;
      const angle = -2 * Math.PI / len;
      const wReal = Math.cos(angle);
      const wImag = Math.sin(angle);

      for (let i = 0; i < n; i += len) {
        let curReal = 1, curImag = 0;
        for (let k = 0; k < halfLen; k++) {
          const evenIdx = i + k;
          const oddIdx = i + k + halfLen;

          const tReal = curReal * real[oddIdx] - curImag * imag[oddIdx];
          const tImag = curReal * imag[oddIdx] + curImag * real[oddIdx];

          real[oddIdx] = real[evenIdx] - tReal;
          imag[oddIdx] = imag[evenIdx] - tImag;
          real[evenIdx] += tReal;
          imag[evenIdx] += tImag;

          const newCurReal = curReal * wReal - curImag * wImag;
          curImag = curReal * wImag + curImag * wReal;
          curReal = newCurReal;
        }
      }
    }
  }

  /**
   * Compute power spectrum of a windowed frame.
   * Returns Float64Array of length frameSize/2 + 1.
   */
  function powerSpectrum(frame, frameSize) {
    const real = new Float64Array(frameSize);
    const imag = new Float64Array(frameSize);

    for (let i = 0; i < frame.length && i < frameSize; i++) {
      real[i] = frame[i];
    }

    fft(real, imag);

    const halfN = (frameSize >> 1) + 1;
    const power = new Float64Array(halfN);
    for (let i = 0; i < halfN; i++) {
      power[i] = real[i] * real[i] + imag[i] * imag[i];
    }
    return power;
  }

  /**
   * Compute magnitude spectrum.
   */
  function magnitudeSpectrum(frame, frameSize) {
    const power = powerSpectrum(frame, frameSize);
    const mag = new Float64Array(power.length);
    for (let i = 0; i < power.length; i++) {
      mag[i] = Math.sqrt(power[i]);
    }
    return mag;
  }

  // ═══════════════════════════════════════════════════════════════
  //  WINDOWING
  // ═══════════════════════════════════════════════════════════════

  function hannWindow(frame) {
    const n = frame.length;
    const windowed = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      windowed[i] = frame[i] * 0.5 * (1 - Math.cos(2 * Math.PI * i / (n - 1)));
    }
    return windowed;
  }

  function hammingWindow(frame) {
    const n = frame.length;
    const windowed = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      windowed[i] = frame[i] * (0.54 - 0.46 * Math.cos(2 * Math.PI * i / (n - 1)));
    }
    return windowed;
  }

  // ═══════════════════════════════════════════════════════════════
  //  MEL FILTERBANK + MFCCs
  // ═══════════════════════════════════════════════════════════════

  function hzToMel(hz) {
    return 2595 * Math.log10(1 + hz / 700);
  }

  function melToHz(mel) {
    return 700 * (Math.pow(10, mel / 2595) - 1);
  }

  /**
   * Build a Mel filterbank matrix.
   * @param {number} numFilters - Number of Mel filters (typically 26-40)
   * @param {number} fftSize - FFT size
   * @param {number} sampleRate
   * @param {number} lowFreq - Lower frequency bound (Hz)
   * @param {number} highFreq - Upper frequency bound (Hz)
   * @returns {Array<Float64Array>} - Array of filter weight vectors
   */
  function melFilterbank(numFilters, fftSize, sampleRate, lowFreq = 0, highFreq = null) {
    if (highFreq === null) highFreq = sampleRate / 2;

    const lowMel = hzToMel(lowFreq);
    const highMel = hzToMel(highFreq);
    const numBins = (fftSize >> 1) + 1;

    // Equally spaced points in Mel scale
    const melPoints = new Float64Array(numFilters + 2);
    for (let i = 0; i < numFilters + 2; i++) {
      melPoints[i] = lowMel + (highMel - lowMel) * i / (numFilters + 1);
    }

    // Convert to Hz then to FFT bin indices
    const binPoints = new Float64Array(numFilters + 2);
    for (let i = 0; i < numFilters + 2; i++) {
      binPoints[i] = Math.floor((fftSize + 1) * melToHz(melPoints[i]) / sampleRate);
    }

    // Build triangular filters
    const filters = [];
    for (let m = 0; m < numFilters; m++) {
      const filter = new Float64Array(numBins);
      const start = binPoints[m];
      const center = binPoints[m + 1];
      const end = binPoints[m + 2];

      for (let k = Math.floor(start); k < Math.ceil(center); k++) {
        if (k >= 0 && k < numBins && center !== start) {
          filter[k] = (k - start) / (center - start);
        }
      }
      for (let k = Math.floor(center); k < Math.ceil(end); k++) {
        if (k >= 0 && k < numBins && end !== center) {
          filter[k] = (end - k) / (end - center);
        }
      }
      filters.push(filter);
    }

    return filters;
  }

  /**
   * Extract MFCCs from a single frame.
   * 
   * @param {Float64Array|Float32Array} frame - Audio frame (already windowed)
   * @param {number} sampleRate
   * @param {number} numCoeffs - Number of MFCCs to return (typically 13)
   * @param {number} numFilters - Number of Mel filters (typically 26)
   * @param {number} fftSize - FFT size
   * @returns {Float64Array} - MFCC vector
   */
  function extractMFCCsFromFrame(frame, sampleRate, numCoeffs = 13, numFilters = 26, fftSize = 2048) {
    // Power spectrum
    const power = powerSpectrum(frame, fftSize);

    // Apply Mel filterbank
    const filters = melFilterbank(numFilters, fftSize, sampleRate);
    const melEnergies = new Float64Array(numFilters);

    for (let m = 0; m < numFilters; m++) {
      let sum = 0;
      for (let k = 0; k < power.length; k++) {
        sum += power[k] * filters[m][k];
      }
      melEnergies[m] = sum > 1e-10 ? Math.log(sum) : -23.026; // log(1e-10)
    }

    // DCT-II to get MFCCs
    const mfccs = new Float64Array(numCoeffs);
    for (let n = 0; n < numCoeffs; n++) {
      let sum = 0;
      for (let m = 0; m < numFilters; m++) {
        sum += melEnergies[m] * Math.cos(Math.PI * n * (m + 0.5) / numFilters);
      }
      mfccs[n] = sum;
    }

    return mfccs;
  }

  /**
   * Extract MFCC contour from full audio.
   * Returns array of { time, mfccs: Float64Array } per frame.
   * Also computes delta (velocity) and delta-delta (acceleration) MFCCs.
   */
  function extractMFCCs(float32Data, sampleRate, hopSize = 512, frameSize = 2048, numCoeffs = 13) {
    const frames = [];
    const numFrames = Math.floor((float32Data.length - frameSize) / hopSize);

    // Pre-compute filterbank (reuse across frames)
    const filters = melFilterbank(26, frameSize, sampleRate);

    for (let i = 0; i < numFrames; i++) {
      const start = i * hopSize;
      const rawFrame = float32Data.subarray(start, start + frameSize);

      // Convert to Float64 and apply Hamming window
      const frame = new Float64Array(frameSize);
      for (let j = 0; j < frameSize; j++) {
        frame[j] = rawFrame[j] * (0.54 - 0.46 * Math.cos(2 * Math.PI * j / (frameSize - 1)));
      }

      // Power spectrum
      const power = powerSpectrum(frame, frameSize);

      // Apply Mel filterbank
      const melEnergies = new Float64Array(26);
      for (let m = 0; m < 26; m++) {
        let sum = 0;
        for (let k = 0; k < power.length; k++) {
          sum += power[k] * filters[m][k];
        }
        melEnergies[m] = sum > 1e-10 ? Math.log(sum) : -23.026;
      }

      // DCT-II
      const mfccs = new Float64Array(numCoeffs);
      for (let n = 0; n < numCoeffs; n++) {
        let sum = 0;
        for (let m = 0; m < 26; m++) {
          sum += melEnergies[m] * Math.cos(Math.PI * n * (m + 0.5) / 26);
        }
        mfccs[n] = sum;
      }

      frames.push({
        time: start / sampleRate,
        mfccs
      });
    }

    // Compute deltas (first derivative)
    const deltaFrames = computeDeltas(frames.map(f => f.mfccs), 2);

    // Compute delta-deltas (second derivative)
    const deltaDeltaFrames = computeDeltas(deltaFrames, 2);

    // Attach deltas to frames
    for (let i = 0; i < frames.length; i++) {
      frames[i].deltaMfccs = deltaFrames[i];
      frames[i].deltaDeltaMfccs = deltaDeltaFrames[i];
    }

    return frames;
  }

  /**
   * Compute delta features using a regression window.
   */
  function computeDeltas(features, windowSize = 2) {
    const numFrames = features.length;
    const numCoeffs = features[0].length;
    const deltas = [];

    let denominator = 0;
    for (let n = 1; n <= windowSize; n++) {
      denominator += 2 * n * n;
    }

    for (let t = 0; t < numFrames; t++) {
      const delta = new Float64Array(numCoeffs);
      for (let c = 0; c < numCoeffs; c++) {
        let numerator = 0;
        for (let n = 1; n <= windowSize; n++) {
          const prev = t - n >= 0 ? features[t - n][c] : features[0][c];
          const next = t + n < numFrames ? features[t + n][c] : features[numFrames - 1][c];
          numerator += n * (next - prev);
        }
        delta[c] = denominator > 0 ? numerator / denominator : 0;
      }
      deltas.push(delta);
    }

    return deltas;
  }

  // ═══════════════════════════════════════════════════════════════
  //  FORMANT ESTIMATION via LPC (Linear Predictive Coding)
  // ═══════════════════════════════════════════════════════════════

  /**
   * Levinson-Durbin recursion for solving the LPC system.
   * @param {Float64Array} autocorr - Autocorrelation values r[0]..r[order]
   * @param {number} order - LPC order
   * @returns {Float64Array} - LPC coefficients a[1]..a[order]
   */
  function levinsonDurbin(autocorr, order) {
    const a = new Float64Array(order + 1);
    const aTemp = new Float64Array(order + 1);
    a[0] = 1;

    let error = autocorr[0];
    if (error === 0) return a;

    for (let i = 1; i <= order; i++) {
      // Compute reflection coefficient
      let lambda = 0;
      for (let j = 1; j < i; j++) {
        lambda += a[j] * autocorr[i - j];
      }
      lambda = -(autocorr[i] + lambda) / error;

      // Update coefficients
      for (let j = 0; j <= i; j++) {
        aTemp[j] = a[j];
      }
      for (let j = 1; j < i; j++) {
        a[j] = aTemp[j] + lambda * aTemp[i - j];
      }
      a[i] = lambda;

      error *= (1 - lambda * lambda);
      if (error <= 0) break;
    }

    return a;
  }

  /**
   * Compute autocorrelation of a signal.
   */
  function autocorrelation(signal, maxLag) {
    const result = new Float64Array(maxLag + 1);
    for (let lag = 0; lag <= maxLag; lag++) {
      let sum = 0;
      for (let i = 0; i < signal.length - lag; i++) {
        sum += signal[i] * signal[i + lag];
      }
      result[lag] = sum;
    }
    return result;
  }

  /**
   * Find roots of LPC polynomial using Durand-Kerner method.
   * Returns complex roots as [{ re, im }, ...].
   */
  function findLPCRoots(coeffs, maxIter = 100) {
    const order = coeffs.length - 1;
    if (order < 1) return [];

    // Initial guesses: evenly spaced on unit circle with small perturbation
    const roots = [];
    for (let i = 0; i < order; i++) {
      const angle = 2 * Math.PI * i / order + 0.01;
      const r = 0.9 + 0.05 * Math.random();
      roots.push({ re: r * Math.cos(angle), im: r * Math.sin(angle) });
    }

    // Durand-Kerner iteration
    for (let iter = 0; iter < maxIter; iter++) {
      let maxChange = 0;

      for (let i = 0; i < order; i++) {
        // Evaluate polynomial at root[i]
        let pRe = coeffs[0], pIm = 0;
        let zRe = 1, zIm = 0;

        for (let k = 1; k <= order; k++) {
          const newZRe = zRe * roots[i].re - zIm * roots[i].im;
          const newZIm = zRe * roots[i].im + zIm * roots[i].re;
          zRe = newZRe;
          zIm = newZIm;
          pRe += coeffs[k] * zRe;
          pIm += coeffs[k] * zIm;
        }

        // Compute product of (root[i] - root[j]) for j != i
        let dRe = 1, dIm = 0;
        for (let j = 0; j < order; j++) {
          if (j === i) continue;
          const diffRe = roots[i].re - roots[j].re;
          const diffIm = roots[i].im - roots[j].im;
          const newDRe = dRe * diffRe - dIm * diffIm;
          const newDIm = dRe * diffIm + dIm * diffRe;
          dRe = newDRe;
          dIm = newDIm;
        }

        // correction = p(z_i) / product
        const dMag2 = dRe * dRe + dIm * dIm;
        if (dMag2 < 1e-30) continue;

        const corrRe = (pRe * dRe + pIm * dIm) / dMag2;
        const corrIm = (pIm * dRe - pRe * dIm) / dMag2;

        roots[i].re -= corrRe;
        roots[i].im -= corrIm;

        maxChange = Math.max(maxChange, Math.sqrt(corrRe * corrRe + corrIm * corrIm));
      }

      if (maxChange < 1e-8) break;
    }

    return roots;
  }

  /**
   * Extract formant frequencies from LPC roots.
   * Formants are the frequencies of complex root pairs with positive imaginary part,
   * sorted by frequency.
   */
  function rootsToFormants(roots, sampleRate) {
    const formants = [];

    for (const root of roots) {
      if (root.im < 0) continue; // Only take positive-imaginary roots

      const freq = Math.abs(Math.atan2(root.im, root.re)) * sampleRate / (2 * Math.PI);
      const mag = Math.sqrt(root.re * root.re + root.im * root.im);
      const bandwidth = -0.5 * sampleRate * Math.log(mag) / Math.PI;

      // Filter: valid formants are 50-5500 Hz with reasonable bandwidth
      if (freq > 50 && freq < 5500 && bandwidth < 500 && bandwidth > 0) {
        formants.push({ frequency: freq, bandwidth });
      }
    }

    // Sort by frequency
    formants.sort((a, b) => a.frequency - b.frequency);

    return formants;
  }

  /**
   * Extract formants (F1, F2, F3) from a single audio frame.
   * Uses LPC analysis at order appropriate for the sample rate.
   */
  function extractFormantsFromFrame(frame, sampleRate) {
    // Pre-emphasis to boost high frequencies
    const preEmph = new Float64Array(frame.length);
    preEmph[0] = frame[0];
    for (let i = 1; i < frame.length; i++) {
      preEmph[i] = frame[i] - 0.97 * frame[i - 1];
    }

    // Apply Hamming window
    const windowed = hammingWindow(preEmph);

    // LPC order: rule of thumb is sampleRate/1000 + 2
    const order = Math.min(Math.round(sampleRate / 1000) + 2, 30);

    // Autocorrelation
    const ac = autocorrelation(windowed, order);

    // Solve for LPC coefficients
    const lpcCoeffs = levinsonDurbin(ac, order);

    // Find roots of LPC polynomial
    const roots = findLPCRoots(lpcCoeffs);

    // Convert to formant frequencies
    const formants = rootsToFormants(roots, sampleRate);

    return {
      F1: formants[0] ? formants[0].frequency : null,
      F2: formants[1] ? formants[1].frequency : null,
      F3: formants[2] ? formants[2].frequency : null,
      F4: formants[3] ? formants[3].frequency : null,
      allFormants: formants,
      lpcCoeffs
    };
  }

  /**
   * Extract formant contour from full audio.
   */
  function extractFormants(float32Data, sampleRate, hopSize = 512, frameSize = 2048) {
    const frames = [];
    const numFrames = Math.floor((float32Data.length - frameSize) / hopSize);

    for (let i = 0; i < numFrames; i++) {
      const start = i * hopSize;
      const rawFrame = float32Data.subarray(start, start + frameSize);

      const frame = new Float64Array(frameSize);
      for (let j = 0; j < frameSize; j++) {
        frame[j] = rawFrame[j];
      }

      const formants = extractFormantsFromFrame(frame, sampleRate);

      frames.push({
        time: start / sampleRate,
        ...formants
      });
    }

    return frames;
  }

  // ═══════════════════════════════════════════════════════════════
  //  SPECTRAL SHAPE DESCRIPTORS
  // ═══════════════════════════════════════════════════════════════

  /**
   * Extract all spectral shape features from a single frame.
   */
  function spectralFeaturesFromFrame(frame, sampleRate, fftSize) {
    const windowed = hannWindow(new Float64Array(frame));
    const mag = magnitudeSpectrum(windowed, fftSize);
    const binWidth = sampleRate / fftSize;

    let magSum = 0;
    for (let i = 0; i < mag.length; i++) magSum += mag[i];
    if (magSum < 1e-10) {
      return { centroid: 0, spread: 0, rolloff: 0, flatness: 0, crest: 0, entropy: 0 };
    }

    // Spectral centroid (center of mass)
    let centroid = 0;
    for (let i = 0; i < mag.length; i++) {
      centroid += (i * binWidth) * mag[i];
    }
    centroid /= magSum;

    // Spectral spread (standard deviation around centroid)
    let spread = 0;
    for (let i = 0; i < mag.length; i++) {
      const diff = (i * binWidth) - centroid;
      spread += diff * diff * mag[i];
    }
    spread = Math.sqrt(spread / magSum);

    // Spectral rolloff (frequency below which 85% of energy sits)
    const rolloffThreshold = 0.85 * magSum;
    let cumSum = 0;
    let rolloff = 0;
    for (let i = 0; i < mag.length; i++) {
      cumSum += mag[i];
      if (cumSum >= rolloffThreshold) {
        rolloff = i * binWidth;
        break;
      }
    }

    // Spectral flatness (geometric mean / arithmetic mean)
    // High = noise-like, Low = tonal
    let logSum = 0;
    let nonZeroCount = 0;
    for (let i = 1; i < mag.length; i++) {
      if (mag[i] > 1e-10) {
        logSum += Math.log(mag[i]);
        nonZeroCount++;
      }
    }
    const geometricMean = nonZeroCount > 0 ? Math.exp(logSum / nonZeroCount) : 0;
    const arithmeticMean = magSum / mag.length;
    const flatness = arithmeticMean > 0 ? geometricMean / arithmeticMean : 0;

    // Spectral crest (peak / arithmetic mean) — high = tonal
    let maxMag = 0;
    for (let i = 0; i < mag.length; i++) {
      if (mag[i] > maxMag) maxMag = mag[i];
    }
    const crest = arithmeticMean > 0 ? maxMag / arithmeticMean : 0;

    // Spectral entropy (Shannon entropy of normalized spectrum)
    let entropy = 0;
    for (let i = 0; i < mag.length; i++) {
      const p = mag[i] / magSum;
      if (p > 1e-10) {
        entropy -= p * Math.log2(p);
      }
    }

    return { centroid, spread, rolloff, flatness, crest, entropy };
  }

  /**
   * Extract spectral features for all frames.
   */
  function extractSpectralFeatures(float32Data, sampleRate, hopSize = 512, frameSize = 2048) {
    const frames = [];
    const numFrames = Math.floor((float32Data.length - frameSize) / hopSize);

    for (let i = 0; i < numFrames; i++) {
      const start = i * hopSize;
      const frame = float32Data.subarray(start, start + frameSize);
      const features = spectralFeaturesFromFrame(frame, sampleRate, frameSize);

      frames.push({
        time: start / sampleRate,
        ...features
      });
    }

    // Spectral flux (frame-to-frame change)
    for (let i = 0; i < frames.length; i++) {
      if (i === 0) {
        frames[i].flux = 0;
      } else {
        // Use centroid difference as a proxy for full spectral flux
        frames[i].flux = Math.abs(frames[i].centroid - frames[i - 1].centroid);
      }
    }

    return frames;
  }

  // ═══════════════════════════════════════════════════════════════
  //  ZERO-CROSSING RATE
  // ═══════════════════════════════════════════════════════════════

  /**
   * Compute zero-crossing rate per frame.
   * High ZCR = fricatives/unvoiced; Low ZCR = vowels/voiced.
   */
  function extractZCR(float32Data, sampleRate, hopSize = 512, frameSize = 2048) {
    const frames = [];
    const numFrames = Math.floor((float32Data.length - frameSize) / hopSize);

    for (let i = 0; i < numFrames; i++) {
      const start = i * hopSize;
      let crossings = 0;

      for (let j = start + 1; j < start + frameSize && j < float32Data.length; j++) {
        if ((float32Data[j] >= 0) !== (float32Data[j - 1] >= 0)) {
          crossings++;
        }
      }

      frames.push({
        time: start / sampleRate,
        zcr: crossings / frameSize
      });
    }

    return frames;
  }

  // ═══════════════════════════════════════════════════════════════
  //  HARMONIC-TO-NOISE RATIO
  // ═══════════════════════════════════════════════════════════════

  /**
   * Estimate HNR from autocorrelation.
   * Higher HNR = cleaner voice, lower = breathier/noisier.
   */
  function extractHNR(float32Data, sampleRate, hopSize = 512, frameSize = 2048) {
    const frames = [];
    const numFrames = Math.floor((float32Data.length - frameSize) / hopSize);

    // Search range for pitch period (80-500 Hz)
    const minLag = Math.floor(sampleRate / 500);
    const maxLag = Math.floor(sampleRate / 80);

    for (let i = 0; i < numFrames; i++) {
      const start = i * hopSize;
      const frame = float32Data.subarray(start, start + frameSize);

      // Autocorrelation at lag 0
      let r0 = 0;
      for (let j = 0; j < frameSize; j++) {
        r0 += frame[j] * frame[j];
      }

      if (r0 < 1e-10) {
        frames.push({ time: start / sampleRate, hnr: 0 });
        continue;
      }

      // Find maximum autocorrelation in pitch range
      let maxR = 0;
      for (let lag = minLag; lag <= maxLag && lag < frameSize; lag++) {
        let r = 0;
        for (let j = 0; j < frameSize - lag; j++) {
          r += frame[j] * frame[j + lag];
        }
        if (r > maxR) maxR = r;
      }

      // HNR in dB
      const harmonicPower = maxR;
      const noisePower = r0 - maxR;
      const hnr = noisePower > 1e-10 ? 10 * Math.log10(harmonicPower / noisePower) : 40;

      frames.push({
        time: start / sampleRate,
        hnr: Math.max(-10, Math.min(40, hnr)) // Clamp to reasonable range
      });
    }

    return frames;
  }

  // ═══════════════════════════════════════════════════════════════
  //  VOICED/UNVOICED SEGMENTATION
  // ═══════════════════════════════════════════════════════════════

  /**
   * Classify each frame as voiced, unvoiced, or silent.
   * Uses energy + ZCR + pitch to make the determination.
   */
  function classifyFrames(energy, zcr, pitchContour) {
    const numFrames = Math.min(energy.length, zcr.length, pitchContour.length);
    const classification = [];

    for (let i = 0; i < numFrames; i++) {
      const rms = energy[i].rms;
      const zcrVal = zcr[i].zcr;
      const hasPitch = pitchContour[i].frequency !== null;

      let type;
      if (rms < 0.005) {
        type = 'silent';
      } else if (hasPitch && zcrVal < 0.15) {
        type = 'voiced';       // Vowels, nasals, voiced stops
      } else if (!hasPitch && zcrVal > 0.1) {
        type = 'unvoiced';     // Fricatives like s, f, th
      } else if (hasPitch) {
        type = 'voiced';
      } else {
        type = 'unvoiced';
      }

      classification.push({
        time: energy[i].time,
        type,
        rms,
        zcr: zcrVal,
        hasPitch
      });
    }

    return classification;
  }

  // ═══════════════════════════════════════════════════════════════
  //  PUBLIC API
  // ═══════════════════════════════════════════════════════════════

  return {
    // Core DSP
    fft,
    powerSpectrum,
    magnitudeSpectrum,
    hannWindow,
    hammingWindow,

    // MFCCs
    extractMFCCs,
    extractMFCCsFromFrame,
    melFilterbank,
    computeDeltas,

    // Formants
    extractFormants,
    extractFormantsFromFrame,
    levinsonDurbin,

    // Spectral shape
    extractSpectralFeatures,
    spectralFeaturesFromFrame,

    // ZCR
    extractZCR,

    // HNR
    extractHNR,

    // Segmentation
    classifyFrames,

    // Utility
    hzToMel,
    melToHz
  };

})();

if (typeof window !== 'undefined') {
  window.SpectralAnalysis = SpectralAnalysis;
}
