/**
 * PhoneticComparison
 * 
 * Compares extracted features between a reference and learner recording.
 * 
 * Comparison dimensions:
 *   1. MFCCs (segmental — what phonemes are being produced) ★ most important
 *   2. Formants F1/F2 (segmental — vowel quality)
 *   3. Voiced/unvoiced pattern (segmental — e.g. "s" vs "z")
 *   4. Pitch contour (prosody — intonation)
 *   5. Rhythm / timing (prosody — stress and duration)
 *   6. Energy / stress (prosody — emphasis placement)
 *   7. Spectral shape (segmental — fricative/stop quality)
 *   8. Voice quality / HNR
 */

const PhoneticComparison = (() => {

  // ─── Dynamic Time Warping ───────────────────────────────────

  function dtw(seq1, seq2, distFn) {
    const n = seq1.length;
    const m = seq2.length;
    if (n === 0 || m === 0) return { path: [], cost: Infinity };

    const cost = new Float32Array(n * m);
    const idx = (i, j) => i * m + j;

    cost[idx(0, 0)] = distFn(seq1[0], seq2[0]);
    for (let i = 1; i < n; i++) cost[idx(i, 0)] = cost[idx(i - 1, 0)] + distFn(seq1[i], seq2[0]);
    for (let j = 1; j < m; j++) cost[idx(0, j)] = cost[idx(0, j - 1)] + distFn(seq1[0], seq2[j]);
    for (let i = 1; i < n; i++) {
      for (let j = 1; j < m; j++) {
        cost[idx(i, j)] = distFn(seq1[i], seq2[j]) + Math.min(
          cost[idx(i - 1, j)], cost[idx(i, j - 1)], cost[idx(i - 1, j - 1)]
        );
      }
    }

    const path = [];
    let i = n - 1, j = m - 1;
    path.push([i, j]);
    while (i > 0 || j > 0) {
      if (i === 0) { j--; }
      else if (j === 0) { i--; }
      else {
        const diag = cost[idx(i - 1, j - 1)];
        const left = cost[idx(i, j - 1)];
        const up = cost[idx(i - 1, j)];
        const min = Math.min(diag, left, up);
        if (min === diag) { i--; j--; }
        else if (min === up) { i--; }
        else { j--; }
      }
      path.push([i, j]);
    }
    path.reverse();
    return { path, cost: cost[idx(n - 1, m - 1)] / path.length };
  }

  // ─── Distance Functions ─────────────────────────────────────

  function pitchDistance(a, b) {
    const freqA = a.semitones ?? a.frequency;
    const freqB = b.semitones ?? b.frequency;
    if (freqA === null && freqB === null) return 0;
    if (freqA === null || freqB === null) return 3;
    return Math.abs(freqA - freqB);
  }

  function energyDistance(a, b) { return Math.abs(a.rms - b.rms); }

  /**
   * Normalize MFCC sequence: subtract mean, divide by std per coefficient.
   * This removes speaker/microphone differences, keeping only phonetic content.
   */
  function normalizeMFCCs(mfccFrames) {
    if (!mfccFrames.length || !mfccFrames[0].mfccs) return mfccFrames;

    const numCoeffs = mfccFrames[0].mfccs.length;
    const numFrames = mfccFrames.length;

    // Compute mean and std per coefficient
    const means = new Float64Array(numCoeffs);
    const stds = new Float64Array(numCoeffs);

    for (let c = 0; c < numCoeffs; c++) {
      let sum = 0;
      for (let f = 0; f < numFrames; f++) sum += mfccFrames[f].mfccs[c];
      means[c] = sum / numFrames;

      let varSum = 0;
      for (let f = 0; f < numFrames; f++) {
        const diff = mfccFrames[f].mfccs[c] - means[c];
        varSum += diff * diff;
      }
      stds[c] = Math.sqrt(varSum / numFrames) || 1;
    }

    // Return normalized copies
    return mfccFrames.map(frame => {
      const normalized = new Float64Array(numCoeffs);
      for (let c = 0; c < numCoeffs; c++) {
        normalized[c] = (frame.mfccs[c] - means[c]) / stds[c];
      }
      return { ...frame, mfccs: normalized };
    });
  }

  function mfccDistance(a, b) {
    if (!a.mfccs || !b.mfccs) return 0;
    let sum = 0;
    const len = Math.min(a.mfccs.length, b.mfccs.length, 13);
    // Skip c0 (energy), compare c1-c12
    for (let i = 1; i < len; i++) {
      const diff = a.mfccs[i] - b.mfccs[i];
      sum += diff * diff;
    }
    return Math.sqrt(sum);
  }

  function mfccFullDistance(a, b) {
    return mfccDistance(a, b);
  }

  function formantDistance(a, b) {
    if (!a.F1 && !b.F1) return 0;
    if (!a.F1 || !b.F1) return 200;
    let dist = 0;
    if (a.F1 && b.F1) dist += Math.pow((a.F1 - b.F1) / 100, 2);
    if (a.F2 && b.F2) dist += Math.pow((a.F2 - b.F2) / 200, 2);
    if (a.F3 && b.F3) dist += 0.5 * Math.pow((a.F3 - b.F3) / 200, 2);
    return Math.sqrt(dist);
  }

  function spectralDistance(a, b) {
    if (!a.centroid && !b.centroid) return 0;
    let d = 0;
    d += Math.pow((a.centroid - b.centroid) / 500, 2);
    d += Math.pow((a.spread - b.spread) / 500, 2);
    d += Math.pow((a.rolloff - b.rolloff) / 1000, 2);
    d += Math.pow((a.flatness - b.flatness) * 5, 2);
    return Math.sqrt(d);
  }

  // ─── Individual Comparisons ─────────────────────────────────

  function comparePitch(refFeatures, learnerFeatures) {
    const rc = refFeatures.pitchContour, lc = learnerFeatures.pitchContour;
    if (!rc.length || !lc.length) return { aligned: [], meanDeviation: Infinity, maxDeviation: Infinity };
    const { path, cost } = dtw(rc, lc, pitchDistance);
    const aligned = path.map(([ri, li]) => {
      let deviation = null;
      if (rc[ri].semitones !== null && lc[li].semitones !== null) deviation = lc[li].semitones - rc[ri].semitones;
      return { refTime: rc[ri].time, learnerTime: lc[li].time, refPitch: rc[ri].frequency, learnerPitch: lc[li].frequency, refSemitones: rc[ri].semitones, learnerSemitones: lc[li].semitones, deviation };
    });
    const devs = aligned.filter(a => a.deviation !== null).map(a => Math.abs(a.deviation));
    return { aligned, meanDeviation: devs.length ? devs.reduce((a, b) => a + b, 0) / devs.length : Infinity, maxDeviation: devs.length ? Math.max(...devs) : Infinity, dtwCost: cost };
  }

  function compareMFCCs(refFeatures, learnerFeatures) {
    const rm = refFeatures.mfccs, lm = learnerFeatures.mfccs;
    if (!rm.length || !lm.length) return { aligned: [], meanDistance: Infinity, score: 0, problemRegions: [] };

    const { path, cost } = dtw(rm, lm, mfccFullDistance);
    const aligned = path.map(([ri, li]) => ({
      refTime: rm[ri].time, learnerTime: lm[li].time,
      distance: mfccFullDistance(rm[ri], lm[li]),
      refMfccs: rm[ri].mfccs, learnerMfccs: lm[li].mfccs
    }));
    const distances = aligned.map(a => a.distance);
    const meanDist = distances.reduce((a, b) => a + b, 0) / distances.length;

    // Absolute thresholds for MFCC distance
    // These are calibrated: same speaker same sentence ≈ 5-15, different pronunciation ≈ 25-60+
    const GOOD_THRESHOLD = 15;      // Below this = good match
    const PROBLEM_THRESHOLD = 30;   // Above this = clear problem
    const MAX_REASONABLE = 60;      // Normalization ceiling

    // Score: based on what fraction of frames are below the good threshold
    // weighted by how far the bad frames are from good
    let scoreSum = 0;
    for (const d of distances) {
      if (d <= GOOD_THRESHOLD) {
        scoreSum += 1;
      } else if (d <= PROBLEM_THRESHOLD) {
        scoreSum += 1 - (d - GOOD_THRESHOLD) / (PROBLEM_THRESHOLD - GOOD_THRESHOLD);
      }
      // Above PROBLEM_THRESHOLD: contributes 0
    }
    const score = Math.round((scoreSum / distances.length) * 100);

    // Find problem regions (consecutive frames above threshold)
    const problemRegions = [];
    let inProblem = false, problemStart = 0;
    for (let i = 0; i < aligned.length; i++) {
      if (aligned[i].distance > PROBLEM_THRESHOLD && !inProblem) { inProblem = true; problemStart = i; }
      else if ((aligned[i].distance <= PROBLEM_THRESHOLD || i === aligned.length - 1) && inProblem) {
        inProblem = false;
        const region = aligned.slice(problemStart, i);
        if (region.length >= 3) {
          problemRegions.push({
            startTime: region[0].refTime, endTime: region[region.length - 1].refTime,
            learnerStartTime: region[0].learnerTime, learnerEndTime: region[region.length - 1].learnerTime,
            meanDistance: region.reduce((s, r) => s + r.distance, 0) / region.length, numFrames: region.length
          });
        }
      }
    }

    console.log('[IPC] MFCC comparison:', { 
      meanDist: meanDist.toFixed(2), 
      minDist: Math.min(...distances).toFixed(2),
      maxDist: Math.max(...distances).toFixed(2),
      score, 
      problemRegions: problemRegions.length,
      thresholds: { good: GOOD_THRESHOLD, problem: PROBLEM_THRESHOLD }
    });

    return { aligned, meanDistance: meanDist, dtwCost: cost, score, problemRegions };
  }

  function compareFormants(refFeatures, learnerFeatures) {
    const rf = refFeatures.formants, lf = learnerFeatures.formants;
    if (!rf.length || !lf.length) return { aligned: [], meanF1Diff: 0, meanF2Diff: 0, vowelSpaceRef: [], vowelSpaceLearner: [] };
    const { path, cost } = dtw(rf, lf, formantDistance);
    const aligned = path.map(([ri, li]) => ({
      refTime: rf[ri].time, learnerTime: lf[li].time,
      refF1: rf[ri].F1, refF2: rf[ri].F2, learnerF1: lf[li].F1, learnerF2: lf[li].F2,
      f1Diff: (rf[ri].F1 && lf[li].F1) ? lf[li].F1 - rf[ri].F1 : null,
      f2Diff: (rf[ri].F2 && lf[li].F2) ? lf[li].F2 - rf[ri].F2 : null,
    }));
    const f1d = aligned.filter(a => a.f1Diff !== null).map(a => Math.abs(a.f1Diff));
    const f2d = aligned.filter(a => a.f2Diff !== null).map(a => Math.abs(a.f2Diff));
    const meanF1Diff = f1d.length ? f1d.reduce((a, b) => a + b, 0) / f1d.length : 0;
    const meanF2Diff = f2d.length ? f2d.reduce((a, b) => a + b, 0) / f2d.length : 0;
    const vowelSpaceRef = rf.filter(f => f.F1 && f.F2 && f.F1 > 200 && f.F1 < 1000 && f.F2 > 500 && f.F2 < 3000).map(f => ({ F1: f.F1, F2: f.F2, time: f.time }));
    const vowelSpaceLearner = lf.filter(f => f.F1 && f.F2 && f.F1 > 200 && f.F1 < 1000 && f.F2 > 500 && f.F2 < 3000).map(f => ({ F1: f.F1, F2: f.F2, time: f.time }));
    return { aligned, meanF1Diff, meanF2Diff, vowelSpaceRef, vowelSpaceLearner, dtwCost: cost };
  }

  function compareSpectral(refFeatures, learnerFeatures) {
    const rs = refFeatures.spectralFeatures, ls = learnerFeatures.spectralFeatures;
    if (!rs.length || !ls.length) return { aligned: [], dtwCost: 0 };
    const { path, cost } = dtw(rs, ls, spectralDistance);
    const aligned = path.map(([ri, li]) => ({
      refTime: rs[ri].time, learnerTime: ls[li].time,
      refCentroid: rs[ri].centroid, learnerCentroid: ls[li].centroid,
      refFlatness: rs[ri].flatness, learnerFlatness: ls[li].flatness,
      centroidDiff: ls[li].centroid - rs[ri].centroid, flatnessDiff: ls[li].flatness - rs[ri].flatness
    }));
    return { aligned, dtwCost: cost };
  }

  function compareVoicing(refFeatures, learnerFeatures) {
    const rc = refFeatures.frameClassification, lc = learnerFeatures.frameClassification;
    if (!rc.length || !lc.length) return { matchRate: 1, mismatches: [] };
    const { path } = dtw(rc, lc, (a, b) => a.type === b.type ? 0 : 1);
    let matches = 0;
    const mismatches = [];
    for (const [ri, li] of path) {
      if (rc[ri].type === lc[li].type) matches++;
      else if (rc[ri].type !== 'silent' && lc[li].type !== 'silent')
        mismatches.push({ refTime: rc[ri].time, learnerTime: lc[li].time, expected: rc[ri].type, actual: lc[li].type });
    }
    return { matchRate: path.length > 0 ? matches / path.length : 1, mismatches, totalFrames: path.length };
  }

  function compareRhythm(refFeatures, learnerFeatures) {
    const ro = refFeatures.onsets, lo = learnerFeatures.onsets;
    const ri = [], li = [];
    for (let i = 1; i < ro.length; i++) ri.push(ro[i] - ro[i - 1]);
    for (let i = 1; i < lo.length; i++) li.push(lo[i] - lo[i - 1]);
    const rt = ri.reduce((a, b) => a + b, 0) || 1;
    const lt = li.reduce((a, b) => a + b, 0) || 1;
    const rn = ri.map(x => x / rt), ln = li.map(x => x / lt);
    let rhythmScore = 0;
    if (rn.length > 0 && ln.length > 0) {
      const { cost } = dtw(rn.map(v => ({ v })), ln.map(v => ({ v })), (a, b) => Math.abs(a.v - b.v));
      rhythmScore = cost;
    }
    return { refOnsets: ro, learnerOnsets: lo, refIOIs: ri, learnerIOIs: li, refNormIOIs: rn, learnerNormIOIs: ln, rhythmScore, onsetCountDiff: lo.length - ro.length };
  }

  function compareDuration(refFeatures, learnerFeatures) {
    const ratio = learnerFeatures.duration / refFeatures.duration;
    return { refDuration: refFeatures.duration, learnerDuration: learnerFeatures.duration, ratio, percentDiff: (ratio - 1) * 100, feedback: ratio > 1.3 ? 'too_slow' : ratio < 0.7 ? 'too_fast' : 'good' };
  }

  function compareEnergy(refFeatures, learnerFeatures) {
    const re = refFeatures.energy, le = learnerFeatures.energy;
    if (!re.length || !le.length) return { aligned: [], stressScore: Infinity };
    const { path, cost } = dtw(re, le, energyDistance);
    const aligned = path.map(([ri, li]) => ({ refTime: re[ri].time, learnerTime: le[li].time, refRms: re[ri].rms, learnerRms: le[li].rms, deviation: le[li].rms - re[ri].rms }));
    return { aligned, stressScore: cost };
  }

  function compareHNR(refFeatures, learnerFeatures) {
    const rh = refFeatures.hnr, lh = learnerFeatures.hnr;
    if (!rh.length || !lh.length) return { refMean: 0, learnerMean: 0, diff: 0 };
    const rm = rh.reduce((s, f) => s + f.hnr, 0) / rh.length;
    const lm = lh.reduce((s, f) => s + f.hnr, 0) / lh.length;
    return { refMean: rm, learnerMean: lm, diff: lm - rm };
  }

  // ─── Full Pipeline ──────────────────────────────────────────

  function compare(refFeatures, learnerFeatures) {
    const pitch = comparePitch(refFeatures, learnerFeatures);
    const rhythm = compareRhythm(refFeatures, learnerFeatures);
    const duration = compareDuration(refFeatures, learnerFeatures);
    const energy = compareEnergy(refFeatures, learnerFeatures);
    const mfcc = compareMFCCs(refFeatures, learnerFeatures);
    const formant = compareFormants(refFeatures, learnerFeatures);
    const spectral = compareSpectral(refFeatures, learnerFeatures);
    const voicing = compareVoicing(refFeatures, learnerFeatures);
    const hnr = compareHNR(refFeatures, learnerFeatures);

    const assessments = [];

    // Sound accuracy (MFCC)
    if (mfcc.score >= 85) {
      assessments.push({ dimension: 'sound accuracy', level: 'good', priority: 1, message: 'Your sounds closely match the reference.' });
    } else if (mfcc.score >= 65) {
      const n = mfcc.problemRegions.length;
      assessments.push({ dimension: 'sound accuracy', level: 'fair', priority: 1, message: `Some sounds differ${n > 0 ? ` — ${n} region${n > 1 ? 's' : ''} need attention` : ''}. Listen and try again.` });
    } else {
      assessments.push({ dimension: 'sound accuracy', level: 'needs_work', priority: 1, message: 'Significant differences from the reference. Listen again and focus on matching each syllable.' });
    }

    // Vowel quality (formants)
    if (formant.meanF1Diff < 50 && formant.meanF2Diff < 100) {
      assessments.push({ dimension: 'vowel quality', level: 'good', priority: 2, message: 'Vowel sounds are well-matched.' });
    } else if (formant.meanF1Diff < 100 && formant.meanF2Diff < 200) {
      const tip = formant.meanF1Diff > formant.meanF2Diff ? 'Try opening or closing your mouth more.' : 'Try moving your tongue further forward or back.';
      assessments.push({ dimension: 'vowel quality', level: 'fair', priority: 2, message: `Some vowels need adjustment — ${tip}` });
    } else if (formant.meanF1Diff > 0 || formant.meanF2Diff > 0) {
      assessments.push({ dimension: 'vowel quality', level: 'needs_work', priority: 2, message: 'Vowel sounds differ noticeably. Focus on mouth shape — see the vowel space chart below.' });
    }

    // Intonation (pitch)
    if (pitch.meanDeviation < 1.5) {
      assessments.push({ dimension: 'intonation', level: 'good', priority: 3, message: 'Pitch contour closely matches the reference.' });
    } else if (pitch.meanDeviation < 3) {
      assessments.push({ dimension: 'intonation', level: 'fair', priority: 3, message: 'Pitch pattern is close but drifts in places.' });
    } else {
      assessments.push({ dimension: 'intonation', level: 'needs_work', priority: 3, message: 'Pitch contour differs significantly. Listen to the melody of the sentence.' });
    }

    // Voicing
    if (voicing.matchRate <= 0.9) {
      assessments.push({ dimension: 'voicing', level: voicing.matchRate > 0.75 ? 'fair' : 'needs_work', priority: 4, message: 'Some voiced/unvoiced sounds may be swapped. Pay attention to vocal cord vibration.' });
    }

    // Tempo
    if (duration.feedback === 'good') {
      assessments.push({ dimension: 'tempo', level: 'good', priority: 5, message: 'Speaking pace matches the reference.' });
    } else if (duration.feedback === 'too_slow') {
      assessments.push({ dimension: 'tempo', level: 'fair', priority: 5, message: `Speaking ${Math.round(duration.percentDiff)}% slower than reference.` });
    } else {
      assessments.push({ dimension: 'tempo', level: 'fair', priority: 5, message: `Speaking ${Math.round(Math.abs(duration.percentDiff))}% faster than reference.` });
    }

    // Rhythm
    if (rhythm.rhythmScore < 0.1) {
      assessments.push({ dimension: 'rhythm', level: 'good', priority: 6, message: 'Rhythm and stress pattern match well.' });
    } else if (rhythm.rhythmScore < 0.25) {
      assessments.push({ dimension: 'rhythm', level: 'fair', priority: 6, message: 'Rhythm is close. Some syllable durations differ.' });
    } else {
      assessments.push({ dimension: 'rhythm', level: 'needs_work', priority: 6, message: 'Rhythm differs. Focus on which syllables are long vs. short.' });
    }

    // Voice quality
    if (hnr.diff < -5) {
      assessments.push({ dimension: 'voice quality', level: 'fair', priority: 7, message: 'Voice sounds breathier than reference. Try a clearer tone.' });
    }

    assessments.sort((a, b) => a.priority - b.priority);

    return { pitch, rhythm, duration, energy, mfcc, formant, spectral, voicing, hnr, assessments };
  }

  return { dtw, comparePitch, compareMFCCs, compareFormants, compareSpectral, compareVoicing, compareRhythm, compareDuration, compareEnergy, compareHNR, compare };

})();

if (typeof window !== 'undefined') {
  window.PhoneticComparison = PhoneticComparison;
}
