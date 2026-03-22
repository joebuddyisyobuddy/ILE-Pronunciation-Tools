/**
 * ILE Phonetic Coach — Content Script
 * 
 * Injected into the ILE website. Detects the current slide's source text
 * and audio, presents the recording/analysis UI, and renders feedback.
 * 
 * Page integration is configurable via SELECTORS — update these to match
 * the actual ILE DOM structure.
 */

(() => {
  'use strict';

  // ═══════════════════════════════════════════════════════════════
  //  CONFIGURATION — Update these selectors to match ILE's DOM
  // ═══════════════════════════════════════════════════════════════

  const CONFIG = {
    // CSS selectors for finding elements on the ILE page
    selectors: {
      // The element containing the source/reference text for the current slide
      sourceText: [
        '[data-source-text]',
        '.source-text',
        '.slide-text',
        '.sentence-text',
        '.reference-text',
        '.card-front .text',
        '.lesson-sentence',
        'h2.prompt',
      ],

      // The audio element (or source within it) for the reference recording
      sourceAudio: [
        'audio[data-reference]',
        'audio.reference-audio',
        'audio.source-audio',
        'audio#reference',
        '.slide audio',
        '.card audio',
        'audio',
      ],

      // Container that changes when the slide/card changes (for MutationObserver)
      slideContainer: [
        '.slide-container',
        '.lesson-container',
        '.card-container',
        '.swiper-slide-active',
        '[data-slide]',
        'main',
        '#app',
      ],
    },

    // Minimum audio duration to accept (seconds)
    minRecordingDuration: 0.5,

    // Maximum audio duration (seconds)
    maxRecordingDuration: 15,
  };

  // ═══════════════════════════════════════════════════════════════
  //  STATE
  // ═══════════════════════════════════════════════════════════════

  const state = {
    isRecording: false,
    recorder: null,
    referenceAudioUrl: null,
    referenceFeatures: null,
    learnerResult: null,
    comparisonResult: null,
    currentText: null,
    panelCollapsed: false,
    phonemeServerAvailable: null, // null = unknown, true/false after check
  };

  // ═══════════════════════════════════════════════════════════════
  //  DOM HELPERS
  // ═══════════════════════════════════════════════════════════════

  /**
   * Try a list of selectors, return the first match.
   */
  function queryFirst(selectorList, root = document) {
    for (const sel of selectorList) {
      const el = root.querySelector(sel);
      if (el) return el;
    }
    return null;
  }

  /**
   * Get the current slide's source text from the page.
   * Skips any elements inside the Phonetic Coach panel.
   */
  function getSourceText() {
    for (const sel of CONFIG.selectors.sourceText) {
      const candidates = document.querySelectorAll(sel);
      for (const el of candidates) {
        // Skip anything inside our own panel
        if (el.closest('#ile-phonetic-panel')) continue;
        const text = el.getAttribute('data-source-text') || el.textContent.trim();
        if (text && text.length > 0 && text.length < 500) return text;
      }
    }
    return null;
  }

  /**
   * Get the current slide's reference audio URL.
   */
  function getSourceAudioUrl() {
    for (const sel of CONFIG.selectors.sourceAudio) {
      const candidates = document.querySelectorAll(sel);
      for (const el of candidates) {
        if (el.closest('#ile-phonetic-panel')) continue;
        if (el.src) return el.src;
        const source = el.querySelector('source');
        if (source && source.src) return source.src;
        if (el.getAttribute('data-src')) return el.getAttribute('data-src');
      }
    }
    return null;
  }

  // ═══════════════════════════════════════════════════════════════
  //  UI CONSTRUCTION
  // ═══════════════════════════════════════════════════════════════

  function createPanel() {
    // Remove existing panel if any
    const existing = document.getElementById('ile-phonetic-panel');
    if (existing) existing.remove();

    const panel = document.createElement('div');
    panel.id = 'ile-phonetic-panel';

    panel.innerHTML = `
      <div class="panel-header">
        <div class="title">
          <span class="icon">🎙</span>
          <span>Phonetic Coach</span>
        </div>
        <button class="collapse-btn" title="Toggle panel">−</button>
      </div>
      <div class="panel-body">

        <div class="source-text" id="ipc-source-text">
          <div class="label">Reference sentence</div>
          <div class="sentence" id="ipc-sentence">Listening for slide content…</div>
        </div>

        <div class="controls">
          <button class="btn btn-play" id="ipc-play-ref" disabled title="Play reference audio">
            ▶ Reference
          </button>
          <button class="btn btn-record" id="ipc-record" title="Record your pronunciation">
            ⬤ Record
          </button>
          <button class="btn btn-play" id="ipc-play-learner" disabled title="Play your recording" style="display:none;">
            ▶ You
          </button>
        </div>

        <div class="waveform-live" id="ipc-waveform-live" style="display:none;">
          <canvas id="ipc-live-canvas"></canvas>
        </div>

        <div class="status" id="ipc-status"></div>

        <div id="ipc-results" style="display:none;">
          <div id="ipc-phoneme-diff" class="phoneme-diff-section" style="display:none;"></div>

          <div class="assessments" id="ipc-assessments"></div>

          <div class="viz-section">
            <div class="viz-label">📈 Pitch contour</div>
            <canvas id="ipc-pitch-canvas" width="776" height="240"></canvas>
          </div>

          <div class="viz-section">
            <div class="viz-label">🔊 Energy / stress</div>
            <canvas id="ipc-energy-canvas" width="776" height="240"></canvas>
          </div>

          <div class="viz-section">
            <div class="viz-label">⏱ Rhythm</div>
            <canvas id="ipc-rhythm-canvas" width="776" height="240"></canvas>
          </div>
        </div>

        <div class="empty-state" id="ipc-empty">
          <div class="icon">🎧</div>
          <div class="msg">Play the reference, then record yourself saying the same sentence. You'll get a detailed comparison — sound accuracy, vowel quality, pitch, rhythm, and stress.</div>
        </div>

      </div>
    `;

    document.body.appendChild(panel);
    bindPanelEvents(panel);
    return panel;
  }

  function bindPanelEvents(panel) {
    // Collapse toggle
    const header = panel.querySelector('.panel-header');
    const collapseBtn = panel.querySelector('.collapse-btn');

    header.addEventListener('click', () => {
      state.panelCollapsed = !state.panelCollapsed;
      panel.classList.toggle('collapsed', state.panelCollapsed);
      collapseBtn.textContent = state.panelCollapsed ? '+' : '−';
    });

    // Play reference
    const playBtn = panel.querySelector('#ipc-play-ref');
    playBtn.addEventListener('click', playReference);

    // Record
    const recordBtn = panel.querySelector('#ipc-record');
    recordBtn.addEventListener('click', toggleRecording);

    // Play learner recording
    const playLearnerBtn = panel.querySelector('#ipc-play-learner');
    playLearnerBtn.addEventListener('click', playLearnerRecording);
  }

  // ═══════════════════════════════════════════════════════════════
  //  REFERENCE AUDIO
  // ═══════════════════════════════════════════════════════════════

  async function loadReference() {
    const audioUrl = getSourceAudioUrl();
    const text = getSourceText();

    const sentenceEl = document.getElementById('ipc-sentence');
    const playBtn = document.getElementById('ipc-play-ref');

    if (text) {
      sentenceEl.textContent = text;
      state.currentText = text;
    } else if (!state.currentText) {
      sentenceEl.textContent = 'Listening for slide content…';
    }

    if (audioUrl) {
      playBtn.disabled = false;
    } else {
      playBtn.disabled = true;
    }

    // Skip if same audio already loaded
    if (audioUrl && audioUrl === state.referenceAudioUrl && state.referenceFeatures) return;
    state.referenceAudioUrl = audioUrl;
    state.referenceFeatures = null;

    if (!audioUrl) {
      setStatus('No reference audio found on this slide.', 'error');
      return;
    }

    try {
      // Step 1: Decode reference audio to Float32
      setStatus('Analyzing reference audio…', 'analyzing');
      const refAudio = await AudioAnalyzer.decodeAudioFromUrl(audioUrl);

      // Step 2: Extract JS features (pitch, energy, rhythm — for prosody comparison)
      state.referenceFeatures = AudioAnalyzer.extractFeatures(refAudio.float32Data, refAudio.sampleRate);

      // Step 3: Get phonemes from the native host (same pipeline as learner)
      if (state.phonemeServerAvailable) {
        try {
          setStatus('Getting reference phonemes…', 'analyzing');
          const refPhonemes = await recognizePhonemes(refAudio.float32Data, refAudio.sampleRate);

          if (refPhonemes?.status === 'downloading') {
            // Model is downloading — phonemes will be available on next try
            setStatus(refPhonemes.message || 'Downloading model…', 'analyzing');
          } else if (refPhonemes?.phonemes) {
            state.referenceFeatures.phonemes = refPhonemes;
            console.log('[IPC] Reference IPA (from native host):', refPhonemes.rawIPA);
          }
        } catch (err) {
          console.warn('[IPC] Could not get reference phonemes from host:', err);
          // Not fatal — prosody comparison still works
        }
      }

      setStatus('');
    } catch (err) {
      console.error('[IPC] Failed to load reference:', err);
      setStatus('Could not load reference audio.', 'error');
    }
  }

  async function playReference() {
    if (!state.referenceAudioUrl) return;

    const playBtn = document.getElementById('ipc-play-ref');
    playBtn.disabled = true;

    try {
      const ctx = AudioAnalyzer.getAudioContext();
      if (ctx.state === 'suspended') await ctx.resume();

      const response = await fetch(state.referenceAudioUrl);
      const arrayBuffer = await response.arrayBuffer();
      const audioBuffer = await ctx.decodeAudioData(arrayBuffer);
      const source = ctx.createBufferSource();
      source.buffer = audioBuffer;
      source.connect(ctx.destination);
      source.start();
      source.onended = () => { playBtn.disabled = false; };
    } catch (err) {
      console.error('[IPC] Playback error:', err);
      playBtn.disabled = false;
    }
  }

  async function playLearnerRecording() {
    if (!state.learnerResult || !state.learnerResult.blob) return;

    const playBtn = document.getElementById('ipc-play-learner');
    playBtn.disabled = true;

    try {
      const ctx = AudioAnalyzer.getAudioContext();
      if (ctx.state === 'suspended') await ctx.resume();

      const arrayBuffer = await state.learnerResult.blob.arrayBuffer();
      const audioBuffer = await ctx.decodeAudioData(arrayBuffer);
      const source = ctx.createBufferSource();
      source.buffer = audioBuffer;
      source.connect(ctx.destination);
      source.start();
      source.onended = () => { playBtn.disabled = false; };
    } catch (err) {
      console.error('[IPC] Learner playback error:', err);
      playBtn.disabled = false;
    }
  }

  // ═══════════════════════════════════════════════════════════════
  //  PHONEME ANALYSIS (via Native Messaging)
  // ═══════════════════════════════════════════════════════════════

  /**
   * Check if the native phoneme host is available.
   */
  async function checkPhonemeServer() {
    try {
      const response = await chrome.runtime.sendMessage({ type: 'phoneme-health' });
      state.phonemeServerAvailable = response?.status === 'ok';
    } catch (e) {
      state.phonemeServerAvailable = false;
    }
    console.log('[IPC] Phoneme host:', state.phonemeServerAvailable ? 'connected' : 'not available');
  }

  /**
   * Send audio to native host, get phonemes back.
   */
  async function recognizePhonemes(float32Data, sampleRate) {
    // Send raw samples as base64 float32 + sample rate
    // More efficient than re-encoding to a file format
    const bytes = new Uint8Array(float32Data.buffer);
    let binary = '';
    for (let i = 0; i < bytes.length; i++) {
      binary += String.fromCharCode(bytes[i]);
    }
    const base64 = btoa(binary);

    const response = await chrome.runtime.sendMessage({
      type: 'phoneme-analyze',
      audio: base64,
      sampleRate: sampleRate,
      format: 'float32',
    });

    if (response?.error) {
      throw new Error(response.error);
    }

    return response;
  }

  /**
   * Diff two phoneme sequences. Returns alignment with match/substitute/delete/insert.
   */
  function diffPhonemes(refPhonemes, learnerPhonemes) {
    const ref = refPhonemes.filter(p => p.phoneme !== ' ' && p.phoneme !== '|');
    const learn = learnerPhonemes.filter(p => p.phoneme !== ' ' && p.phoneme !== '|');

    const n = ref.length, m = learn.length;
    const dp = Array(n + 1).fill(null).map(() => Array(m + 1).fill(0));
    const ops = Array(n + 1).fill(null).map(() => Array(m + 1).fill(''));

    for (let i = 0; i <= n; i++) { dp[i][0] = i; ops[i][0] = 'delete'; }
    for (let j = 0; j <= m; j++) { dp[0][j] = j; ops[0][j] = 'insert'; }
    ops[0][0] = '';

    // Similar phonemes get lower substitution cost
    const SIMILAR_PAIRS = [
      ['ɪ', 'iː'], ['ɛ', 'eɪ'], ['æ', 'ɛ'], ['ɑ', 'ʌ'],
      ['ɔ', 'oʊ'], ['ʊ', 'uː'], ['ə', 'ʌ'], ['ə', 'ɪ'],
      ['t', 'd'], ['p', 'b'], ['k', 'ɡ'], ['s', 'z'],
      ['f', 'v'], ['θ', 'ð'], ['ʃ', 'ʒ'], ['tʃ', 'dʒ'],
    ];
    function isSimilar(a, b) {
      return SIMILAR_PAIRS.some(([x, y]) => (a === x && b === y) || (a === y && b === x));
    }

    for (let i = 1; i <= n; i++) {
      for (let j = 1; j <= m; j++) {
        if (ref[i-1].phoneme === learn[j-1].phoneme) {
          dp[i][j] = dp[i-1][j-1];
          ops[i][j] = 'match';
        } else {
          const subCost = isSimilar(ref[i-1].phoneme, learn[j-1].phoneme) ? 0.5 : 1;
          const sub = dp[i-1][j-1] + subCost;
          const del = dp[i-1][j] + 1;
          const ins = dp[i][j-1] + 1;
          const best = Math.min(sub, del, ins);
          dp[i][j] = best;
          if (best === sub) ops[i][j] = isSimilar(ref[i-1].phoneme, learn[j-1].phoneme) ? 'similar' : 'substitute';
          else if (best === del) ops[i][j] = 'delete';
          else ops[i][j] = 'insert';
        }
      }
    }

    // Traceback
    const alignment = [];
    let i = n, j = m;
    while (i > 0 || j > 0) {
      const op = ops[i][j];
      if (op === 'match' || op === 'substitute' || op === 'similar') {
        alignment.unshift({ type: op, expected: ref[i-1], actual: learn[j-1] });
        i--; j--;
      } else if (op === 'delete') {
        alignment.unshift({ type: 'delete', expected: ref[i-1], actual: null });
        i--;
      } else {
        alignment.unshift({ type: 'insert', expected: null, actual: learn[j-1] });
        j--;
      }
    }

    const matches = alignment.filter(a => a.type === 'match').length;
    const similar = alignment.filter(a => a.type === 'similar').length;
    const accuracy = ref.length > 0 ? (matches + similar * 0.5) / ref.length : 0;

    return { alignment, accuracy, editDistance: dp[n][m] };
  }

  // ═══════════════════════════════════════════════════════════════
  //  RECORDING
  // ═══════════════════════════════════════════════════════════════

  async function toggleRecording() {
    if (state.isRecording) {
      stopRecording();
    } else {
      startRecording();
    }
  }

  async function startRecording() {
    const recordBtn = document.getElementById('ipc-record');
    const liveWaveform = document.getElementById('ipc-waveform-live');

    try {
      state.recorder = AudioAnalyzer.createRecorder();
      await state.recorder.start();

      state.isRecording = true;
      recordBtn.classList.add('recording');
      recordBtn.innerHTML = '⬛ Stop';
      liveWaveform.style.display = 'block';
      setStatus('Recording… speak now', '');

      // Auto-stop after max duration
      state.recordingTimeout = setTimeout(() => {
        if (state.isRecording) stopRecording();
      }, CONFIG.maxRecordingDuration * 1000);

    } catch (err) {
      console.error('[IPC] Recording error:', err);
      if (err.name === 'NotAllowedError') {
        setStatus('Microphone access denied. Please allow microphone access and try again.', 'error');
      } else {
        setStatus('Could not start recording: ' + err.message, 'error');
      }
    }
  }

  async function stopRecording() {
    if (!state.recorder || !state.isRecording) return;

    clearTimeout(state.recordingTimeout);

    const recordBtn = document.getElementById('ipc-record');
    const liveWaveform = document.getElementById('ipc-waveform-live');

    state.isRecording = false;
    recordBtn.classList.remove('recording');
    recordBtn.innerHTML = '⬤ Record';
    liveWaveform.style.display = 'none';

    setStatus('Analyzing…', 'analyzing');

    try {
      const result = await state.recorder.stop();
      state.learnerResult = result;

      if (result.duration < CONFIG.minRecordingDuration) {
        setStatus('Recording too short. Try again.', 'error');
        return;
      }

      if (!state.referenceFeatures) {
        setStatus('No reference audio loaded. Cannot compare.', 'error');
        return;
      }

      // Build comparison object
      const comparison = {
        assessments: [],
        phonemeDiff: null,
        refPhonemes: null,
        learnerPhonemes: null,
        pitch: null,
        rhythm: null,
        duration: null,
        energy: null,
      };

      // ── Step 1: Phoneme analysis via native host ──
      if (state.phonemeServerAvailable) {
        try {
          // Get reference phonemes if we don't have them yet
          if (!state.referenceFeatures.phonemes) {
            setStatus('Getting reference phonemes…', 'analyzing');
            const refAudio = await AudioAnalyzer.decodeAudioFromUrl(state.referenceAudioUrl);
            const refPhonemes = await recognizePhonemes(refAudio.float32Data, refAudio.sampleRate);
            if (refPhonemes?.phonemes) {
              state.referenceFeatures.phonemes = refPhonemes;
            }
          }

          if (state.referenceFeatures.phonemes) {
            setStatus('Running phoneme analysis…', 'analyzing');
            const serverResult = await recognizePhonemes(result.float32Data, result.sampleRate);
            comparison.learnerPhonemes = serverResult;
            comparison.refPhonemes = state.referenceFeatures.phonemes;

            console.log('[IPC] Reference IPA:', comparison.refPhonemes.rawIPA);
            console.log('[IPC] Learner IPA:  ', serverResult.rawIPA);

            // Diff the phoneme sequences
            const diff = diffPhonemes(
              comparison.refPhonemes.phonemes,
              serverResult.phonemes
            );
            comparison.phonemeDiff = diff;

            const pct = Math.round(diff.accuracy * 100);
            console.log('[IPC] Phoneme accuracy:', pct + '%');

            // Build phoneme assessment
            if (diff.accuracy >= 0.85) {
              comparison.assessments.push({
                dimension: 'phoneme accuracy', level: 'good', priority: 1,
                message: `Phoneme accuracy: ${pct}% — your sounds closely match the reference.`
              });
            } else if (diff.accuracy >= 0.6) {
              const errors = diff.alignment
                .filter(a => a.type === 'substitute')
                .slice(0, 4)
                .map(a => `/${a.expected.phoneme}/ → /${a.actual.phoneme}/`);
              comparison.assessments.push({
                dimension: 'phoneme accuracy', level: 'fair', priority: 1,
                message: `Phoneme accuracy: ${pct}%. ${errors.length > 0 ? 'Substitutions: ' + errors.join(', ') : ''}`
              });
            } else {
              comparison.assessments.push({
                dimension: 'phoneme accuracy', level: 'needs_work', priority: 1,
                message: `Phoneme accuracy: ${pct}%. Significant differences — listen to the reference and try again.`
              });
            }

            // Post results to the website via custom event
            document.dispatchEvent(new CustomEvent('ile-phoneme-result', {
              detail: {
                referenceIPA: comparison.refPhonemes.rawIPA,
                learnerIPA: serverResult.rawIPA,
                accuracy: diff.accuracy,
                alignment: diff.alignment,
                sourceText: state.currentText,
              }
            }));
          }

        } catch (err) {
          console.warn('[IPC] Phoneme host error:', err);
          comparison.assessments.push({
            dimension: 'phoneme analysis', level: 'fair', priority: 1,
            message: 'Phoneme analysis failed: ' + err.message
          });
        }
      } else {
        comparison.assessments.push({
          dimension: 'phoneme analysis', level: 'fair', priority: 1,
          message: 'Phoneme host not connected. Run setup.py to enable pronunciation feedback.'
        });
      }

      // ── Step 2: Prosody analysis via JS (always runs) ──
      setStatus('Analyzing prosody…', 'analyzing');
      const learnerFeatures = AudioAnalyzer.extractFeatures(result.float32Data, result.sampleRate);

      // Pitch comparison
      comparison.pitch = PhoneticComparison.comparePitch(state.referenceFeatures, learnerFeatures);
      if (comparison.pitch.meanDeviation < 1.5) {
        comparison.assessments.push({ dimension: 'intonation', level: 'good', priority: 3, message: 'Pitch contour closely matches the reference.' });
      } else if (comparison.pitch.meanDeviation < 3) {
        comparison.assessments.push({ dimension: 'intonation', level: 'fair', priority: 3, message: 'Pitch pattern is close but drifts in places.' });
      } else {
        comparison.assessments.push({ dimension: 'intonation', level: 'needs_work', priority: 3, message: 'Pitch contour differs significantly. Listen to the melody of the sentence.' });
      }

      // Duration
      comparison.duration = PhoneticComparison.compareDuration(state.referenceFeatures, learnerFeatures);
      if (comparison.duration.feedback === 'good') {
        comparison.assessments.push({ dimension: 'tempo', level: 'good', priority: 5, message: 'Speaking pace matches the reference.' });
      } else if (comparison.duration.feedback === 'too_slow') {
        comparison.assessments.push({ dimension: 'tempo', level: 'fair', priority: 5, message: `Speaking ${Math.round(comparison.duration.percentDiff)}% slower than reference.` });
      } else {
        comparison.assessments.push({ dimension: 'tempo', level: 'fair', priority: 5, message: `Speaking ${Math.round(Math.abs(comparison.duration.percentDiff))}% faster than reference.` });
      }

      // Rhythm
      comparison.rhythm = PhoneticComparison.compareRhythm(state.referenceFeatures, learnerFeatures);
      if (comparison.rhythm.rhythmScore < 0.1) {
        comparison.assessments.push({ dimension: 'rhythm', level: 'good', priority: 6, message: 'Rhythm and stress pattern match well.' });
      } else if (comparison.rhythm.rhythmScore < 0.25) {
        comparison.assessments.push({ dimension: 'rhythm', level: 'fair', priority: 6, message: 'Rhythm is close. Some syllable durations differ.' });
      } else {
        comparison.assessments.push({ dimension: 'rhythm', level: 'needs_work', priority: 6, message: 'Rhythm differs. Focus on which syllables are long vs. short.' });
      }

      // Energy
      comparison.energy = PhoneticComparison.compareEnergy(state.referenceFeatures, learnerFeatures);

      // Store for rendering
      state.comparisonResult = comparison;

      // Render
      renderResults(comparison, state.referenceFeatures, learnerFeatures);

      // Show learner playback button
      const playLearnerBtn = document.getElementById('ipc-play-learner');
      if (playLearnerBtn) {
        playLearnerBtn.style.display = '';
        playLearnerBtn.disabled = false;
      }

      setStatus('');

    } catch (err) {
      console.error('[IPC] Analysis error:', err);
      setStatus('Analysis failed: ' + err.message, 'error');
    }
  }

  // ═══════════════════════════════════════════════════════════════
  //  VISUALIZATION
  // ═══════════════════════════════════════════════════════════════

  function renderResults(comparison, refFeatures, learnerFeatures) {
    const resultsEl = document.getElementById('ipc-results');
    const emptyEl = document.getElementById('ipc-empty');

    resultsEl.style.display = 'block';
    emptyEl.style.display = 'none';

    // Phoneme diff (the primary feedback if server is available)
    renderPhonemeDiff(comparison);

    // Assessments
    renderAssessments(comparison.assessments);

    // Pitch contour overlay
    if (comparison.pitch) {
      renderPitchContour(comparison.pitch, refFeatures, learnerFeatures);
    }

    // Energy comparison
    if (comparison.energy) {
      renderEnergyComparison(comparison.energy);
    }

    // Rhythm comparison
    if (comparison.rhythm) {
      renderRhythmComparison(comparison.rhythm, refFeatures, learnerFeatures);
    }
  }

  /**
   * Render phoneme diff — two-sided when server is available.
   * Shows expected vs actual IPA with color coding.
   */
  function renderPhonemeDiff(comparison) {
    const container = document.getElementById('ipc-phoneme-diff');
    if (!container) return;

    // Full two-sided diff (server provided learner phonemes)
    if (comparison.phonemeDiff && comparison.phonemeDiff.alignment) {
      container.style.display = 'block';
      const diff = comparison.phonemeDiff;
      const pct = Math.round(diff.accuracy * 100);

      // Build expected row
      const expectedHTML = diff.alignment.map(a => {
        if (a.type === 'match') return `<span class="ph-match">/${a.expected.phoneme}/</span>`;
        if (a.type === 'similar') return `<span class="ph-similar" title="Close — you said /${a.actual.phoneme}/">/${a.expected.phoneme}/</span>`;
        if (a.type === 'substitute') return `<span class="ph-wrong" title="You said /${a.actual.phoneme}/">/${a.expected.phoneme}/</span>`;
        if (a.type === 'delete') return `<span class="ph-missing" title="Sound missing">/${a.expected.phoneme}/</span>`;
        return '';
      }).filter(Boolean).join('');

      // Build learner row
      const learnerHTML = diff.alignment.map(a => {
        if (a.type === 'match') return `<span class="ph-match">/${a.actual.phoneme}/</span>`;
        if (a.type === 'similar') return `<span class="ph-similar">/${a.actual.phoneme}/</span>`;
        if (a.type === 'substitute') return `<span class="ph-wrong">/${a.actual.phoneme}/</span>`;
        if (a.type === 'insert') return `<span class="ph-extra">/${a.actual.phoneme}/</span>`;
        if (a.type === 'delete') return `<span class="ph-gap">—</span>`;
        return '';
      }).join('');

      // Error summary
      const subs = diff.alignment.filter(a => a.type === 'substitute');
      const dels = diff.alignment.filter(a => a.type === 'delete');
      const errorItems = [];
      subs.slice(0, 4).forEach(s => errorItems.push(`<span class="ph-error-item">/${s.expected.phoneme}/ → /${s.actual.phoneme}/</span>`));
      dels.slice(0, 2).forEach(d => errorItems.push(`<span class="ph-error-item">/${d.expected.phoneme}/ missing</span>`));
      const errorHTML = errorItems.length > 0 ? `<div class="ph-errors">${errorItems.join(' ')}</div>` : '';

      container.innerHTML = `
        <div class="ph-diff-label">🔬 Phoneme analysis <span class="word-accuracy ${pct >= 80 ? 'good' : pct >= 50 ? 'fair' : 'poor'}">${pct}%</span></div>
        <div class="ph-diff-row">
          <div class="ph-diff-line">
            <span class="word-diff-tag">Expected</span>
            <div class="ph-diff-phonemes">${expectedHTML}</div>
          </div>
          <div class="ph-diff-line">
            <span class="word-diff-tag">You said</span>
            <div class="ph-diff-phonemes">${learnerHTML}</div>
          </div>
        </div>
        ${errorHTML}
      `;
      return;
    }

    // Fallback: just show reference IPA (no server)
    const refPhonemes = comparison.refPhonemes;
    if (refPhonemes && refPhonemes.phonemes && refPhonemes.phonemes.length > 0) {
      container.style.display = 'block';
      const phonemeHTML = refPhonemes.phonemes
        .filter(p => p.phoneme !== ' ' && p.phoneme !== '|')
        .map(p => `<span class="ph-match">/${p.phoneme}/</span>`)
        .join('');

      container.innerHTML = `
        <div class="ph-diff-label">🔬 Reference phonemes</div>
        <div class="ph-diff-row">
          <div class="ph-diff-line">
            <span class="word-diff-tag">Target</span>
            <div class="ph-diff-phonemes">${phonemeHTML}</div>
          </div>
        </div>
      `;
      return;
    }

    container.style.display = 'none';
  }

  function renderAssessments(assessments) {
    const container = document.getElementById('ipc-assessments');
    container.innerHTML = assessments.map(a => `
      <div class="assessment">
        <div class="badge ${a.level}"></div>
        <div>
          <div class="dim-label">${a.dimension}</div>
          <div class="message">${a.message}</div>
        </div>
      </div>
    `).join('');
  }

  /**
   * Draw MFCC distance heatmap — shows where the learner's sounds diverge.
   * Green = close match, yellow = moderate, red = significant deviation.
   * Problem regions are highlighted with markers.
   */
  function renderMFCCHeatmap(mfccResult) {
    const canvas = document.getElementById('ipc-mfcc-canvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;

    canvas.width = canvas.clientWidth * dpr;
    canvas.height = canvas.clientHeight * dpr;
    ctx.scale(dpr, dpr);

    const W = canvas.clientWidth;
    const H = canvas.clientHeight;
    const aligned = mfccResult.aligned;

    if (!aligned || aligned.length === 0) {
      ctx.fillStyle = '#666';
      ctx.font = '12px -apple-system, sans-serif';
      ctx.fillText('MFCC data not available', 10, H / 2);
      return;
    }

    ctx.clearRect(0, 0, W, H);

    const barH = H - 30;
    const barW = W / aligned.length;

    // Find distance range for color mapping
    const maxDist = Math.max(...aligned.map(a => a.distance));
    const meanDist = mfccResult.meanDistance;

    // Draw heatmap bars
    for (let i = 0; i < aligned.length; i++) {
      const d = aligned[i].distance;
      const normalized = Math.min(d / 50, 1); // Absolute scale: 0=perfect, 50+=max red

      // Green → Yellow → Red gradient
      let r, g, b;
      if (normalized < 0.4) {
        // Green to yellow
        const t = normalized / 0.4;
        r = Math.round(50 + 205 * t);
        g = Math.round(200 - 20 * t);
        b = 50;
      } else {
        // Yellow to red
        const t = (normalized - 0.4) / 0.6;
        r = Math.round(230 + 25 * t);
        g = Math.round(180 * (1 - t));
        b = Math.round(50 * (1 - t));
      }

      ctx.fillStyle = `rgba(${r}, ${g}, ${b}, 0.8)`;
      ctx.fillRect(i * barW, 10, Math.max(barW, 1), barH);
    }

    // Mark problem regions with red underlines
    if (mfccResult.problemRegions) {
      ctx.strokeStyle = '#e74c3c';
      ctx.lineWidth = 3;
      for (const region of mfccResult.problemRegions) {
        // Find indices in aligned array
        const startIdx = aligned.findIndex(a => a.refTime >= region.startTime);
        const endIdx = aligned.findIndex(a => a.refTime >= region.endTime);
        if (startIdx >= 0 && endIdx >= 0) {
          ctx.beginPath();
          ctx.moveTo(startIdx * barW, barH + 14);
          ctx.lineTo(endIdx * barW, barH + 14);
          ctx.stroke();
        }
      }
    }

    // Score label
    ctx.font = '11px -apple-system, sans-serif';
    ctx.fillStyle = '#aaa';
    ctx.fillText(`Score: ${Math.round(mfccResult.score)}/100`, W - 90, H - 4);

    // Legend
    ctx.fillStyle = 'rgba(50, 200, 50, 0.8)';
    ctx.fillRect(8, H - 14, 10, 10);
    ctx.fillStyle = '#888';
    ctx.fillText('match', 22, H - 5);

    ctx.fillStyle = 'rgba(230, 180, 50, 0.8)';
    ctx.fillRect(68, H - 14, 10, 10);
    ctx.fillStyle = '#888';
    ctx.fillText('close', 82, H - 5);

    ctx.fillStyle = 'rgba(230, 50, 50, 0.8)';
    ctx.fillRect(122, H - 14, 10, 10);
    ctx.fillStyle = '#888';
    ctx.fillText('differs', 136, H - 5);
  }

  /**
   * Draw vowel space scatter plot (F1 vs F2).
   * Traditional phonetics layout: F2 on x-axis (reversed), F1 on y-axis (reversed).
   * This means "ee" (high front) is top-right, "ah" (low back) is bottom-left.
   */
  function renderVowelSpace(formantResult) {
    const canvas = document.getElementById('ipc-vowel-canvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;

    canvas.width = canvas.clientWidth * dpr;
    canvas.height = canvas.clientHeight * dpr;
    ctx.scale(dpr, dpr);

    const W = canvas.clientWidth;
    const H = canvas.clientHeight;

    ctx.clearRect(0, 0, W, H);

    const refPoints = formantResult.vowelSpaceRef;
    const learnerPoints = formantResult.vowelSpaceLearner;

    if ((!refPoints || refPoints.length === 0) && (!learnerPoints || learnerPoints.length === 0)) {
      ctx.fillStyle = '#666';
      ctx.font = '12px -apple-system, sans-serif';
      ctx.fillText('Formant data not available', 10, H / 2);
      return;
    }

    const pad = { top: 20, bottom: 30, left: 40, right: 20 };
    const plotW = W - pad.left - pad.right;
    const plotH = H - pad.top - pad.bottom;

    // F2 range (reversed): ~2800 to 600 Hz (left to right = front to back)
    // F1 range (reversed): ~200 to 900 Hz (top to bottom = close to open)
    const f2Min = 500, f2Max = 2800;
    const f1Min = 200, f1Max = 950;

    const xScale = (f2) => pad.left + (1 - (f2 - f2Min) / (f2Max - f2Min)) * plotW;
    const yScale = (f1) => pad.top + ((f1 - f1Min) / (f1Max - f1Min)) * plotH;

    // Grid
    ctx.strokeStyle = 'rgba(255,255,255,0.05)';
    ctx.lineWidth = 1;
    for (let f2 = 800; f2 <= 2600; f2 += 400) {
      const x = xScale(f2);
      ctx.beginPath(); ctx.moveTo(x, pad.top); ctx.lineTo(x, H - pad.bottom); ctx.stroke();
      ctx.fillStyle = '#444';
      ctx.font = '9px -apple-system, sans-serif';
      ctx.fillText(`${f2}`, x - 10, H - pad.bottom + 12);
    }
    for (let f1 = 300; f1 <= 800; f1 += 100) {
      const y = yScale(f1);
      ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(W - pad.right, y); ctx.stroke();
      ctx.fillStyle = '#444';
      ctx.fillText(`${f1}`, 4, y + 3);
    }

    // Axis labels
    ctx.fillStyle = '#555';
    ctx.font = '10px -apple-system, sans-serif';
    ctx.fillText('← F2 (front)', pad.left, pad.top - 6);
    ctx.fillText('(back) →', W - pad.right - 50, pad.top - 6);
    ctx.save();
    ctx.translate(10, pad.top + plotH / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('F1 (open) →', 0, 0);
    ctx.restore();

    // Reference vowel cloud (blue, more transparent)
    if (refPoints && refPoints.length > 0) {
      ctx.fillStyle = 'rgba(100, 160, 255, 0.15)';
      for (const p of refPoints) {
        const x = xScale(p.F2);
        const y = yScale(p.F1);
        if (x >= pad.left && x <= W - pad.right && y >= pad.top && y <= H - pad.bottom) {
          ctx.beginPath();
          ctx.arc(x, y, 3, 0, Math.PI * 2);
          ctx.fill();
        }
      }
      // Draw convex hull-ish boundary (simplified: just draw larger dots at extremes)
      ctx.fillStyle = 'rgba(100, 160, 255, 0.5)';
      // Sample 20 evenly spaced points for clearer display
      const step = Math.max(1, Math.floor(refPoints.length / 20));
      for (let i = 0; i < refPoints.length; i += step) {
        const p = refPoints[i];
        const x = xScale(p.F2);
        const y = yScale(p.F1);
        if (x >= pad.left && x <= W - pad.right && y >= pad.top && y <= H - pad.bottom) {
          ctx.beginPath();
          ctx.arc(x, y, 4, 0, Math.PI * 2);
          ctx.fill();
        }
      }
    }

    // Learner vowel cloud (orange)
    if (learnerPoints && learnerPoints.length > 0) {
      ctx.fillStyle = 'rgba(255, 160, 60, 0.15)';
      for (const p of learnerPoints) {
        const x = xScale(p.F2);
        const y = yScale(p.F1);
        if (x >= pad.left && x <= W - pad.right && y >= pad.top && y <= H - pad.bottom) {
          ctx.beginPath();
          ctx.arc(x, y, 3, 0, Math.PI * 2);
          ctx.fill();
        }
      }
      ctx.fillStyle = 'rgba(255, 160, 60, 0.5)';
      const step = Math.max(1, Math.floor(learnerPoints.length / 20));
      for (let i = 0; i < learnerPoints.length; i += step) {
        const p = learnerPoints[i];
        const x = xScale(p.F2);
        const y = yScale(p.F1);
        if (x >= pad.left && x <= W - pad.right && y >= pad.top && y <= H - pad.bottom) {
          ctx.beginPath();
          ctx.arc(x, y, 4, 0, Math.PI * 2);
          ctx.fill();
        }
      }
    }

    // Legend
    ctx.font = '11px -apple-system, sans-serif';
    ctx.fillStyle = 'rgba(100, 160, 255, 0.9)';
    ctx.fillText('● Reference', W - 160, H - 8);
    ctx.fillStyle = 'rgba(255, 160, 60, 0.9)';
    ctx.fillText('● You', W - 70, H - 8);
  }

  /**
   * Draw pitch contour: reference in blue, learner in orange.
   */
  function renderPitchContour(pitchComparison, refFeatures, learnerFeatures) {
    const canvas = document.getElementById('ipc-pitch-canvas');
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;

    canvas.width = canvas.clientWidth * dpr;
    canvas.height = canvas.clientHeight * dpr;
    ctx.scale(dpr, dpr);

    const W = canvas.clientWidth;
    const H = canvas.clientHeight;
    const pad = { top: 10, bottom: 20, left: 10, right: 10 };

    ctx.clearRect(0, 0, W, H);

    const refContour = refFeatures.pitchContour.filter(p => p.semitones !== null);
    const learnerContour = learnerFeatures.pitchContour.filter(p => p.semitones !== null);

    if (refContour.length === 0 && learnerContour.length === 0) return;

    // Find semitone range
    const allST = [
      ...refContour.map(p => p.semitones),
      ...learnerContour.map(p => p.semitones)
    ];
    const minST = Math.min(...allST) - 2;
    const maxST = Math.max(...allST) + 2;

    const refDuration = refFeatures.duration;
    const learnerDuration = learnerFeatures.duration;
    const maxDuration = Math.max(refDuration, learnerDuration);

    const xScale = (t, dur) => pad.left + (t / dur) * (W - pad.left - pad.right);
    const yScale = (st) => pad.top + (1 - (st - minST) / (maxST - minST)) * (H - pad.top - pad.bottom);

    // Draw reference contour
    ctx.strokeStyle = 'rgba(100, 160, 255, 0.8)';
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    refContour.forEach((p, i) => {
      const x = xScale(p.time, refDuration);
      const y = yScale(p.semitones);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();

    // Draw learner contour
    ctx.strokeStyle = 'rgba(255, 160, 60, 0.8)';
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    learnerContour.forEach((p, i) => {
      const x = xScale(p.time, learnerDuration);
      const y = yScale(p.semitones);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();

    // Legend
    ctx.font = '11px -apple-system, sans-serif';
    ctx.fillStyle = 'rgba(100, 160, 255, 0.9)';
    ctx.fillText('● Reference', W - 160, H - 4);
    ctx.fillStyle = 'rgba(255, 160, 60, 0.9)';
    ctx.fillText('● You', W - 70, H - 4);
  }

  /**
   * Draw energy/stress comparison.
   */
  function renderEnergyComparison(energyResult) {
    const canvas = document.getElementById('ipc-energy-canvas');
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;

    canvas.width = canvas.clientWidth * dpr;
    canvas.height = canvas.clientHeight * dpr;
    ctx.scale(dpr, dpr);

    const W = canvas.clientWidth;
    const H = canvas.clientHeight;
    const aligned = energyResult.aligned;

    if (aligned.length === 0) return;

    ctx.clearRect(0, 0, W, H);

    const maxRms = Math.max(
      ...aligned.map(a => a.refRms),
      ...aligned.map(a => a.learnerRms)
    ) || 0.1;

    const midY = H / 2;

    // Reference energy (top half, blue)
    ctx.fillStyle = 'rgba(100, 160, 255, 0.3)';
    ctx.strokeStyle = 'rgba(100, 160, 255, 0.7)';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    aligned.forEach((a, i) => {
      const x = (i / aligned.length) * W;
      const barH = (a.refRms / maxRms) * (midY - 10);
      if (i === 0) ctx.moveTo(x, midY - barH);
      else ctx.lineTo(x, midY - barH);
    });
    ctx.lineTo(W, midY);
    ctx.lineTo(0, midY);
    ctx.closePath();
    ctx.fill();
    ctx.stroke();

    // Learner energy (bottom half, orange, mirrored)
    ctx.fillStyle = 'rgba(255, 160, 60, 0.3)';
    ctx.strokeStyle = 'rgba(255, 160, 60, 0.7)';
    ctx.beginPath();
    aligned.forEach((a, i) => {
      const x = (i / aligned.length) * W;
      const barH = (a.learnerRms / maxRms) * (midY - 10);
      if (i === 0) ctx.moveTo(x, midY + barH);
      else ctx.lineTo(x, midY + barH);
    });
    ctx.lineTo(W, midY);
    ctx.lineTo(0, midY);
    ctx.closePath();
    ctx.fill();
    ctx.stroke();

    // Center line
    ctx.strokeStyle = 'rgba(255,255,255,0.15)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, midY);
    ctx.lineTo(W, midY);
    ctx.stroke();

    // Labels
    ctx.font = '10px -apple-system, sans-serif';
    ctx.fillStyle = 'rgba(100, 160, 255, 0.7)';
    ctx.fillText('Reference ↑', 8, 14);
    ctx.fillStyle = 'rgba(255, 160, 60, 0.7)';
    ctx.fillText('You ↓', 8, H - 6);
  }

  /**
   * Draw rhythm comparison as onset markers on a timeline.
   */
  function renderRhythmComparison(rhythmResult, refFeatures, learnerFeatures) {
    const canvas = document.getElementById('ipc-rhythm-canvas');
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;

    canvas.width = canvas.clientWidth * dpr;
    canvas.height = canvas.clientHeight * dpr;
    ctx.scale(dpr, dpr);

    const W = canvas.clientWidth;
    const H = canvas.clientHeight;
    const pad = 20;

    ctx.clearRect(0, 0, W, H);

    const maxDur = Math.max(refFeatures.duration, learnerFeatures.duration) || 1;
    const xScale = (t) => pad + (t / maxDur) * (W - pad * 2);

    const refY = H * 0.33;
    const learnerY = H * 0.67;

    // Reference timeline
    ctx.strokeStyle = 'rgba(100, 160, 255, 0.4)';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(xScale(0), refY);
    ctx.lineTo(xScale(refFeatures.duration), refY);
    ctx.stroke();

    // Reference onsets
    ctx.fillStyle = 'rgba(100, 160, 255, 0.9)';
    rhythmResult.refOnsets.forEach(t => {
      ctx.beginPath();
      ctx.arc(xScale(t), refY, 5, 0, Math.PI * 2);
      ctx.fill();
    });

    // Learner timeline
    ctx.strokeStyle = 'rgba(255, 160, 60, 0.4)';
    ctx.beginPath();
    ctx.moveTo(xScale(0), learnerY);
    ctx.lineTo(xScale(learnerFeatures.duration), learnerY);
    ctx.stroke();

    // Learner onsets
    ctx.fillStyle = 'rgba(255, 160, 60, 0.9)';
    rhythmResult.learnerOnsets.forEach(t => {
      ctx.beginPath();
      ctx.arc(xScale(t), learnerY, 5, 0, Math.PI * 2);
      ctx.fill();
    });

    // Labels
    ctx.font = '10px -apple-system, sans-serif';
    ctx.fillStyle = 'rgba(100, 160, 255, 0.7)';
    ctx.fillText('Reference', 8, refY - 12);
    ctx.fillStyle = 'rgba(255, 160, 60, 0.7)';
    ctx.fillText('You', 8, learnerY - 12);

    // Duration labels
    ctx.fillStyle = '#666';
    ctx.font = '10px -apple-system, sans-serif';
    ctx.fillText(`${refFeatures.duration.toFixed(1)}s`, xScale(refFeatures.duration) + 6, refY + 4);
    ctx.fillText(`${learnerFeatures.duration.toFixed(1)}s`, xScale(learnerFeatures.duration) + 6, learnerY + 4);
  }

  // ═══════════════════════════════════════════════════════════════
  //  HELPERS
  // ═══════════════════════════════════════════════════════════════

  function setStatus(message, className = '') {
    const el = document.getElementById('ipc-status');
    if (!el) return;
    el.textContent = message;
    el.className = 'status' + (className ? ' ' + className : '');
  }

  // ═══════════════════════════════════════════════════════════════
  //  SLIDE CHANGE DETECTION
  // ═══════════════════════════════════════════════════════════════

  function watchForSlideChanges() {
    let debounceTimer = null;

    function checkForChange() {
      const newText = getSourceText();
      if (newText && newText !== state.currentText) {
        resetForNewSlide();
        loadReference();
      }
    }

    function debouncedCheck() {
      if (debounceTimer) clearTimeout(debounceTimer);
      debounceTimer = setTimeout(checkForChange, 300);
    }

    // Use MutationObserver on the slide container
    const container = queryFirst(CONFIG.selectors.slideContainer);
    if (!container) {
      // Fallback: poll every 3 seconds
      setInterval(checkForChange, 3000);
      return;
    }

    const observer = new MutationObserver((mutations) => {
      // Ignore mutations inside our own panel
      const panelMutation = mutations.every(m =>
        m.target.closest && m.target.closest('#ile-phonetic-panel')
      );
      if (panelMutation) return;

      debouncedCheck();
    });

    observer.observe(container, {
      childList: true,
      subtree: true,
      characterData: true,
    });
  }

  function resetForNewSlide() {
    state.referenceFeatures = null;
    state.referenceAudioUrl = null;
    state.learnerResult = null;
    state.comparisonResult = null;

    const resultsEl = document.getElementById('ipc-results');
    const emptyEl = document.getElementById('ipc-empty');
    const phonemeDiff = document.getElementById('ipc-phoneme-diff');
    if (resultsEl) resultsEl.style.display = 'none';
    if (emptyEl) emptyEl.style.display = 'block';
    if (phonemeDiff) phonemeDiff.style.display = 'none';
    setStatus('');
  }

  // ═══════════════════════════════════════════════════════════════
  //  INIT
  // ═══════════════════════════════════════════════════════════════

  function init() {
    console.log('[ILE Phonetic Coach] Initializing…');

    createPanel();
    loadReference();
    watchForSlideChanges();

    // Check if local phoneme server is running
    checkPhonemeServer();

    console.log('[ILE Phonetic Coach] Ready.');
  }

  // Wait for page to be ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    // Small delay to ensure ILE app has rendered
    setTimeout(init, 500);
  }

})();
