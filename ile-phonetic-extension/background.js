/**
 * ILE Phonetic Coach — Background Service Worker
 * 
 * Bridges the content script and the native messaging host.
 * Content script sends audio → background forwards to Python → returns phonemes.
 */

const HOST_NAME = 'com.ile.phoneme_host';

let nativePort = null;
let pendingResolve = null;
let hostAvailable = null; // null = unknown

/**
 * Connect to the native host. Keeps connection alive for fast subsequent calls.
 */
function connectToHost() {
  if (nativePort) return true;

  try {
    nativePort = chrome.runtime.connectNative(HOST_NAME);

    nativePort.onMessage.addListener((msg) => {
      console.log('[IPC Background] Native message:', msg);

      if (msg.status === 'ready') {
        hostAvailable = true;
        console.log('[IPC Background] Native host ready');
        return;
      }

      if (pendingResolve) {
        pendingResolve(msg);
        pendingResolve = null;
      }
    });

    nativePort.onDisconnect.addListener(() => {
      const err = chrome.runtime.lastError;
      console.warn('[IPC Background] Native host disconnected:', err?.message || 'unknown');
      nativePort = null;
      hostAvailable = false;

      if (pendingResolve) {
        pendingResolve({ error: err?.message || 'Host disconnected' });
        pendingResolve = null;
      }
    });

    return true;
  } catch (err) {
    console.error('[IPC Background] Failed to connect:', err);
    hostAvailable = false;
    return false;
  }
}

/**
 * Send a message to the native host and wait for response.
 */
function sendToHost(message) {
  return new Promise((resolve) => {
    if (!nativePort) {
      if (!connectToHost()) {
        resolve({ error: 'Could not connect to native host. Run install_host.py first.' });
        return;
      }
    }

    pendingResolve = resolve;

    // Timeout after 30 seconds
    setTimeout(() => {
      if (pendingResolve === resolve) {
        pendingResolve = null;
        resolve({ error: 'Timeout — model may still be loading' });
      }
    }, 30000);

    nativePort.postMessage(message);
  });
}

/**
 * Handle messages from content scripts.
 */
chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (msg.type === 'phoneme-health') {
    // Quick check
    if (hostAvailable === true) {
      sendResponse({ status: 'ok' });
    } else if (hostAvailable === false) {
      sendResponse({ status: 'unavailable' });
    } else {
      // Try connecting
      connectToHost();
      // Give it a moment
      setTimeout(() => {
        sendResponse({ status: hostAvailable ? 'ok' : 'unavailable' });
      }, 2000);
    }
    return true; // async response
  }

  if (msg.type === 'phoneme-analyze') {
    (async () => {
      const result = await sendToHost({
        action: 'analyze',
        audio: msg.audio,
        sampleRate: msg.sampleRate || 44100,
        format: msg.format || 'file',
      });
      sendResponse(result);
    })();
    return true; // async response
  }

  return false;
});

// Try connecting on startup
connectToHost();
