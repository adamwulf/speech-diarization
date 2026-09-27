// Main-thread audio: decoding any browser-supported audio (or the audio track of
// a video) to 16 kHz mono PCM, microphone capture with MediaRecorder, and the
// native-canvas level meter.

export const TARGET_SAMPLE_RATE = 16000;

/** An error whose message is written for the person using the page. */
export class UserFacingError extends Error {
  constructor(message, options) {
    super(message, options);
    this.name = 'UserFacingError';
  }
}

/**
 * Decodes a recording or file and returns 16 kHz mono PCM (Float32Array).
 * decodeAudioData can also read the audio track of some video files (for
 * example, MP4 in Chrome); browser support for video containers varies.
 * Decoding in an OfflineAudioContext resamples to 16 kHz; rendering through a
 * 1-channel context then down-mixes with the standard Web Audio rules.
 */
export async function decodeToMono16k(blob) {
  if (typeof OfflineAudioContext === 'undefined') {
    throw new UserFacingError('This browser cannot decode audio (Web Audio is unavailable). Use a current version of Chrome, Edge, Firefox, or Safari.');
  }
  let data;
  try {
    data = await blob.arrayBuffer();
  } catch (cause) {
    throw new UserFacingError('The audio could not be read. Choose the file again.', { cause });
  }
  if (!data.byteLength) {
    throw new UserFacingError('The audio is empty. Record again or choose a different file.');
  }

  let decoded;
  try {
    decoded = await new OfflineAudioContext(1, 1, TARGET_SAMPLE_RATE).decodeAudioData(data);
  } catch (cause) {
    throw new UserFacingError('This file could not be decoded. Try a common format such as WAV, MP3, M4A, WebM, or MP4. A video file must have an audio track.', { cause });
  }

  const length = Math.round(decoded.duration * TARGET_SAMPLE_RATE);
  if (length < 1) {
    throw new UserFacingError('The audio is empty. Record again or choose a different file.');
  }
  const offline = new OfflineAudioContext(1, length, TARGET_SAMPLE_RATE);
  const source = offline.createBufferSource();
  source.buffer = decoded;
  source.connect(offline.destination);
  source.start();
  const rendered = await offline.startRendering();
  return rendered.getChannelData(0);
}

function microphoneErrorMessage(error) {
  switch (error?.name) {
    case 'NotAllowedError':
    case 'SecurityError':
      return 'Microphone access was blocked. Allow microphone access for this page in your browser’s site settings, then click “Start recording” again.';
    case 'NotFoundError':
    case 'OverconstrainedError':
      return 'No microphone was found. Connect a microphone, then click “Start recording” again.';
    case 'NotReadableError':
    case 'AbortError':
      return 'The microphone could not be started — another app may be using it. Close that app, then click “Start recording” again.';
    default:
      return `The microphone could not be started${error?.message ? ` (${error.message})` : ''}. Click “Start recording” to try again, or use “Transcribe file…” instead.`;
  }
}

/**
 * Asks for the microphone and starts recording. Resolves to a session:
 * - `analyser`: AnalyserNode for the level meter, or null if unavailable.
 * - `stop()`: finishes recording and resolves to the recorded Blob.
 * - `release()`: discards the recording.
 * Both stop() and release() stop every track and close the meter's context.
 * `onInterrupted` fires when recording ends without the page asking, so the
 * page can call stop() and leave the recording state:
 * - an audio track ended (e.g. the device was unplugged): stop() resolves
 *   with the audio captured so far;
 * - the recorder reported an error: stop() rejects with a UserFacingError and
 *   the partial audio is discarded.
 */
export async function startMicrophone({ onInterrupted } = {}) {
  if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) {
    throw new UserFacingError('Microphone recording needs a secure page (https:// or http://localhost). Open the demo from a secure address, or use “Transcribe file…” instead.');
  }
  if (typeof MediaRecorder === 'undefined') {
    throw new UserFacingError('This browser can’t record audio. Use “Transcribe file…” instead.');
  }

  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch (cause) {
    throw new UserFacingError(microphoneErrorMessage(cause), { cause });
  }

  let context = null;
  let released = false;
  const releaseDevices = () => {
    if (released) return;
    released = true;
    for (const track of stream.getTracks()) track.stop();
    if (context) context.close().catch(() => {});
  };

  try {
    const recorder = new MediaRecorder(stream);
    const chunks = [];
    recorder.addEventListener('dataavailable', (event) => {
      if (event.data && event.data.size > 0) chunks.push(event.data);
    });
    const finished = new Promise((resolve, reject) => {
      recorder.addEventListener('stop', () => {
        resolve(new Blob(chunks, { type: recorder.mimeType || chunks[0]?.type || 'audio/webm' }));
      });
      recorder.addEventListener('error', (event) => {
        const detail = event.error?.message || 'the recorder stopped unexpectedly';
        reject(new UserFacingError(`Recording failed: ${detail}. Click “Start recording” to try again.`));
        onInterrupted?.();
      });
    });
    finished.catch(() => {}); // Surfaced through stop(); avoid an unhandled rejection after release().

    // The level meter is a nicety; recording works without it.
    let analyser = null;
    try {
      const AudioContextClass = window.AudioContext || window.webkitAudioContext;
      context = new AudioContextClass();
      analyser = context.createAnalyser();
      analyser.fftSize = 1024;
      context.createMediaStreamSource(stream).connect(analyser);
      if (context.state === 'suspended') context.resume().catch(() => {});
    } catch {
      analyser = null;
    }

    for (const track of stream.getAudioTracks()) {
      track.addEventListener('ended', () => onInterrupted?.(), { once: true });
    }
    recorder.start();

    return {
      analyser,
      async stop() {
        if (recorder.state !== 'inactive') recorder.stop();
        try {
          return await finished;
        } finally {
          releaseDevices();
        }
      },
      release() {
        if (recorder.state !== 'inactive') {
          try { recorder.stop(); } catch { /* already stopping */ }
        }
        releaseDevices();
      },
    };
  } catch (cause) {
    releaseDevices();
    throw new UserFacingError(`The microphone could not be started (${cause.message}). Click “Start recording” to try again, or use “Transcribe file…” instead.`, { cause });
  }
}

/** Peak absolute amplitude (0..1) of unsigned 8-bit time-domain samples. */
export function peakLevel(samples) {
  let peak = 0;
  for (const sample of samples) {
    const value = Math.abs(sample - 128) / 128;
    if (value > peak) peak = value;
  }
  return peak;
}

/**
 * Horizontal mic level bar drawn on a native canvas, matching the reference
 * meter: --bar-bg track, --accent fill, --chosen once the level runs hot.
 * Animates only while attached (plus a short decay), so an idle page is idle.
 */
export class LevelMeter {
  #canvas;
  #ctx;
  #analyser = null;
  #samples = null;
  #level = 0;
  #frame = 0;
  #width = 0;
  #height = 0;
  #colors;

  constructor(canvas) {
    this.#canvas = canvas;
    this.#ctx = canvas.getContext('2d');
    const css = getComputedStyle(document.documentElement);
    const token = (name, fallback) => css.getPropertyValue(name).trim() || fallback;
    this.#colors = {
      track: token('--bar-bg', '#2b3242'),
      normal: token('--accent', '#4da3ff'),
      hot: token('--chosen', '#ffd166'),
    };
    this.#resize();
    new ResizeObserver(() => {
      this.#resize();
      this.#draw();
    }).observe(canvas);
    this.#draw();
  }

  attach(analyser) {
    this.#analyser = analyser;
    this.#samples = analyser ? new Uint8Array(analyser.fftSize) : null;
    this.#run();
  }

  detach() {
    this.#analyser = null;
    this.#samples = null;
    this.#run();
  }

  #resize() {
    const ratio = window.devicePixelRatio || 1;
    this.#width = this.#canvas.clientWidth || 400;
    this.#height = this.#canvas.clientHeight || 64;
    this.#canvas.width = Math.round(this.#width * ratio);
    this.#canvas.height = Math.round(this.#height * ratio);
    this.#ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  }

  #run() {
    if (this.#frame) return;
    const tick = () => {
      this.#frame = 0;
      let target = 0;
      if (this.#analyser && this.#samples) {
        this.#analyser.getByteTimeDomainData(this.#samples);
        target = peakLevel(this.#samples);
      }
      this.#level += (target - this.#level) * 0.3;
      if (!this.#analyser && this.#level < 0.002) this.#level = 0;
      this.#draw();
      if (this.#analyser || this.#level > 0) this.#frame = requestAnimationFrame(tick);
    };
    this.#frame = requestAnimationFrame(tick);
  }

  #draw() {
    const ctx = this.#ctx;
    ctx.fillStyle = this.#colors.track;
    ctx.fillRect(0, 0, this.#width, this.#height);
    const fill = this.#width * Math.min(1, this.#level * 1.6);
    if (fill > 0) {
      ctx.fillStyle = this.#level > 0.6 ? this.#colors.hot : this.#colors.normal;
      ctx.fillRect(0, 0, fill, this.#height);
    }
  }
}
