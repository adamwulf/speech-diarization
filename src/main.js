// Page controller. The main thread records or decodes audio to 16 kHz mono PCM,
// keeps it for re-diarization, and hands copies to the inference worker, which
// owns every model download and inference (see IMPLEMENTATION.md for the protocol).

import { ASR_MODELS, DEFAULT_ASR_MODEL, downloadSize, getAsrModel } from './asr-models.js';
import { LevelMeter, TARGET_SAMPLE_RATE, UserFacingError, decodeToMono16k, startMicrophone } from './audio.js';
import {
  EXPORT_FORMATS,
  MAX_SPEAKER_NAME_LENGTH,
  cleanSpeakerName,
  createSpeakerNamer,
  defaultSpeakerName,
  exportFileName,
  formatClock,
  groupTurns,
  normalizeSegments,
  speakerOrder,
  toMarkdown,
  toWebVTT,
} from './exports.js';

// ---- DOM ----
const $ = (id) => document.getElementById(id);
const micBtn = $('micBtn');
const fileBtn = $('fileBtn');
const fileInput = $('fileInput');
const clearBtn = $('clearBtn');
const speakersToggle = $('speakersToggle');
const asrModelSelect = $('asrModelSelect');
const asrModel = $('asrModel');
const diarModel = $('diarModel');
const track = $('track');
const fill = $('fill');
const statusEl = $('status');
const retryRow = $('retryRow');
const retryBtn = $('retryBtn');
const transcriptPanel = $('transcriptPanel');
const sourceBlock = $('sourceBlock');
const sourceInfo = $('sourceInfo');
const player = $('player');
const speakersSection = $('speakersSection');
const speakerList = $('speakerList');
const transcriptEmpty = $('transcriptEmpty');
const transcriptList = $('transcriptList');
const exportMdBtn = $('exportMdBtn');
const exportVttBtn = $('exportVttBtn');

const SPEAKER_COLOR_COUNT = 6; // matches the [data-color] rules in style.css

const STAGE_TEXT = {
  'asr-load': 'Loading the speech-recognition model…',
  asr: 'Transcribing…',
  'diarization-load': 'Loading the speaker-detection models…',
  diarization: 'Detecting speakers…',
};

// ---- State ----
const state = {
  // 'idle' | 'starting' (mic permission prompt) | 'recording' | 'busy' (decode or inference)
  phase: 'idle',
  // Current audio: { kind: 'recording'|'file', name, blob, audio: Float32Array, duration, url }
  source: null,
  // Normalized segments for display and exports.
  segments: [],
  // The worker's segments exactly as received (including word timings); sent
  // back unchanged in a diarize request so speakers align at word boundaries.
  workerSegments: [],
  // True once the current segments carry speaker ids from diarization.
  diarized: false,
  // Speaker id -> custom name. Reset whenever the source or its speaker ids change.
  speakerNames: new Map(),
  // Pending retry: { label, kind: 'transcribe'|'diarize', run }
  retry: null,
};

let mic = null;
let recordingTimer = 0;
let recordingStartedAt = 0;

// ---- Inference worker client ----
// Created lazily, so nothing is downloaded until the first transcription.
// A crashed or unloadable worker is discarded and recreated on the next run.
class InferenceClient {
  #worker = null;
  #nextId = 1;
  #pending = null;

  run(request, onProgress) {
    if (this.#pending) return Promise.reject(new Error('Another task is still running.'));
    const id = this.#nextId++;
    return new Promise((resolve, reject) => {
      this.#pending = { id, resolve, reject, onProgress };
      try {
        const worker = this.#ensureWorker();
        // Transfer a copy; the page keeps its own PCM for later re-diarization.
        const audio = request.audio.slice();
        worker.postMessage({ ...request, id, audio }, [audio.buffer]);
      } catch (error) {
        this.#pending = null;
        reject(error);
      }
    });
  }

  #ensureWorker() {
    if (!this.#worker) {
      const worker = new Worker(new URL('./inference-worker.js', import.meta.url), { type: 'module' });
      worker.addEventListener('message', (event) => this.#handleMessage(event.data));
      worker.addEventListener('error', (event) => {
        event.preventDefault();
        this.#discard(worker, event.message
          ? `the speech engine crashed (${event.message})`
          : 'the speech engine could not start');
      });
      worker.addEventListener('messageerror', () => this.#discard(worker, 'the speech engine sent an unreadable message'));
      this.#worker = worker;
    }
    return this.#worker;
  }

  #handleMessage(message) {
    const pending = this.#pending;
    if (!pending || !message || message.id !== pending.id) return;
    if (message.type === 'progress') {
      pending.onProgress?.(message);
    } else if (message.type === 'result') {
      this.#pending = null;
      pending.resolve(message);
    } else if (message.type === 'error') {
      this.#pending = null;
      pending.reject(new Error(message.message || 'the speech engine reported an unknown error'));
    }
  }

  #discard(worker, detail) {
    if (worker !== this.#worker) return;
    worker.terminate();
    this.#worker = null;
    const pending = this.#pending;
    this.#pending = null;
    pending?.reject(new Error(detail));
  }
}

const inference = new InferenceClient();
const meter = new LevelMeter($('meter'));

// ---- Small helpers ----
function messageOf(error) {
  const text = error instanceof Error ? error.message : typeof error === 'string' ? error : '';
  return text.trim() || 'an unknown error occurred';
}

/** Joins sentence fragments, adding a period where one is missing. */
function sentences(...parts) {
  return parts
    .filter(Boolean)
    .map((part) => (/[.!?…]$/.test(part) ? part : `${part}.`))
    .join(' ');
}

function plural(count, noun) {
  return `${count} ${noun}${count === 1 ? '' : 's'}`;
}

/** Speaker detection is on and has run on the current transcript. */
function speakersDetected() {
  return speakersToggle.checked && state.diarized;
}

/**
 * Show speaker labels only when detection found at least one speaker; if
 * every segment is unattributed, the transcript keeps plain per-segment
 * timestamps instead of one merged "Unknown speaker" turn.
 */
function showSpeakers() {
  return speakersDetected() && speakerOrder(state.segments).length > 0;
}

function keptNote() {
  return state.source ? 'The previous transcript is still shown' : '';
}

// ---- Status, progress, controls ----
// The status line is an aria-live region. Its changing number (download
// percentage, recording time) sits in an aria-hidden span that is updated in
// place, so frequent ticks don't re-announce the whole message.
let statusKey = null;
let statusValue = null;

function metaValue() {
  const span = document.createElement('span');
  span.className = 'meta-value';
  span.setAttribute('aria-hidden', 'true');
  return span;
}

function setStatusTone(tone) {
  statusEl.classList.toggle('error', tone === 'error');
  statusEl.classList.toggle('warning', tone === 'warning');
}

function setStatus(text, { tone = 'info', value = '' } = {}) {
  if (statusKey !== text) {
    statusValue = metaValue();
    statusEl.replaceChildren(text, statusValue);
    statusKey = text;
  }
  statusValue.textContent = value ? ` ${value}` : '';
  setStatusTone(tone);
}

function setRecordingStatus(elapsed) {
  if (statusKey !== 'recording') {
    const dot = document.createElement('span');
    dot.className = 'rec-dot';
    dot.setAttribute('aria-hidden', 'true');
    statusValue = metaValue();
    statusEl.replaceChildren(dot, 'Recording ', statusValue, ' — click “Stop recording” to transcribe.');
    statusKey = 'recording';
  }
  statusValue.textContent = elapsed;
  setStatusTone('info');
}

/** `null` hides the bar, 'indeterminate' animates it, a number 0..1 fills it. */
function setProgress(value) {
  if (value === null) {
    track.hidden = true;
    track.classList.remove('indeterminate');
    fill.style.width = '0%';
    track.removeAttribute('aria-valuenow');
    return;
  }
  track.hidden = false;
  if (value === 'indeterminate') {
    track.classList.add('indeterminate');
    fill.style.width = '';
    track.removeAttribute('aria-valuenow');
  } else {
    const percent = Math.round(Math.min(1, Math.max(0, value)) * 100);
    track.classList.remove('indeterminate');
    fill.style.width = `${percent}%`;
    track.setAttribute('aria-valuenow', String(percent));
  }
}

function renderAsrModel() {
  const model = getAsrModel(asrModelSelect.value);
  asrModel.textContent = `${model.label} · speech-to-text · ${downloadSize(model.bytes)} · runs locally`;
}

function setActiveModel(stage) {
  asrModel.classList.toggle('active', stage === 'asr-load' || stage === 'asr');
  diarModel.classList.toggle('active', stage === 'diarization-load' || stage === 'diarization');
}

function handleProgress(message) {
  setActiveModel(message.stage);
  const determinate = typeof message.progress === 'number' && Number.isFinite(message.progress);
  setProgress(determinate ? message.progress : 'indeterminate');
  const text = (typeof message.message === 'string' && message.message.trim())
    || STAGE_TEXT[message.stage]
    || 'Working…';
  setStatus(text, { value: determinate ? `${Math.round(Math.min(1, Math.max(0, message.progress)) * 100)}%` : '' });
}

function setRetry(retry) {
  state.retry = retry;
  if (retry) retryBtn.textContent = retry.label;
}

function updateControls() {
  const { phase } = state;
  const idle = phase === 'idle';
  const recording = phase === 'recording';

  micBtn.disabled = !(idle || recording);
  micBtn.textContent = recording ? 'Stop recording' : 'Start recording';
  micBtn.classList.toggle('recording', recording);
  fileBtn.disabled = !idle;
  clearBtn.disabled = !idle || (!state.source && !state.retry);
  speakersToggle.disabled = !idle;
  asrModelSelect.disabled = !idle;
  // Hiding or disabling the focused Retry button would drop keyboard focus to
  // the page; park it on the status line, which announces what happens next.
  if (retryRow.contains(document.activeElement) && (!state.retry || !idle)) statusEl.focus();
  retryRow.hidden = !state.retry;
  retryBtn.disabled = !idle;

  const canExport = Boolean(state.source) && state.segments.length > 0;
  exportMdBtn.disabled = !canExport;
  exportVttBtn.disabled = !canExport;

  transcriptPanel.setAttribute('aria-busy', String(phase === 'busy'));
  diarModel.classList.toggle('off', !speakersToggle.checked);
}

/** Leaves the busy phase: hides progress and re-enables controls. */
function finishTask() {
  state.phase = 'idle';
  setProgress(null);
  setActiveModel(null);
  render();
}

// ---- Rendering ----
function renderSource() {
  const source = state.source;
  sourceBlock.hidden = !source;
  if (!source) {
    if (player.hasAttribute('src')) {
      player.removeAttribute('src');
      player.load();
    }
    return;
  }
  if (player.getAttribute('src') !== source.url) player.src = source.url;

  const name = document.createElement('strong');
  name.textContent = source.kind === 'recording' ? 'Microphone recording' : source.name;
  const details = [formatClock(source.duration)];
  if (showSpeakers()) details.push(plural(speakerOrder(state.segments).length, 'speaker'));
  sourceInfo.replaceChildren(name, ` · ${details.join(' · ')}`);
}

function renderSpeakers() {
  const ids = showSpeakers() ? speakerOrder(state.segments) : [];
  speakersSection.hidden = ids.length === 0;
  const rows = document.createDocumentFragment();
  ids.forEach((id, index) => {
    const row = document.createElement('div');
    row.className = 'speaker-row';
    row.dataset.color = String(index % SPEAKER_COLOR_COUNT);

    const inputId = `speaker-name-${index}`;
    const label = document.createElement('label');
    label.htmlFor = inputId;
    const swatch = document.createElement('span');
    swatch.className = 'swatch';
    swatch.setAttribute('aria-hidden', 'true');
    const hint = document.createElement('span');
    hint.className = 'sr-only';
    hint.textContent = ' name';
    label.append(swatch, defaultSpeakerName(index), hint);

    const input = document.createElement('input');
    input.type = 'text';
    input.id = inputId;
    input.maxLength = MAX_SPEAKER_NAME_LENGTH;
    input.autocomplete = 'off';
    input.spellcheck = false;
    input.placeholder = defaultSpeakerName(index);
    input.value = state.speakerNames.get(id) ?? '';
    input.addEventListener('input', () => {
      const name = cleanSpeakerName(input.value);
      if (name) state.speakerNames.set(id, name);
      else state.speakerNames.delete(id);
      renderTranscript();
    });
    // Show the stored (trimmed) form once editing is done.
    input.addEventListener('change', () => { input.value = state.speakerNames.get(id) ?? ''; });

    row.append(label, input);
    rows.append(row);
  });
  speakerList.replaceChildren(rows);
}

function renderTranscript() {
  const segments = state.segments;
  const show = showSpeakers();
  const nameOf = createSpeakerNamer(segments, state.speakerNames, show);
  const colors = new Map(speakerOrder(segments).map((id, i) => [id, String(i % SPEAKER_COLOR_COUNT)]));

  const items = document.createDocumentFragment();
  for (const turn of groupTurns(segments, { showSpeakers: show })) {
    const item = document.createElement('li');
    item.className = 'turn';
    const meta = document.createElement('div');
    meta.className = 'turn-meta';

    if (show) {
      const who = document.createElement('span');
      who.className = 'turn-speaker';
      who.textContent = nameOf(turn.speaker);
      if (turn.speaker === null) who.classList.add('unknown');
      else item.dataset.color = colors.get(turn.speaker);
      meta.append(who);
    }
    const time = document.createElement('span');
    time.className = 'turn-time';
    time.textContent = `${formatClock(turn.start)} – ${formatClock(turn.end)}`;
    meta.append(time);

    const text = document.createElement('p');
    text.className = 'turn-text';
    text.textContent = turn.text;
    item.append(meta, text);
    items.append(item);
  }
  const hasTurns = items.childNodes.length > 0;
  transcriptList.replaceChildren(items);
  transcriptList.hidden = !hasTurns;
  transcriptEmpty.hidden = hasTurns;
  transcriptEmpty.textContent = state.source
    ? 'No speech was detected in this audio.'
    : 'No transcript yet. Record or choose an audio file to get started.';
}

function render() {
  renderSource();
  renderSpeakers();
  renderTranscript();
  updateControls();
}

/** Status after a successful task. `warning` is the worker's optional note. */
function setDoneStatus(warning) {
  const count = speakerOrder(state.segments).length;
  if (speakersDetected() && !count) {
    // Say explicitly that detection ran but found nobody.
    setStatus(sentences(
      'Done',
      warning || 'No speakers could be identified',
      'The transcript is shown without speaker names',
    ), { tone: 'warning' });
    return;
  }
  const done = speakersDetected()
    ? `Done — ${plural(count, 'speaker')} detected. Rename speakers or export the transcript.`
    : 'Done. Export the transcript below.';
  setStatus(sentences(done, warning), { tone: warning ? 'warning' : 'info' });
}

/**
 * "<prefix>: <error>", unless the worker's message (already a full sentence)
 * starts with the same words.
 */
function failureText(prefix, error) {
  const message = messageOf(error);
  return message.toLowerCase().startsWith(prefix.toLowerCase()) ? message : `${prefix}: ${message}`;
}

// ---- Transcription and diarization ----
async function processSource({ blob, kind, name }) {
  state.phase = 'busy';
  setRetry(null);
  updateControls();
  setProgress('indeterminate');
  setStatus('Decoding audio…');

  let audio;
  try {
    audio = await decodeToMono16k(blob);
  } catch (error) {
    const detail = error instanceof UserFacingError ? error.message : `The audio could not be decoded: ${messageOf(error)}`;
    finishTask();
    setStatus(sentences(detail, keptNote()), { tone: 'error' });
    return;
  }
  await transcribe({ kind, name, blob, audio, duration: audio.length / TARGET_SAMPLE_RATE });
}

async function transcribe(candidate) {
  const speakers = speakersToggle.checked;
  const model = asrModelSelect.value;
  state.phase = 'busy';
  setRetry(null);
  updateControls();
  setProgress('indeterminate');
  setStatus(`Preparing to transcribe ${formatClock(candidate.duration)} of audio…`);

  let result;
  try {
    result = await inference.run(
      { type: 'transcribe', audio: candidate.audio, speakers, asrModel: model },
      handleProgress,
    );
  } catch (error) {
    setRetry({ label: 'Retry transcription', kind: 'transcribe', run: () => transcribe(candidate) });
    finishTask();
    setStatus(sentences(
      failureText('Transcription failed', error),
      'Click “Retry transcription” to try again',
      keptNote(),
    ), { tone: 'error' });
    return;
  }

  // Success: this source replaces the previous audio, transcript, and names.
  if (state.source) URL.revokeObjectURL(state.source.url);
  state.source = { ...candidate, url: URL.createObjectURL(candidate.blob) };
  state.workerSegments = Array.isArray(result.segments) ? result.segments : [];
  state.segments = normalizeSegments(state.workerSegments);
  state.diarized = speakers && result.diarized === true;
  state.speakerNames = new Map();

  const speakerDetectionFailed = speakers && !state.diarized && state.segments.length > 0;
  if (speakerDetectionFailed) {
    setRetry({ label: 'Retry speaker detection', kind: 'diarize', run: diarize });
  }
  finishTask();
  if (speakerDetectionFailed) {
    setStatus(sentences(
      result.warning || 'Speaker detection returned no speaker data. The transcript is shown without speaker names',
      'Click “Retry speaker detection” to try again',
    ), { tone: 'warning' });
  } else if (!state.segments.length) {
    setStatus(sentences(result.warning || 'No speech was detected', 'Try another recording or file'), { tone: 'warning' });
  } else {
    setDoneStatus(result.warning);
  }
}

/** Adds speaker labels to the current transcript using the retained audio. */
async function diarize() {
  const source = state.source;
  if (!source || !state.segments.length) return;

  state.phase = 'busy';
  setRetry(null);
  updateControls();
  setProgress('indeterminate');
  setStatus('Preparing speaker detection…');

  const fail = (error) => {
    setRetry({ label: 'Retry speaker detection', kind: 'diarize', run: diarize });
    finishTask();
    setStatus(sentences(
      failureText('Speaker detection failed', error),
      'The transcript is unchanged',
      'Click “Retry speaker detection” to try again',
    ), { tone: 'error' });
  };

  let result;
  try {
    result = await inference.run(
      { type: 'diarize', audio: source.audio, segments: state.workerSegments },
      handleProgress,
    );
  } catch (error) {
    fail(error);
    return;
  }
  const next = normalizeSegments(result.segments);
  if (!result.diarized || !next.length) {
    fail(result.warning || 'no speaker data was returned');
    return;
  }

  state.workerSegments = result.segments;
  state.segments = next;
  state.diarized = true;
  state.speakerNames = new Map();
  finishTask();
  setDoneStatus(result.warning);
}

// ---- Recording ----
function renderRecordingStatus() {
  setRecordingStatus(formatClock((performance.now() - recordingStartedAt) / 1000));
}

function endRecordingUi() {
  clearInterval(recordingTimer);
  recordingTimer = 0;
  meter.detach();
}

async function startRecording() {
  state.phase = 'starting';
  updateControls();
  setStatus('Waiting for microphone access…');

  let session;
  try {
    session = await startMicrophone({ onInterrupted: () => stopRecording() });
  } catch (error) {
    state.phase = 'idle';
    updateControls();
    setStatus(error instanceof UserFacingError ? error.message : sentences(`The microphone could not be started: ${messageOf(error)}`), { tone: 'error' });
    return;
  }

  mic = session;
  state.phase = 'recording';
  // A new recording supersedes any failed attempt still offering a retry.
  setRetry(null);
  recordingStartedAt = performance.now();
  meter.attach(session.analyser);
  renderRecordingStatus();
  recordingTimer = setInterval(renderRecordingStatus, 500);
  updateControls();
}

async function stopRecording() {
  if (state.phase !== 'recording' || !mic) return;
  const session = mic;
  mic = null;
  endRecordingUi();
  state.phase = 'busy';
  updateControls();
  setProgress('indeterminate');
  setStatus('Finishing the recording…');

  let blob;
  try {
    blob = await session.stop();
  } catch (error) {
    finishTask();
    setStatus(sentences(error instanceof UserFacingError ? error.message : `Recording failed: ${messageOf(error)}`, keptNote()), { tone: 'error' });
    return;
  }
  await processSource({ blob, kind: 'recording', name: recordingName() });
}

function recordingName() {
  const now = new Date();
  const pad = (n) => String(n).padStart(2, '0');
  return `recording-${now.getFullYear()}-${pad(now.getMonth() + 1)}-${pad(now.getDate())}-${pad(now.getHours())}-${pad(now.getMinutes())}-${pad(now.getSeconds())}`;
}

// ---- Exports ----
function download(format) {
  const source = state.source;
  if (!source || !state.segments.length) return;
  const options = {
    speakerNames: state.speakerNames,
    showSpeakers: showSpeakers(),
    sourceName: source.name,
    duration: source.duration,
  };
  const text = format === 'markdown' ? toMarkdown(state.segments, options) : toWebVTT(state.segments, options);
  const { extension, mimeType } = EXPORT_FORMATS[format];
  const url = URL.createObjectURL(new Blob([text], { type: mimeType }));
  const link = document.createElement('a');
  link.href = url;
  link.download = exportFileName(source.name, extension);
  document.body.append(link);
  link.click();
  link.remove();
  // Give the browser time to start the download before revoking.
  setTimeout(() => URL.revokeObjectURL(url), 10_000);
}

// ---- Events ----
micBtn.addEventListener('click', () => {
  if (state.phase === 'recording') stopRecording();
  else if (state.phase === 'idle') startRecording();
});

fileBtn.addEventListener('click', () => {
  if (state.phase === 'idle') fileInput.click();
});

fileInput.addEventListener('change', () => {
  const file = fileInput.files?.[0];
  fileInput.value = ''; // so picking the same file again still fires 'change'
  if (file && state.phase === 'idle') processSource({ blob: file, kind: 'file', name: file.name });
});

clearBtn.addEventListener('click', () => {
  if (state.phase !== 'idle') return;
  if (state.source) URL.revokeObjectURL(state.source.url);
  state.source = null;
  state.segments = [];
  state.workerSegments = [];
  state.diarized = false;
  state.speakerNames = new Map();
  setRetry(null);
  render();
  setStatus('Cleared. Click “Start recording”, or transcribe an audio file.');
});

retryBtn.addEventListener('click', () => {
  if (state.phase === 'idle' && state.retry) state.retry.run();
});

speakersToggle.addEventListener('change', () => {
  if (state.phase !== 'idle') return;
  // A failed new recording or file is waiting for "Retry transcription", which
  // reads this setting when it runs. Keep that retry, and don't start
  // speaker detection on the older transcript still on screen.
  if (state.retry?.kind === 'transcribe') {
    render();
    setStatus(speakersToggle.checked
      ? '“Detect speakers” is on: “Retry transcription” will also detect speakers in the new audio.'
      : '“Detect speakers” is off: “Retry transcription” will skip speaker detection, and speaker names are hidden.');
    return;
  }
  if (!speakersToggle.checked) {
    if (state.retry?.kind === 'diarize') setRetry(null);
    render();
    if (state.source) {
      setStatus('Speaker names are hidden in the transcript and exports. Turn “Detect speakers” back on to show them.');
    }
    return;
  }
  if (state.source && state.segments.length && !state.diarized) {
    diarize();
    return;
  }
  render();
  if (state.source && state.diarized) {
    setStatus(showSpeakers()
      ? 'Speaker names are shown in the transcript and exports.'
      : 'No speakers were identified in this audio, so the transcript has no speaker names.');
  }
});

asrModelSelect.addEventListener('change', renderAsrModel);

exportMdBtn.addEventListener('click', () => download('markdown'));
exportVttBtn.addEventListener('click', () => download('webvtt'));

// Never leave the microphone open when the page goes away (including bfcache).
window.addEventListener('pagehide', () => {
  if (!mic) return;
  mic.release();
  mic = null;
  endRecordingUi();
  state.phase = 'idle';
  updateControls();
  setStatus('Recording stopped because the page was closed or hidden. Click “Start recording” to record again.', { tone: 'warning' });
});

// ---- Startup ----
for (const [key, model] of Object.entries(ASR_MODELS)) {
  asrModelSelect.add(new Option(`${model.level} — ${model.label} (${downloadSize(model.bytes)})`, key));
}
asrModelSelect.value = DEFAULT_ASR_MODEL;
renderAsrModel();

if (typeof Worker === 'undefined' || typeof OfflineAudioContext === 'undefined') {
  render();
  micBtn.disabled = true;
  fileBtn.disabled = true;
  speakersToggle.disabled = true;
  asrModelSelect.disabled = true;
  setStatus('This browser is missing Web Workers or Web Audio, which this demo needs. Use a current version of Chrome, Edge, Firefox, or Safari.', { tone: 'error' });
} else {
  render();
}
