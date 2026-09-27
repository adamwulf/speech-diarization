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
const asrModelInfo = $('asrModelInfo');
const asrModelState = $('asrModelState');
const diarModel = $('diarModel');
const diarModelState = $('diarModelState');
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
  // What each model line shows: { state: 'queued'|'loading'|'ready'|'failed', progress?: 0..1 }.
  // Whisper entries are per model key, so each selection shows its own state.
  models: { asr: new Map(), diarization: null },
};

let mic = null;
// A transcribe or diarize request is waiting in the worker queue.
let taskRunning = false;
let asrPreloadTimer = 0;
let recordingTimer = 0;
let recordingStartedAt = 0;

// ---- Inference worker client ----
// Created on the first request (the model preloads at page startup). The worker
// runs requests one at a time. A crashed or unloadable worker is discarded
// and recreated on the next request.
class InferenceClient {
  #worker = null;
  #nextId = 1;
  #pending = new Map(); // id -> {resolve, reject, onProgress}
  #onModelState;
  #onCrash;

  /**
   * `onModelState(message)` gets the worker's `{type: 'model'}` messages.
   * `onCrash()` runs after a crash rejects the pending requests.
   */
  constructor({ onModelState, onCrash }) {
    this.#onModelState = onModelState;
    this.#onCrash = onCrash;
  }

  run(request, onProgress) {
    const id = this.#nextId++;
    return new Promise((resolve, reject) => {
      this.#pending.set(id, { resolve, reject, onProgress });
      try {
        const worker = this.#ensureWorker();
        if (request.audio) {
          // Transfer a copy; the page keeps its own PCM for later re-diarization.
          const audio = request.audio.slice();
          worker.postMessage({ ...request, id, audio }, [audio.buffer]);
        } else {
          worker.postMessage({ ...request, id });
        }
      } catch (error) {
        this.#pending.delete(id);
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
    if (message?.type === 'model') {
      this.#onModelState(message);
      return;
    }
    const pending = this.#pending.get(message?.id);
    if (!pending) return;
    if (message.type === 'progress') {
      pending.onProgress?.(message);
    } else if (message.type === 'result') {
      this.#pending.delete(message.id);
      pending.resolve(message);
    } else if (message.type === 'error') {
      this.#pending.delete(message.id);
      pending.reject(new Error(message.message || 'the speech engine reported an unknown error'));
    }
  }

  #discard(worker, detail) {
    if (worker !== this.#worker) return;
    worker.terminate();
    this.#worker = null;
    const pending = [...this.#pending.values()];
    this.#pending.clear();
    for (const request of pending) request.reject(new Error(detail));
    this.#onCrash();
  }
}

const inference = new InferenceClient({ onModelState: handleModelState, onCrash: handleCrash });
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

/** A 0..1 fraction as a whole percentage, clamped to 0..100. */
function percent(fraction) {
  return Math.round(Math.min(1, Math.max(0, fraction)) * 100);
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
    const filled = percent(value);
    track.classList.remove('indeterminate');
    fill.style.width = `${filled}%`;
    track.setAttribute('aria-valuenow', String(filled));
  }
}

function modelStateText(entry) {
  switch (entry?.state) {
    // "loading", not "downloading": a model in the browser cache loads the same way.
    case 'queued': return 'waiting to load';
    case 'loading': return Number.isFinite(entry.progress) ? `loading ${percent(entry.progress)}%` : 'preparing';
    case 'ready': return 'ready';
    case 'failed': return 'could not load, tries again when needed';
    default: return '';
  }
}

function renderModelState(element, entry) {
  const text = modelStateText(entry);
  element.textContent = text ? ` · ${text}` : '';
  element.dataset.state = entry?.state ?? '';
}

function renderModelLines() {
  const model = getAsrModel(asrModelSelect.value);
  asrModelInfo.textContent = `${model.label} · speech-to-text · ${downloadSize(model.bytes)} · runs locally`;
  renderModelState(asrModelState, state.models.asr.get(asrModelSelect.value));
  renderModelState(diarModelState, state.models.diarization);
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
  setStatus(text, { value: determinate ? `${percent(message.progress)}%` : '' });
}

/** Model-line updates from the worker, whichever request started the load. */
function handleModelState(message) {
  const entry = { state: message.state, progress: message.progress };
  if (message.model === 'asr') state.models.asr.set(message.asrModel, entry);
  else if (message.model === 'diarization') state.models.diarization = entry;
  renderModelLines();
}

/** A worker crash stops every load in progress; the next request tries again. */
function handleCrash() {
  const stopped = (entry) => (entry?.state === 'queued' || entry?.state === 'loading' ? { state: 'failed' } : entry);
  for (const [key, entry] of state.models.asr) state.models.asr.set(key, stopped(entry));
  state.models.diarization = stopped(state.models.diarization);
  renderModelLines();
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

// ---- Model preloads ----
// Both models start to load when the page opens, and the selected Whisper
// model loads when the selection changes. Loading a different Whisper model
// frees the previous one from memory, but its files stay in the browser
// cache, so selecting it again does not download it again.

// Arrow keys on a closed select can fire 'change' for each model they pass,
// so a new selection preloads only after it stays selected this long.
const ASR_PRELOAD_DELAY_MS = 1000;

/** A task that waits behind a preload shows the preload's progress. */
function showPreloadProgress(message) {
  if (taskRunning) handleProgress(message);
}

function queueAsrModelState() {
  state.models.asr.set(asrModelSelect.value, { state: 'queued' });
  renderModelLines();
}

function preloadAsr() {
  queueAsrModelState();
  // The worker drops this request if a newer Whisper preload arrives before it starts.
  inference.run({ type: 'preload', model: 'asr', asrModel: asrModelSelect.value }, showPreloadProgress)
    .catch(() => {}); // The model line shows the failure; the next transcription tries again.
}

function preloadDiarization() {
  state.models.diarization = { state: 'queued' };
  renderModelLines();
  inference.run({ type: 'preload', model: 'diarization' }, showPreloadProgress)
    .catch(() => {}); // The model line shows the failure; the next speaker detection tries again.
}

// ---- Transcription and diarization ----
/** Runs a transcribe or diarize request, which waits for a preload that has started. */
async function runTask(request) {
  taskRunning = true;
  try {
    return await inference.run(request, handleProgress);
  } finally {
    taskRunning = false;
  }
}

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
  const asrModelKey = asrModelSelect.value;
  state.phase = 'busy';
  setRetry(null);
  updateControls();
  setProgress('indeterminate');
  setStatus(`Preparing to transcribe ${formatClock(candidate.duration)} of audio…`);

  let result;
  try {
    result = await runTask({ type: 'transcribe', audio: candidate.audio, speakers, asrModel: asrModelKey });
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
    result = await runTask({ type: 'diarize', audio: source.audio, segments: state.workerSegments });
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

asrModelSelect.addEventListener('change', () => {
  queueAsrModelState();
  clearTimeout(asrPreloadTimer);
  asrPreloadTimer = setTimeout(preloadAsr, ASR_PRELOAD_DELAY_MS);
});

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
  // Short text, so it fits a phone-width select; the model line shows the full name.
  asrModelSelect.add(new Option(`${model.level} — ${key} (${downloadSize(model.bytes)})`, key));
}
asrModelSelect.value = DEFAULT_ASR_MODEL;
renderModelLines();

if (typeof Worker === 'undefined' || typeof OfflineAudioContext === 'undefined') {
  render();
  micBtn.disabled = true;
  fileBtn.disabled = true;
  speakersToggle.disabled = true;
  asrModelSelect.disabled = true;
  setStatus('This browser is missing Web Workers or Web Audio, which this demo needs. Use a current version of Chrome, Edge, Firefox, or Safari.', { tone: 'error' });
} else {
  render();
  preloadAsr();
  preloadDiarization();
}
