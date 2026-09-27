// Task flow for the inference worker. The model calls are injected as a
// `backend` ({transcribe, diarize}) so Node tests can run this without models.

import {
  assignSpeakers,
  dropNonSpeechTags,
  groupWords,
  normalizeTurns,
  normalizeWords,
  relabelSpeakers,
  whisperChunksToWords,
  wordsFromSegments,
} from './alignment.js';

export const SAMPLE_RATE = 16000;
export const MIN_AUDIO_SECONDS = 0.1;

/** Frame RMS below this everywhere (about -54 dBFS) counts as silence. */
export const SILENCE_RMS = 0.002;
const ENERGY_FRAME_SAMPLES = 320; // 20 ms at 16 kHz

export const WARNINGS = Object.freeze({
  tooShort: 'The audio is too short to transcribe. Record or choose at least a few seconds of speech.',
  noSpeech: 'No speech was detected in this audio.',
  noText: 'There is no transcript text to label with speakers.',
  noSpeakers: 'No speaker could be identified. Each speaker needs about one second or more of clear speech.',
  diarizationFailed: (message) => `${message} The transcript is shown without speakers.`,
});

/** Diarization pipeline steps and their share of the total run time. */
const DIARIZATION_STEPS = {
  segmentation: [0, 0.25],
  embedding: [0.25, 0.7],
  clustering: [0.95, 0.03],
  reconstruction: [0.98, 0.02],
};

/** Map a diarization-js `PipelineProgress` to one 0..1 value. */
export function diarizationFraction(progress) {
  const [base, share] = DIARIZATION_STEPS[progress?.step] ?? [0, 0];
  const fraction = Number.isFinite(progress?.fraction) ? Math.min(Math.max(progress.fraction, 0), 1) : 0;
  return base + share * fraction;
}

function assertAudio(audio) {
  if (!(audio instanceof Float32Array)) {
    throw new Error('The audio must be 16 kHz mono PCM in a Float32Array.');
  }
}

/** True if any 20 ms frame of the audio is louder than `SILENCE_RMS`. */
export function hasSpeechEnergy(audio, threshold = SILENCE_RMS) {
  const limit = threshold * threshold * ENERGY_FRAME_SAMPLES;
  for (let start = 0; start < audio.length; start += ENERGY_FRAME_SAMPLES) {
    const end = Math.min(start + ENERGY_FRAME_SAMPLES, audio.length);
    let sum = 0;
    for (let i = start; i < end; i++) sum += audio[i] * audio[i];
    // Compare the frame's mean square without a sqrt; scale for a short last frame.
    if (sum * (ENERGY_FRAME_SAMPLES / (end - start)) >= limit) return true;
  }
  return false;
}

/**
 * Report download progress for several files as one 0..1 value, weighted by
 * bytes. The denominator is at least `expectedBytes`, so small files that
 * finish first do not look like most of the download. Values never decrease,
 * and changes smaller than `step` are not reported (1 is always reported).
 */
export function createByteProgress(expectedBytes, onFraction, step = 0.01) {
  const files = new Map();
  let reported = 0;
  return (file, loaded, total) => {
    files.set(file, { loaded, total: Math.max(total || 0, loaded) });
    let sumLoaded = 0;
    let sumTotal = 0;
    for (const entry of files.values()) {
      sumLoaded += entry.loaded;
      sumTotal += entry.total;
    }
    const fraction = Math.min(1, sumLoaded / Math.max(sumTotal, expectedBytes, 1));
    if (fraction - reported >= step || (fraction === 1 && reported < 1)) {
      reported = fraction;
      onFraction(fraction);
    }
  };
}

/** Adapt a Transformers.js `progress_callback` event to `createByteProgress`. */
export function transformersProgressHandler(update) {
  return (event) => {
    if (event?.status === 'progress' && event.file
      && Number.isFinite(event.loaded) && Number.isFinite(event.total)) {
      update(event.file, event.loaded, event.total);
    }
  };
}

/** Read a fetch Response body to bytes, calling `onBytes(loaded, total)` per chunk. */
export async function readResponseBytes(response, onBytes) {
  const total = Number.parseInt(response.headers.get('content-length') ?? '', 10) || 0;
  if (!response.body) {
    const bytes = new Uint8Array(await response.arrayBuffer());
    onBytes(bytes.byteLength, bytes.byteLength);
    return bytes;
  }
  const reader = response.body.getReader();
  const chunks = [];
  let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.byteLength;
    onBytes(loaded, total);
  }
  const bytes = new Uint8Array(loaded);
  let offset = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, offset);
    offset += chunk.byteLength;
  }
  onBytes(loaded, loaded);
  return bytes;
}

function withoutSpeakers(words) {
  return words.map((word) => ({ ...word, speaker: null }));
}

function labelWords(words, turns, duration) {
  const labeled = relabelSpeakers(assignSpeakers(words, normalizeTurns(turns, duration)));
  const result = { segments: groupWords(labeled), diarized: true };
  if (labeled.every((word) => word.speaker === null)) result.warning = WARNINGS.noSpeakers;
  return result;
}

/**
 * Handle a `transcribe` request. ASR errors are thrown. A diarization error
 * does not throw: the transcript comes back with `diarized: false` and a warning.
 * Backend errors should have messages that are full sentences the user can act on.
 * @param {{audio: Float32Array, speakers: boolean}} request
 * @param {{transcribe: (audio: Float32Array) => Promise<Array<{text: string, timestamp: number[]}>>,
 *          diarize: (audio: Float32Array) => Promise<Array<{start: number, end: number, speaker: string}>>}} backend
 */
export async function runTranscribe(request, backend) {
  const { audio } = request;
  assertAudio(audio);
  const duration = audio.length / SAMPLE_RATE;
  if (duration < MIN_AUDIO_SECONDS) return { segments: [], diarized: false, warning: WARNINGS.tooShort };
  if (!hasSpeechEnergy(audio)) return { segments: [], diarized: false, warning: WARNINGS.noSpeech };

  const chunks = await backend.transcribe(audio);
  const words = dropNonSpeechTags(normalizeWords(whisperChunksToWords(chunks), duration));
  if (words.length === 0) return { segments: [], diarized: false, warning: WARNINGS.noSpeech };
  if (!request.speakers) return { segments: groupWords(withoutSpeakers(words)), diarized: false };

  let turns;
  try {
    turns = await backend.diarize(audio);
  } catch (error) {
    return {
      segments: groupWords(withoutSpeakers(words)),
      diarized: false,
      warning: WARNINGS.diarizationFailed(error?.message ?? String(error)),
    };
  }
  return labelWords(words, turns, duration);
}

/**
 * Handle a `diarize` request for an existing transcript. Errors are thrown so
 * the caller can report them; the caller keeps its own transcript.
 * @param {{audio: Float32Array, segments: Array<object>}} request
 * @param {{diarize: (audio: Float32Array) => Promise<Array<{start: number, end: number, speaker: string}>>}} backend
 */
export async function runDiarize(request, backend) {
  const { audio } = request;
  assertAudio(audio);
  const duration = audio.length / SAMPLE_RATE;
  const words = normalizeWords(wordsFromSegments(request.segments), duration);
  if (words.length === 0) return { segments: [], diarized: false, warning: WARNINGS.noText };
  const turns = await backend.diarize(audio);
  return labelWords(words, turns, duration);
}
