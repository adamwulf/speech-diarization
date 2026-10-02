// Web Worker: downloads the models and runs all speech inference.
// Message contract: see IMPLEMENTATION.md and INFERENCE.md.

import { env, pipeline } from '@huggingface/transformers';
import * as ort from 'onnxruntime-web';
import { DiarizationPipeline } from 'diarization-js';
import ortWasmUrl from '../node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.jsep.wasm?url';
import { asrVariant, DEFAULT_ASR_MODEL, getAsrModel } from './asr-models.js';
import {
  detectBackend,
  diarizationExecutionProviders,
  parseBackendOverride,
  segmentationBatchSize,
} from './backend.js';
import {
  ASR_CHUNK_OPTIONS,
  asrWindowCount,
  createByteProgress,
  createModelSlot,
  diarizationFraction,
  enqueueRequest,
  readResponseBytes,
  runDiarize,
  runTranscribe,
  SAMPLE_RATE,
  transformersProgressHandler,
} from './inference-tasks.js';

// Transformers.js and diarization-js share this one onnxruntime-web instance.
// Transformers.js points ORT at WASM files on a CDN that come from a different
// ORT build. Use the local WASM that matches the installed onnxruntime-web.
ort.env.wasm.wasmPaths = { wasm: ortWasmUrl };
env.allowLocalModels = false;

// Settled once, before any model loads. The same local WASM file carries the
// WebGPU provider: it is the `jsep` build, which is why the GPU needs no extra
// runtime file.
const backendPromise = detectBackend({
  override: parseBackendOverride(self.location?.search),
  requestAdapter: navigator.gpu ? () => navigator.gpu.requestAdapter() : null,
}).then((choice) => {
  for (const model of ['asr', 'diarization']) postBackend(model, choice.backend, choice.reason);
  return choice.backend;
});

// The provider Whisper loads on. It is the detected one until a WebGPU load or
// a WebGPU run fails, and then it is WASM for the life of the worker: a GPU
// that cannot run this model will not start working later, and WASM always can.
let asrBackend = null;

async function currentAsrBackend() {
  asrBackend ??= await backendPromise;
  return asrBackend;
}

const DIARIZATION_BASE_URL =
  'https://huggingface.co/briox/diarization-js-community-1/resolve/7f02c43a14eff6d4dea0a4f8099f1ca48eb163b8/';
const DIARIZATION_FILES = {
  segmentation: 'segmentation-3.0.onnx',
  embedding: 'embedding-resnet34.onnx',
  plda: 'plda-params-vbx.json',
};
const DIARIZATION_EXPECTED_BYTES = 33_494_108;
const DIARIZATION_CACHE = 'diarization-js-community-1';

let diarizerPromise = null;

function errorText(error) {
  return error?.message ?? String(error);
}

/**
 * Tell the page which execution provider one pipeline runs on. Sent once per
 * pipeline after detection, and again for `asr` if it falls back to the CPU.
 */
function postBackend(model, backend, reason) {
  self.postMessage({ type: 'backend', model, backend, reason });
}

/**
 * Tell the page the state of a model load, whatever request started it:
 * `{type: 'model', model: 'asr'|'diarization', asrModel?, state: 'loading'|'ready'|'failed', progress?}`.
 */
function postModelState(model, asrModel, state, progress) {
  const event = { type: 'model', model, state };
  if (asrModel) event.asrModel = asrModel;
  if (Number.isFinite(progress)) event.progress = progress;
  self.postMessage(event);
}

// Only one Whisper model is kept loaded: the one the last request used.
// The slot key is `<key of ASR_MODELS>|<backend>`, because the two providers
// load different weights, so a change of provider reloads the model the same
// way a change of model does. Disposing a model frees its memory only; its
// files stay in the Transformers.js browser cache.
const asrSlot = createModelSlot({
  load(slotKey, report) {
    const [key, backend] = slotKey.split('|');
    const model = getAsrModel(key);
    const { dtype, bytes } = asrVariant(key, backend);
    const message = `Downloading the ${model.label} speech recognition model`;
    const onFraction = (fraction) => {
      report('asr-load', message, fraction);
      postModelState('asr', key, 'loading', fraction);
    };
    onFraction(0);
    // Stay a little below the file total so the bar can reach 100%.
    const expectedBytes = Math.floor(bytes * 0.999);
    return pipeline('automatic-speech-recognition', model.repo, {
      revision: model.revision,
      device: backend,
      dtype,
      progress_callback: transformersProgressHandler(createByteProgress(expectedBytes, onFraction)),
    }).then((asr) => {
      postModelState('asr', key, 'ready');
      return asr;
    }, (error) => {
      postModelState('asr', key, 'failed');
      throw new Error(`Could not load the ${model.label} speech recognition model (${errorText(error)}). `
        + 'Check the network connection, then try again.');
    });
  },
  dispose: (asr) => asr.dispose(),
});

/**
 * The Whisper model for a key (no key: the default). An unknown key throws
 * before the loaded model is disposed. A WebGPU load that fails is not the
 * user's problem to solve: the worker drops to the CPU weights and loads those
 * instead, and only a second failure is reported.
 */
async function loadAsr(key = DEFAULT_ASR_MODEL, report) {
  getAsrModel(key);
  const backend = await currentAsrBackend();
  try {
    return await asrSlot(`${key}|${backend}`, report);
  } catch (error) {
    if (backend !== 'webgpu') throw error;
    demoteAsrToWasm('the speech recognition model could not load on the GPU');
    return asrSlot(`${key}|${asrBackend}`, report);
  }
}

/**
 * Move Whisper to the CPU for good. The next `loadAsr` asks the slot for a
 * different key, so the slot disposes the GPU model before loading the CPU one.
 */
function demoteAsrToWasm(reason) {
  if (asrBackend === 'wasm') return;
  asrBackend = 'wasm';
  postBackend('asr', 'wasm', reason);
}

async function openArtifactCache() {
  try {
    return typeof caches === 'undefined' ? null : await caches.open(DIARIZATION_CACHE);
  } catch {
    return null; // The Cache API can be unavailable (for example, private browsing).
  }
}

async function fetchArtifact(cache, file, onBytes) {
  const url = DIARIZATION_BASE_URL + file;
  const cached = await cache?.match(url);
  if (cached) return readResponseBytes(cached, onBytes);
  const response = await fetch(url);
  if (!response.ok) throw new Error(`HTTP ${response.status} for ${file}`);
  const bytes = await readResponseBytes(response, onBytes);
  try {
    await cache?.put(url, new Response(bytes, { headers: { 'content-length': String(bytes.byteLength) } }));
  } catch {
    // Not cached (for example, storage quota); the download still worked.
  }
  return bytes;
}

async function createDiarizer(report) {
  const message = 'Downloading the speaker detection models';
  const onFraction = (fraction) => {
    report('diarization-load', message, fraction);
    postModelState('diarization', null, 'loading', fraction);
  };
  onFraction(0);
  const update = createByteProgress(DIARIZATION_EXPECTED_BYTES, onFraction);
  const cache = await openArtifactCache();
  const [segmentationModel, embeddingModel, pldaBytes] = await Promise.all(
    Object.values(DIARIZATION_FILES).map((file) =>
      fetchArtifact(cache, file, (loaded, total) => update(file, loaded, total))),
  );
  report('diarization-load', 'Preparing the speaker detection models');
  postModelState('diarization', null, 'loading');
  // The same model files run on either provider, so the backend changes no
  // download.
  const backend = await backendPromise;
  return DiarizationPipeline.create({
    ort,
    segmentationModel,
    embeddingModel,
    pldaParamsJson: JSON.parse(new TextDecoder().decode(pldaBytes)),
    executionProviders: diarizationExecutionProviders(backend),
    segmentationBatchSize: segmentationBatchSize(backend),
    // diarization-js zero-pads each embedding batch to its longest crop, and
    // the padding changes the embeddings. In browser tests on a two-speaker
    // sample, batch 8 merged both speakers into one; batch 1 (no padding) gave
    // the best match to the reference and was also the fastest on WASM.
    embeddingBatchSize: 1,
  });
}

function loadDiarizer(report) {
  diarizerPromise ??= createDiarizer(report).then((diarizer) => {
    postModelState('diarization', null, 'ready');
    return diarizer;
  }, (error) => {
    diarizerPromise = null;
    postModelState('diarization', null, 'failed');
    throw new Error(`Could not load the speaker detection models (${errorText(error)}). `
      + 'Check the network connection, then try again.');
  });
  return diarizerPromise;
}

/**
 * Handle a `preload` request: load one model now, so a later task does not wait
 * for it. `ready` is sent here too, because a model that is already loaded sends
 * no model messages of its own.
 */
async function preload(message, report) {
  if (message.model === 'asr') {
    const key = message.asrModel ?? DEFAULT_ASR_MODEL;
    await loadAsr(key, report);
    postModelState('asr', key, 'ready');
  } else if (message.model === 'diarization') {
    await loadDiarizer(report);
    postModelState('diarization', null, 'ready');
  } else {
    throw new Error(`Unknown model to preload: ${message.model}`);
  }
  return {};
}

function createBackend(report, asrModelKey) {
  return {
    async transcribe(audio) {
      const asr = await loadAsr(asrModelKey, report);
      // Transformers.js first computes the audio features of every window (no
      // progress), then runs generate() once per window. generate() runs the
      // window's encoder, calls the streamer's put() with the prompt tokens and
      // then with each new token, and calls end() when the window is done.
      report('asr', 'Preparing transcription');
      const windows = asrWindowCount(audio.length);
      let started = false;
      let finished = 0;
      const streamer = {
        put() {
          if (started) return;
          started = true;
          report('asr', 'Transcribing', 0);
        },
        end() {
          finished += 1;
          report('asr', 'Transcribing', Math.min(1, finished / windows));
        },
      };
      try {
        const output = await asr(audio, { return_timestamps: 'word', ...ASR_CHUNK_OPTIONS, streamer });
        return output.chunks ?? [];
      } catch (error) {
        // A GPU that fails mid-run will fail the same way again, so the next
        // attempt uses the CPU weights. The retry is the user's, so one failed
        // request never silently starts a second model download.
        if (asrBackend === 'webgpu') demoteAsrToWasm('the GPU could not run the speech recognition model');
        throw new Error(`Transcription failed (${errorText(error)}). Try again.`);
      }
    },
    async diarize(audio) {
      const diarizer = await loadDiarizer(report);
      report('diarization', 'Detecting speakers', 0);
      try {
        const { result } = await diarizer.run(audio, SAMPLE_RATE, {
          onProgress: (progress) => report('diarization', 'Detecting speakers', diarizationFraction(progress)),
        });
        return result.segments;
      } catch (error) {
        throw new Error(`Speaker detection failed (${errorText(error)}). Try again.`);
      }
    },
  };
}

async function handle(message) {
  const { type, id } = message ?? {};
  const report = (stage, text, progress) => {
    const event = { type: 'progress', id, stage, message: text };
    if (Number.isFinite(progress)) event.progress = progress;
    self.postMessage(event);
  };
  try {
    let result;
    const backend = createBackend(report, message?.asrModel);
    if (type === 'transcribe') {
      result = await runTranscribe(message, backend);
    } else if (type === 'diarize') {
      result = await runDiarize(message, backend);
    } else if (type === 'preload') {
      result = await preload(message, report);
    } else {
      throw new Error(`Unknown request type: ${type}`);
    }
    self.postMessage({ type: 'result', id, ...result });
  } catch (error) {
    self.postMessage({ type: 'error', id, message: errorText(error) });
  }
}

// One request at a time, preloads included, so the Whisper slot never switches
// models mid-task. `enqueueRequest` sets the order of the waiting requests.
const waiting = [];
let running = false;

async function drain() {
  running = true;
  try {
    while (waiting.length) await handle(waiting.shift());
  } finally {
    running = false;
  }
}

self.addEventListener('message', (event) => {
  const replaced = enqueueRequest(waiting, event.data);
  // A newer preload of the same model took its place before it started.
  if (replaced) self.postMessage({ type: 'result', id: replaced.id });
  if (!running) drain();
});
