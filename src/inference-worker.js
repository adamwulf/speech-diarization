// Web Worker: downloads the models and runs all speech inference.
// Message contract: see IMPLEMENTATION.md and INFERENCE.md.

import { env, pipeline } from '@huggingface/transformers';
import * as ort from 'onnxruntime-web';
import { DiarizationPipeline } from 'diarization-js';
import ortWasmUrl from '../node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.jsep.wasm?url';
import { DEFAULT_ASR_MODEL, getAsrModel } from './asr-models.js';
import {
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
// The slot key is a key of ASR_MODELS. Disposing a model frees its memory
// only; its files stay in the Transformers.js browser cache.
const asrSlot = createModelSlot({
  load(key, report) {
    const model = getAsrModel(key);
    const message = `Downloading the ${model.label} speech recognition model`;
    const onFraction = (fraction) => {
      report('asr-load', message, fraction);
      postModelState('asr', key, 'loading', fraction);
    };
    onFraction(0);
    // Stay a little below the file total so the bar can reach 100%.
    const expectedBytes = Math.floor(model.bytes * 0.999);
    return pipeline('automatic-speech-recognition', model.repo, {
      revision: model.revision,
      device: 'wasm',
      dtype: 'q8',
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

/** The Whisper model for a key (no key: the default). An unknown key throws before the loaded model is disposed. */
function loadAsr(key = DEFAULT_ASR_MODEL, report) {
  getAsrModel(key);
  return asrSlot(key, report);
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
  return DiarizationPipeline.create({
    ort,
    segmentationModel,
    embeddingModel,
    pldaParamsJson: JSON.parse(new TextDecoder().decode(pldaBytes)),
    executionProviders: ['wasm'],
    // A smaller batch than the WebGPU-tuned default keeps WASM memory use low.
    segmentationBatchSize: 8,
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
      report('asr', 'Transcribing');
      try {
        const output = await asr(audio, { return_timestamps: 'word', chunk_length_s: 30, stride_length_s: 5 });
        return output.chunks ?? [];
      } catch (error) {
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
