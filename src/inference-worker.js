// Web Worker: downloads the models and runs all speech inference.
// Message contract: see IMPLEMENTATION.md and INFERENCE.md.

import { env, pipeline } from '@huggingface/transformers';
import * as ort from 'onnxruntime-web';
import { DiarizationPipeline } from 'diarization-js';
import ortWasmUrl from '../node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.jsep.wasm?url';
import { getAsrModel } from './asr-models.js';
import {
  createByteProgress,
  createModelSlot,
  diarizationFraction,
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

// Only one Whisper model is kept loaded: the one the last transcription used.
// The slot key is the model entry from asr-models.js.
const asrSlot = createModelSlot({
  load(model, report) {
    const message = `Downloading the ${model.label} speech recognition model`;
    report('asr-load', message, 0);
    const onFraction = (fraction) => report('asr-load', message, fraction);
    // Stay a little below the file total so the bar can reach 100%.
    const expectedBytes = Math.floor(model.bytes * 0.999);
    return pipeline('automatic-speech-recognition', model.repo, {
      revision: model.revision,
      device: 'wasm',
      dtype: 'q8',
      progress_callback: transformersProgressHandler(createByteProgress(expectedBytes, onFraction)),
    }).catch((error) => {
      throw new Error(`Could not load the ${model.label} speech recognition model (${errorText(error)}). `
        + 'Check the network connection, then try again.');
    });
  },
  dispose: (asr) => asr.dispose(),
});

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
  report('diarization-load', message, 0);
  const update = createByteProgress(DIARIZATION_EXPECTED_BYTES,
    (fraction) => report('diarization-load', message, fraction));
  const cache = await openArtifactCache();
  const [segmentationModel, embeddingModel, pldaBytes] = await Promise.all(
    Object.values(DIARIZATION_FILES).map((file) =>
      fetchArtifact(cache, file, (loaded, total) => update(file, loaded, total))),
  );
  report('diarization-load', 'Preparing the speaker detection models');
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
  diarizerPromise ??= createDiarizer(report).catch((error) => {
    diarizerPromise = null;
    throw new Error(`Could not load the speaker detection models (${errorText(error)}). `
      + 'Check the network connection, then try again.');
  });
  return diarizerPromise;
}

function createBackend(report, asrModelKey) {
  return {
    async transcribe(audio) {
      const asr = await asrSlot(getAsrModel(asrModelKey), report);
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
    } else {
      throw new Error(`Unknown request type: ${type}`);
    }
    self.postMessage({ type: 'result', id, ...result });
  } catch (error) {
    self.postMessage({ type: 'error', id, message: errorText(error) });
  }
}

// One task at a time: each request waits for the previous one to finish.
let queue = Promise.resolve();
self.addEventListener('message', (event) => {
  queue = queue.then(() => handle(event.data));
});
