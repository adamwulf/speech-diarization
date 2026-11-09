// Which ONNX execution provider runs the models in the worker.
//
// WebGPU puts the matrix work on the GPU, which on a typical laptop is the
// integrated one. WASM is the fallback: it always works, but without
// cross-origin isolation it gets one thread, so it is much slower. The choice
// is made once per worker, before any model loads.
//
// Both pipelines run on WebGPU when there is a GPU. The speaker models load the
// same files either way; Whisper does not, so `BACKEND_VARIANTS` in
// `asr-models.js` picks its weights per provider, and Whisper alone falls back
// to WASM if the GPU cannot load or run it. INFERENCE.md has the measurements.

/** The backends the worker knows, fastest first. */
export const BACKENDS = Object.freeze(['webgpu', 'wasm']);

export const FALLBACK_BACKEND = 'wasm';

/**
 * A `?backend=` override from the worker's own URL. It is how the browser
 * checks compare the two providers on one build, and it forces WASM on a
 * machine whose GPU driver misbehaves. An absent or unknown value gives null,
 * which means "detect".
 */
export function parseBackendOverride(search) {
  const value = new URLSearchParams(search ?? '').get('backend');
  return BACKENDS.includes(value) ? value : null;
}

/**
 * The backend to use: the override when there is one, otherwise `webgpu` when
 * the environment hands out a GPU adapter, otherwise `wasm`.
 *
 * `requestAdapter` calls `navigator.gpu.requestAdapter()`, or is null when the
 * browser has no WebGPU. It can also resolve to null, or reject, when WebGPU
 * exists but no adapter is usable; all three cases mean WASM. This has to be
 * settled before any model loads, because Transformers.js throws
 * `Unsupported device` for a WebGPU request in a browser without WebGPU.
 */
export async function detectBackend({ override = null, requestAdapter = null } = {}) {
  if (override) return { backend: override, reason: 'forced by the backend setting' };
  if (!requestAdapter) return { backend: FALLBACK_BACKEND, reason: 'this browser has no WebGPU' };
  try {
    const adapter = await requestAdapter();
    if (!adapter) return { backend: FALLBACK_BACKEND, reason: 'no GPU adapter is available' };
    return { backend: 'webgpu', reason: 'a GPU adapter is available' };
  } catch (error) {
    return { backend: FALLBACK_BACKEND, reason: `the GPU adapter request failed (${error?.message ?? error})` };
  }
}

/**
 * The execution providers for the diarization-js sessions. A WebGPU session
 * keeps `wasm` behind the GPU, so an operator that the WebGPU provider does
 * not implement runs on the CPU instead of failing the whole session.
 */
export function diarizationExecutionProviders(backend) {
  return backend === 'webgpu' ? ['webgpu', 'wasm'] : ['wasm'];
}

/**
 * Batch size for the 10 s segmentation windows. The GPU needs a large batch to
 * stay busy, and diarization-js defaults to 32; on WASM a batch that large only
 * grows the heap, so it stays small there.
 * (`embeddingBatchSize` is 1 on both backends: see the note at its use.)
 */
export function segmentationBatchSize(backend) {
  return backend === 'webgpu' ? 32 : 8;
}
