// The Whisper models the user can choose, shared by the page and the worker.
// Each one is an English-only `_timestamped` export (word timestamps need its
// cross-attention outputs), pinned to a commit.
//
// Each model has the download size of both weight sets the app uses, because
// the two execution providers want different files (see BACKEND_VARIANTS). A
// size is the encoder file + the merged decoder file + the ~2.7 MB of config
// and tokenizer files that Transformers.js loads with them.

export const ASR_MODELS = Object.freeze({
  'tiny.en': Object.freeze({
    label: 'Whisper tiny.en',
    level: 'Fast',
    repo: 'onnx-community/whisper-tiny.en_timestamped',
    revision: 'aeaa13760958b03fac5062f457d317d3319c3168',
    wasmBytes: 43_519_516,
    webgpuBytes: 95_160_623,
  }),
  'base.en': Object.freeze({
    label: 'Whisper base.en',
    level: 'Balanced',
    repo: 'onnx-community/whisper-base.en_timestamped',
    revision: 'fa239a41836c3305f6beec180e5940f3823ff5b8',
    wasmBytes: 79_563_802,
    webgpuBytes: 189_845_128,
  }),
  'small.en': Object.freeze({
    label: 'Whisper small.en',
    level: 'Accurate',
    repo: 'onnx-community/whisper-small.en_timestamped',
    revision: '8085393831131554ad978a4e438295a734d3d2c0',
    wasmBytes: 251_728_328,
    webgpuBytes: 664_021_016,
  }),
});

export const DEFAULT_ASR_MODEL = 'tiny.en';

// The weights each execution provider loads.
//
// WASM gets q8. The WebGPU provider cannot use q8: the integer matrix operators
// it needs are not WebGPU operators, so every one of them copies tensors back
// to the CPU. Measured on an Intel Xe iGPU, q8 tiny.en on WebGPU took 18x
// longer than on WASM and returned nonsense text.
//
// WebGPU gets an fp32 encoder and an fp16 merged decoder. The encoder's
// precision is what matters: in the same browser checks every fp16 encoder
// degenerated into repeating one phrase for the whole clip, while every fp32
// encoder was correct. The fp16 decoder was both correct and the closest match
// to the WASM transcript. INFERENCE.md has the table.
export const BACKEND_VARIANTS = Object.freeze({
  wasm: Object.freeze({ dtype: 'q8', bytesKey: 'wasmBytes' }),
  webgpu: Object.freeze({
    // Per file, not one string: a plain `dtype: 'fp16'` would also ask for an
    // fp16 decoder, and a plain `'q4f16'` asks for `encoder_model_q4f16.onnx`,
    // which these repositories do not have.
    dtype: Object.freeze({ encoder_model: 'fp32', decoder_model_merged: 'fp16' }),
    bytesKey: 'webgpuBytes',
  }),
});

/** The model for a key. No key gives the default model; an unknown key throws. */
export function getAsrModel(key = DEFAULT_ASR_MODEL) {
  if (!Object.hasOwn(ASR_MODELS, key)) throw new Error(`Unknown speech recognition model: ${key}.`);
  return ASR_MODELS[key];
}

/**
 * The weights one backend loads for one model: `{dtype, bytes}`. `dtype` goes
 * to Transformers.js, and `bytes` is the download, for the size on the page and
 * for the progress bar. An unknown backend is treated as WASM, so a backend the
 * table does not cover can never ask for a size that does not exist.
 */
export function asrVariant(key = DEFAULT_ASR_MODEL, backend = 'wasm') {
  const model = getAsrModel(key);
  const variant = Object.hasOwn(BACKEND_VARIANTS, backend) ? BACKEND_VARIANTS[backend] : BACKEND_VARIANTS.wasm;
  return { dtype: variant.dtype, bytes: model[variant.bytesKey] };
}

/** Approximate download size for display, for example "~44 MB". */
export function downloadSize(bytes) {
  return `~${Math.round(bytes / 1e6)} MB`;
}
