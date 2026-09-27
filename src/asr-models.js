// The Whisper models the user can choose, shared by the page and the worker.
// Each one is an English-only `_timestamped` export (word timestamps need its
// cross-attention outputs), loaded as q8 and pinned to a commit.
// `bytes` is the q8 download: encoder_model_quantized + decoder_model_merged_quantized
// + the config and tokenizer files that Transformers.js loads.

export const ASR_MODELS = Object.freeze({
  'tiny.en': Object.freeze({
    label: 'Whisper tiny.en',
    level: 'Fast',
    repo: 'onnx-community/whisper-tiny.en_timestamped',
    revision: 'aeaa13760958b03fac5062f457d317d3319c3168',
    bytes: 43_519_516,
  }),
  'base.en': Object.freeze({
    label: 'Whisper base.en',
    level: 'Balanced',
    repo: 'onnx-community/whisper-base.en_timestamped',
    revision: 'fa239a41836c3305f6beec180e5940f3823ff5b8',
    bytes: 79_563_802,
  }),
  'small.en': Object.freeze({
    label: 'Whisper small.en',
    level: 'Accurate',
    repo: 'onnx-community/whisper-small.en_timestamped',
    revision: '8085393831131554ad978a4e438295a734d3d2c0',
    bytes: 251_728_328,
  }),
});

export const DEFAULT_ASR_MODEL = 'tiny.en';

/** The model for a key. No key gives the default model; an unknown key throws. */
export function getAsrModel(key = DEFAULT_ASR_MODEL) {
  if (!Object.hasOwn(ASR_MODELS, key)) throw new Error(`Unknown speech recognition model: ${key}.`);
  return ASR_MODELS[key];
}

/** Approximate download size for display, for example "~44 MB". */
export function downloadSize(bytes) {
  return `~${Math.round(bytes / 1e6)} MB`;
}
