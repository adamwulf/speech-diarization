import test from 'node:test';
import assert from 'node:assert/strict';
import {
  ASR_MODELS,
  asrVariant,
  BACKEND_VARIANTS,
  DEFAULT_ASR_MODEL,
  downloadSize,
  getAsrModel,
} from '../src/asr-models.js';

test('the default model is Whisper tiny.en', () => {
  assert.equal(DEFAULT_ASR_MODEL, 'tiny.en');
  assert.equal(getAsrModel(), ASR_MODELS['tiny.en']);
  assert.equal(getAsrModel().repo, 'onnx-community/whisper-tiny.en_timestamped');
});

test('the models are tiny.en, base.en, and small.en, from fastest to most accurate', () => {
  assert.deepEqual(Object.keys(ASR_MODELS), ['tiny.en', 'base.en', 'small.en']);
  assert.deepEqual(Object.values(ASR_MODELS).map((model) => model.level), ['Fast', 'Balanced', 'Accurate']);
});

test('each model is an English-only timestamped export pinned to a commit', () => {
  for (const [key, model] of Object.entries(ASR_MODELS)) {
    assert.equal(model.repo, `onnx-community/whisper-${key}_timestamped`);
    assert.match(model.revision, /^[0-9a-f]{40}$/);
    for (const field of ['wasmBytes', 'webgpuBytes']) {
      assert.ok(Number.isInteger(model[field]) && model[field] > 0, `${key} ${field}`);
    }
    // The GPU weights keep an fp32 encoder, so they are always the larger download.
    assert.ok(model.webgpuBytes > model.wasmBytes, key);
  }
});

test('getAsrModel returns the model for a known key', () => {
  assert.equal(getAsrModel('base.en').label, 'Whisper base.en');
  assert.equal(getAsrModel('small.en').label, 'Whisper small.en');
});

test('getAsrModel throws for an unknown key, including inherited property names', () => {
  for (const key of ['medium.en', 'tiny', '', null, 'toString', '__proto__']) {
    assert.throws(() => getAsrModel(key), /Unknown speech recognition model/);
  }
});

test('downloadSize rounds to whole megabytes', () => {
  assert.equal(downloadSize(ASR_MODELS['tiny.en'].wasmBytes), '~44 MB');
  assert.equal(downloadSize(ASR_MODELS['base.en'].wasmBytes), '~80 MB');
  assert.equal(downloadSize(ASR_MODELS['small.en'].wasmBytes), '~252 MB');
});

test('asrVariant gives the q8 weights for WASM and the fp32 encoder for WebGPU', () => {
  assert.deepEqual(asrVariant('tiny.en', 'wasm'), { dtype: 'q8', bytes: ASR_MODELS['tiny.en'].wasmBytes });
  const gpu = asrVariant('tiny.en', 'webgpu');
  assert.equal(gpu.bytes, ASR_MODELS['tiny.en'].webgpuBytes);
  // Per file: a single dtype string would ask for files these repositories lack.
  assert.deepEqual({ ...gpu.dtype }, { encoder_model: 'fp32', decoder_model_merged: 'fp16' });
});

test('asrVariant defaults to the default model and to the WASM weights', () => {
  assert.deepEqual(asrVariant(), asrVariant(DEFAULT_ASR_MODEL, 'wasm'));
  assert.deepEqual(asrVariant('base.en'), asrVariant('base.en', 'wasm'));
});

test('asrVariant treats an unknown backend as WASM, inherited names included', () => {
  for (const backend of ['cuda', '', null, undefined, 'toString', '__proto__', 'constructor']) {
    assert.deepEqual(asrVariant('base.en', backend), asrVariant('base.en', 'wasm'), `for ${String(backend)}`);
  }
});

test('asrVariant throws for an unknown model whatever the backend', () => {
  for (const backend of ['wasm', 'webgpu']) {
    assert.throws(() => asrVariant('medium.en', backend), /Unknown speech recognition model/);
  }
});

test('every backend variant names a byte field that each model has', () => {
  for (const variant of Object.values(BACKEND_VARIANTS)) {
    for (const model of Object.values(ASR_MODELS)) {
      assert.ok(Number.isInteger(model[variant.bytesKey]), variant.bytesKey);
    }
  }
});
