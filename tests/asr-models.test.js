import test from 'node:test';
import assert from 'node:assert/strict';
import { ASR_MODELS, DEFAULT_ASR_MODEL, downloadSize, getAsrModel } from '../src/asr-models.js';

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
    assert.ok(Number.isInteger(model.bytes) && model.bytes > 0);
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
  assert.equal(downloadSize(ASR_MODELS['tiny.en'].bytes), '~44 MB');
  assert.equal(downloadSize(ASR_MODELS['base.en'].bytes), '~80 MB');
  assert.equal(downloadSize(ASR_MODELS['small.en'].bytes), '~252 MB');
});
