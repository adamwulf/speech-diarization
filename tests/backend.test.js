import test from 'node:test';
import assert from 'node:assert/strict';
import {
  BACKENDS,
  detectBackend,
  diarizationExecutionProviders,
  FALLBACK_BACKEND,
  parseBackendOverride,
  segmentationBatchSize,
} from '../src/backend.js';

test('the backends are webgpu and wasm, and wasm is the fallback', () => {
  assert.deepEqual([...BACKENDS], ['webgpu', 'wasm']);
  assert.equal(FALLBACK_BACKEND, 'wasm');
  assert.ok(BACKENDS.includes(FALLBACK_BACKEND));
});

test('parseBackendOverride reads a known backend from a search string', () => {
  assert.equal(parseBackendOverride('?backend=webgpu'), 'webgpu');
  assert.equal(parseBackendOverride('?backend=wasm'), 'wasm');
  assert.equal(parseBackendOverride('backend=wasm&other=1'), 'wasm');
});

test('parseBackendOverride ignores a missing, empty, or unknown value', () => {
  for (const search of ['', '?', undefined, null, '?backend=', '?backend=cuda', '?backend=WASM', '?other=wasm']) {
    assert.equal(parseBackendOverride(search), null, `for ${JSON.stringify(search)}`);
  }
});

test('detectBackend uses the override without asking for an adapter', async () => {
  let asked = false;
  const requestAdapter = () => { asked = true; return Promise.resolve({}); };
  const choice = await detectBackend({ override: 'wasm', requestAdapter });
  assert.equal(choice.backend, 'wasm');
  assert.equal(asked, false);
  assert.match(choice.reason, /forced/);
});

test('detectBackend chooses webgpu when an adapter comes back', async () => {
  const choice = await detectBackend({ requestAdapter: () => Promise.resolve({ features: new Set() }) });
  assert.equal(choice.backend, 'webgpu');
});

test('detectBackend falls back to wasm without WebGPU, without an adapter, or on a rejection', async () => {
  const noWebGpu = await detectBackend({});
  assert.equal(noWebGpu.backend, 'wasm');
  assert.match(noWebGpu.reason, /no WebGPU/);

  const noAdapter = await detectBackend({ requestAdapter: () => Promise.resolve(null) });
  assert.equal(noAdapter.backend, 'wasm');
  assert.match(noAdapter.reason, /no GPU adapter/);

  const failed = await detectBackend({ requestAdapter: () => Promise.reject(new Error('driver said no')) });
  assert.equal(failed.backend, 'wasm');
  assert.match(failed.reason, /driver said no/);
});

test('a WebGPU diarization session keeps wasm behind the GPU', () => {
  assert.deepEqual(diarizationExecutionProviders('webgpu'), ['webgpu', 'wasm']);
  assert.deepEqual(diarizationExecutionProviders('wasm'), ['wasm']);
});

test('the segmentation batch is larger on the GPU than on WASM', () => {
  assert.equal(segmentationBatchSize('webgpu'), 32);
  assert.equal(segmentationBatchSize('wasm'), 8);
  assert.ok(segmentationBatchSize('webgpu') > segmentationBatchSize('wasm'));
});
