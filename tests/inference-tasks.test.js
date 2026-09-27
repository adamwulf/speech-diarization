import test from 'node:test';
import assert from 'node:assert/strict';
import {
  ASR_CHUNK_OPTIONS,
  asrWindowCount,
  createByteProgress,
  createModelSlot,
  diarizationFraction,
  enqueueRequest,
  hasSpeechEnergy,
  readResponseBytes,
  runDiarize,
  runTranscribe,
  SAMPLE_RATE,
  SILENCE_RMS,
  transformersProgressHandler,
  WARNINGS,
} from '../src/inference-tasks.js';

function tone(seconds, amplitude = 0.1) {
  const audio = new Float32Array(Math.round(seconds * SAMPLE_RATE));
  for (let i = 0; i < audio.length; i++) audio[i] = amplitude * Math.sin((2 * Math.PI * 220 * i) / SAMPLE_RATE);
  return audio;
}

const chunk = (text, start, end) => ({ text, timestamp: [start, end] });

const ASR_CHUNKS = [
  chunk(' Hello', 0.2, 0.6), chunk(' there.', 0.6, 1.1),
  chunk(' Hi', 1.6, 1.9), chunk(' back.', 1.9, 2.4),
  chunk(' Bye.', 2.6, 12), // Whisper end past the audio end
];
const TURNS = [
  { start: 0.1, end: 1.2, speaker: 'SPEAKER_01' },
  { start: 1.5, end: 2.5, speaker: 'SPEAKER_00' },
  { start: 2.55, end: 3, speaker: 'SPEAKER_01' },
];

function fakeBackend({ chunks = ASR_CHUNKS, turns = TURNS, asrError, diarizeError } = {}) {
  const calls = { transcribe: 0, diarize: 0 };
  return {
    calls,
    async transcribe(audio) {
      calls.transcribe += 1;
      assert.ok(audio instanceof Float32Array);
      if (asrError) throw asrError;
      return chunks;
    },
    async diarize(audio) {
      calls.diarize += 1;
      assert.ok(audio instanceof Float32Array);
      if (diarizeError) throw diarizeError;
      return turns;
    },
  };
}

function assertContractSegments(segments, duration) {
  for (const s of segments) {
    assert.ok(Number.isFinite(s.start) && Number.isFinite(s.end) && s.start >= 0 && s.end > s.start && s.end <= duration);
    assert.equal(typeof s.text, 'string');
    assert.ok(s.speaker === null || typeof s.speaker === 'string');
    assert.equal(s.text, s.words.map((w) => w.text).join('').trim());
  }
}

test('transcribe with speakers aligns words to stable speaker IDs', async () => {
  const backend = fakeBackend();
  const result = await runTranscribe({ audio: tone(3), speakers: true }, backend);
  assert.deepEqual(backend.calls, { transcribe: 1, diarize: 1 });
  assert.equal(result.diarized, true);
  assert.equal(result.warning, undefined);
  assert.deepEqual(result.segments.map((s) => [s.speaker, s.text]), [
    ['SPEAKER_1', 'Hello there.'],
    ['SPEAKER_2', 'Hi back.'],
    ['SPEAKER_1', 'Bye.'],
  ]);
  assert.equal(result.segments.at(-1).end, 3);
  assertContractSegments(result.segments, 3);
});

test('transcribe without speakers skips diarization and leaves speakers null', async () => {
  const backend = fakeBackend();
  const result = await runTranscribe({ audio: tone(3), speakers: false }, backend);
  assert.deepEqual(backend.calls, { transcribe: 1, diarize: 0 });
  assert.equal(result.diarized, false);
  assert.equal(result.warning, undefined);
  assert.ok(result.segments.every((s) => s.speaker === null));
  assert.equal(result.segments.map((s) => s.text).join(' '), 'Hello there. Hi back. Bye.');
});

test('a diarization failure keeps the transcript and returns a warning', async () => {
  const backend = fakeBackend({ diarizeError: new Error('Could not load the speaker detection models (HTTP 503).') });
  const result = await runTranscribe({ audio: tone(3), speakers: true }, backend);
  assert.equal(result.diarized, false);
  assert.match(result.warning, /HTTP 503/);
  assert.match(result.warning, /without speakers/);
  assert.equal(result.segments.map((s) => s.text).join(' '), 'Hello there. Hi back. Bye.');
  assert.ok(result.segments.every((s) => s.speaker === null));
});

test('an ASR failure rejects so the worker can report an error', async () => {
  const backend = fakeBackend({ asrError: new Error('Transcription failed (out of memory). Try again.') });
  await assert.rejects(runTranscribe({ audio: tone(3), speakers: true }, backend), /out of memory/);
  assert.equal(backend.calls.diarize, 0);
});

test('silent and too-short audio return an empty result without running models', async () => {
  const backend = fakeBackend();
  assert.deepEqual(
    await runTranscribe({ audio: new Float32Array(SAMPLE_RATE * 2), speakers: true }, backend),
    { segments: [], diarized: false, warning: WARNINGS.noSpeech },
  );
  assert.deepEqual(
    await runTranscribe({ audio: tone(0.05), speakers: true }, backend),
    { segments: [], diarized: false, warning: WARNINGS.tooShort },
  );
  assert.deepEqual(backend.calls, { transcribe: 0, diarize: 0 });
});

test('Whisper non-speech output is treated as no speech', async () => {
  const backend = fakeBackend({ chunks: [chunk(' [BLANK', 0, 1), chunk('_AUDIO]', 1, 2)] });
  const result = await runTranscribe({ audio: tone(3), speakers: true }, backend);
  assert.deepEqual(result, { segments: [], diarized: false, warning: WARNINGS.noSpeech });
  assert.equal(backend.calls.diarize, 0);
});

test('no detected speakers gives null speakers and a warning', async () => {
  const result = await runTranscribe({ audio: tone(3), speakers: true }, fakeBackend({ turns: [] }));
  assert.equal(result.diarized, true);
  assert.equal(result.warning, WARNINGS.noSpeakers);
  assert.ok(result.segments.length > 0 && result.segments.every((s) => s.speaker === null));
});

test('requests without Float32Array audio are rejected', async () => {
  await assert.rejects(runTranscribe({ audio: [0, 0.1], speakers: false }, fakeBackend()), /Float32Array/);
  await assert.rejects(runDiarize({ audio: null, segments: [] }, fakeBackend()), /Float32Array/);
});

test('diarize realigns an existing transcript at word boundaries', async () => {
  const plain = await runTranscribe({ audio: tone(3), speakers: false }, fakeBackend());
  const backend = fakeBackend();
  const result = await runDiarize({ audio: tone(3), segments: plain.segments }, backend);
  assert.deepEqual(backend.calls, { transcribe: 0, diarize: 1 });
  const direct = await runTranscribe({ audio: tone(3), speakers: true }, fakeBackend());
  assert.deepEqual(result, direct);
});

test('diarize rejects on failure and reports an empty transcript', async () => {
  const plain = await runTranscribe({ audio: tone(3), speakers: false }, fakeBackend());
  await assert.rejects(
    runDiarize({ audio: tone(3), segments: plain.segments }, fakeBackend({ diarizeError: new Error('boom') })),
    /boom/,
  );
  const backend = fakeBackend();
  assert.deepEqual(
    await runDiarize({ audio: tone(3), segments: [] }, backend),
    { segments: [], diarized: false, warning: WARNINGS.noText },
  );
  assert.equal(backend.calls.diarize, 0);
});

test('hasSpeechEnergy finds sound above the silence level, including a short last frame', () => {
  assert.equal(hasSpeechEnergy(new Float32Array(16000)), false);
  assert.equal(hasSpeechEnergy(tone(1, SILENCE_RMS)), false); // sine RMS = amplitude / sqrt(2)
  assert.equal(hasSpeechEnergy(tone(1, 0.05)), true);
  const tail = new Float32Array(16000 + 10);
  tail.fill(0.01, 16000);
  assert.equal(hasSpeechEnergy(tail), true);
});

test('createByteProgress weights by bytes, uses the expected total, and never goes back', () => {
  const reported = [];
  const update = createByteProgress(1000, (f) => reported.push(f));
  update('config.json', 10, 10);
  update('model.onnx', 200, 900);
  update('model.onnx', 201, 900); // below the reporting step
  update('late.onnx', 0, 600); // total grows past the expected size
  update('model.onnx', 900, 900);
  update('late.onnx', 600, 600);
  assert.deepEqual(reported, [0.01, 0.21, 910 / 1510, 1]);
  for (let i = 1; i < reported.length; i++) assert.ok(reported[i] > reported[i - 1]);
});

test('transformersProgressHandler forwards only byte progress events', () => {
  const seen = [];
  const handler = transformersProgressHandler((...args) => seen.push(args));
  handler({ status: 'initiate', file: 'a' });
  handler({ status: 'progress', file: 'a', loaded: 5, total: 10, progress: 50 });
  handler({ status: 'progress', file: 'b', loaded: undefined, total: 10 });
  handler({ status: 'done', file: 'a' });
  handler({ status: 'ready' });
  assert.deepEqual(seen, [['a', 5, 10]]);
});

test('readResponseBytes streams the body and reports byte counts', async () => {
  const parts = [new Uint8Array([1, 2]), new Uint8Array([3, 4, 5])];
  const body = new ReadableStream({
    start(controller) {
      for (const part of parts) controller.enqueue(part);
      controller.close();
    },
  });
  const seen = [];
  const bytes = await readResponseBytes(new Response(body, { headers: { 'content-length': '5' } }), (...args) => seen.push(args));
  assert.deepEqual([...bytes], [1, 2, 3, 4, 5]);
  assert.deepEqual(seen, [[2, 5], [5, 5], [5, 5]]);
});

test('diarizationFraction maps pipeline steps to one increasing 0..1 value', () => {
  const steps = [
    { step: 'segmentation', fraction: 0 },
    { step: 'segmentation', fraction: 1 },
    { step: 'embedding', fraction: 0.5 },
    { step: 'embedding', fraction: 1 },
    { step: 'clustering', fraction: 1 },
    { step: 'reconstruction', fraction: 1 },
  ].map(diarizationFraction);
  assert.equal(steps[0], 0);
  assert.ok(Math.abs(steps.at(-1) - 1) < 1e-12);
  for (let i = 1; i < steps.length; i++) assert.ok(steps[i] > steps[i - 1]);
  assert.equal(diarizationFraction({ step: 'unknown', fraction: 1 }), 0);
  assert.equal(diarizationFraction({ step: 'embedding', fraction: NaN }), 0.25);
});

const preloadAsr = (id, asrModel) => ({ type: 'preload', id, model: 'asr', asrModel });
const preloadDiarization = (id) => ({ type: 'preload', id, model: 'diarization' });

test('enqueueRequest puts tasks in order before the waiting preloads', () => {
  const waiting = [];
  assert.equal(enqueueRequest(waiting, preloadAsr(1, 'tiny.en')), null);
  assert.equal(enqueueRequest(waiting, preloadDiarization(2)), null);
  assert.equal(enqueueRequest(waiting, { type: 'transcribe', id: 3 }), null);
  assert.equal(enqueueRequest(waiting, { type: 'diarize', id: 4 }), null);
  assert.deepEqual(waiting.map((request) => request.id), [3, 4, 1, 2]);
});

test('enqueueRequest appends a task when no preload is waiting', () => {
  const waiting = [{ type: 'transcribe', id: 1 }];
  enqueueRequest(waiting, { type: 'diarize', id: 2 });
  assert.deepEqual(waiting.map((request) => request.id), [1, 2]);
});

test('enqueueRequest replaces a waiting preload of the same model and returns it', () => {
  const waiting = [];
  enqueueRequest(waiting, preloadAsr(1, 'base.en'));
  enqueueRequest(waiting, preloadDiarization(2));
  const replaced = enqueueRequest(waiting, preloadAsr(3, 'small.en'));
  assert.equal(replaced.id, 1);
  assert.deepEqual(waiting.map((request) => request.id), [3, 2]);
  assert.equal(enqueueRequest(waiting, preloadDiarization(4)).id, 2);
  assert.deepEqual(waiting.map((request) => request.id), [3, 4]);
});

test('asrWindowCount matches the Transformers.js Whisper window loop', () => {
  // The window loop of AutomaticSpeechRecognitionPipeline._call_whisper (3.8.1).
  const windowsLike = (length) => {
    const window = SAMPLE_RATE * ASR_CHUNK_OPTIONS.chunk_length_s;
    const jump = window - 2 * SAMPLE_RATE * ASR_CHUNK_OPTIONS.stride_length_s;
    let count = 0;
    for (let offset = 0; ; offset += jump) {
      count++;
      if (offset + window >= length) return count;
    }
  };
  const seconds = [0, 1, 29.99, 30, 30.01, 49.99, 50, 50.01, 70, 3600, 5400.5];
  for (const length of seconds.map((s) => Math.round(s * SAMPLE_RATE))) {
    assert.equal(asrWindowCount(length), windowsLike(length), `${length} samples`);
  }
  assert.equal(asrWindowCount(30 * SAMPLE_RATE), 1);
  assert.equal(asrWindowCount(3600 * SAMPLE_RATE), 180);
});

function fakeModelSlot({ failLoads = new Set(), disposeError } = {}) {
  const events = [];
  const get = createModelSlot({
    async load(key, context) {
      events.push(`load ${key} ${context}`);
      if (failLoads.has(key)) {
        failLoads.delete(key);
        throw new Error(`load ${key} failed`);
      }
      return { key };
    },
    async dispose(model) {
      events.push(`dispose ${model.key}`);
      // Finish in a later macrotask, like an async session release, so a load
      // that does not wait for the dispose would come before "disposed".
      await new Promise((resolve) => setImmediate(resolve));
      events.push(`disposed ${model.key}`);
      if (disposeError) throw disposeError;
    },
  });
  return { get, events };
}

test('createModelSlot loads a model once and reuses it for the same key', async () => {
  const { get, events } = fakeModelSlot();
  const first = await get('tiny', 'a');
  const second = await get('tiny', 'b');
  assert.equal(second, first);
  assert.deepEqual(events, ['load tiny a']);
});

test('createModelSlot disposes the loaded model before it loads a different one', async () => {
  const { get, events } = fakeModelSlot();
  await get('tiny', 'a');
  assert.deepEqual(await get('small', 'b'), { key: 'small' });
  assert.deepEqual(await get('tiny', 'c'), { key: 'tiny' });
  assert.deepEqual(events, [
    'load tiny a',
    'dispose tiny', 'disposed tiny', 'load small b',
    'dispose small', 'disposed small', 'load tiny c',
  ]);
});

test('createModelSlot tries a failed load again on the next request', async () => {
  const { get, events } = fakeModelSlot({ failLoads: new Set(['small']) });
  await assert.rejects(get('small', 'a'), /load small failed/);
  assert.deepEqual(await get('small', 'b'), { key: 'small' });
  assert.deepEqual(events, ['load small a', 'load small b']);
});

test('createModelSlot has nothing to dispose after a failed load', async () => {
  const { get, events } = fakeModelSlot({ failLoads: new Set(['small']) });
  await assert.rejects(get('small', 'a'), /load small failed/);
  assert.deepEqual(await get('tiny', 'b'), { key: 'tiny' });
  assert.deepEqual(events, ['load small a', 'load tiny b']);
});

test('createModelSlot still loads the new model when dispose fails', async () => {
  const { get, events } = fakeModelSlot({ disposeError: new Error('dispose failed') });
  await get('tiny', 'a');
  assert.deepEqual(await get('small', 'b'), { key: 'small' });
  assert.deepEqual(events, ['load tiny a', 'dispose tiny', 'disposed tiny', 'load small b']);
});
