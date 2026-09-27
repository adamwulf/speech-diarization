import test from 'node:test';
import assert from 'node:assert/strict';
import { StreamingResampler } from '../src/resample.js';

function tone(frequency, rate, length) {
  const samples = new Float32Array(length);
  for (let i = 0; i < length; i++) samples[i] = Math.sin((2 * Math.PI * frequency * i) / rate);
  return samples;
}

/** Resamples `input`, pushing it in blocks of the given sizes (repeated). */
function resample(input, inputRate, outputRate, blockSizes = [input.length || 1]) {
  const resampler = new StreamingResampler(inputRate, outputRate);
  const parts = [];
  let offset = 0;
  for (let i = 0; offset < input.length; i++) {
    const size = blockSizes[i % blockSizes.length];
    parts.push(resampler.push(input.subarray(offset, offset + size)));
    offset += size;
  }
  parts.push(resampler.flush());
  const output = new Float32Array(parts.reduce((sum, part) => sum + part.length, 0));
  let position = 0;
  for (const part of parts) {
    output.set(part, position);
    position += part.length;
  }
  return output;
}

/** Largest difference from a sine of `frequency`, away from the edges. */
function maxToneError(output, frequency, rate, margin) {
  let error = 0;
  for (let n = margin; n < output.length - margin; n++) {
    error = Math.max(error, Math.abs(output[n] - Math.sin((2 * Math.PI * frequency * n) / rate)));
  }
  return error;
}

function rms(samples, margin) {
  let sum = 0;
  for (let i = margin; i < samples.length - margin; i++) sum += samples[i] ** 2;
  return Math.sqrt(sum / (samples.length - 2 * margin));
}

test('the output has ceil(N × out / in) samples, whatever the block sizes', () => {
  for (const inputRate of [48000, 44100, 32000, 22050, 8000]) {
    for (const length of [0, 1, 2, 999, inputRate + 7]) {
      const input = tone(440, inputRate, length);
      const expected = Math.ceil((length * 16000) / inputRate);
      assert.equal(resample(input, inputRate, 16000).length, expected, `${inputRate} Hz × ${length}`);
      assert.equal(resample(input, inputRate, 16000, [1, 7, 1024, 333]).length, expected, `${inputRate} Hz × ${length} in blocks`);
    }
  }
});

test('block sizes do not change the output', () => {
  const input = tone(1000, 44100, 44100);
  const whole = resample(input, 44100, 16000);
  assert.deepEqual(resample(input, 44100, 16000, [1024]), whole);
  assert.deepEqual(resample(input, 44100, 16000, [1, 2, 3, 500, 4096]), whole);
});

test('tones in the pass band keep their amplitude and timing', () => {
  for (const inputRate of [48000, 44100, 32000, 22050, 8000]) {
    for (const frequency of [440, 3000]) {
      if (frequency > 0.45 * Math.min(inputRate, 16000)) continue;
      const output = resample(tone(frequency, inputRate, inputRate), inputRate, 16000, [1024]);
      const error = maxToneError(output, frequency, 16000, 400);
      assert.ok(error < 2e-3, `${frequency} Hz from ${inputRate} Hz: max error ${error}`);
    }
  }
});

test('content above the new Nyquist frequency is removed', () => {
  for (const [inputRate, frequency] of [[48000, 12000], [44100, 9000], [32000, 8500]]) {
    const output = resample(tone(frequency, inputRate, inputRate), inputRate, 16000, [1024]);
    const level = rms(output, 400);
    assert.ok(level < 1e-3, `${frequency} Hz from ${inputRate} Hz: RMS ${level}`);
  }
});

test('equal rates pass the input through unchanged', () => {
  const input = tone(440, 16000, 5000);
  assert.deepEqual(resample(input, 16000, 16000, [1000, 3]), input);
});

test('invalid sample rates throw', () => {
  for (const rate of [0, -16000, 44100.5, NaN, undefined]) {
    assert.throws(() => new StreamingResampler(rate, 16000), RangeError);
    assert.throws(() => new StreamingResampler(48000, rate), RangeError);
  }
});

test('push after flush throws', () => {
  const resampler = new StreamingResampler(48000, 16000);
  resampler.push(new Float32Array(10));
  resampler.flush();
  assert.throws(() => resampler.push(new Float32Array(10)), /after flush/);
  assert.equal(resampler.flush().length, 0);
});
