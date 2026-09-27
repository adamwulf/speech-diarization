import test from 'node:test';
import assert from 'node:assert/strict';
import { decodeMp4AudioTrack, isMp4Header, readMp4AudioTrack } from '../src/mp4-audio.js';
import { AAC_LC_48K_STEREO, buildMp4 } from './helpers/mp4-fixture.js';

/** A Blob stand-in that records every range read from it. */
function recordingBlob(bytes) {
  const blob = new Blob([bytes]);
  const reads = [];
  return {
    reads,
    size: blob.size,
    slice(start, end) {
      reads.push([start, Math.min(end, blob.size)]);
      return blob.slice(start, end);
    },
  };
}

function assertSamplesMatch(bytes, audio, expected) {
  assert.equal(audio.samples.length, expected.length);
  audio.samples.forEach((sample, i) => {
    assert.deepEqual(bytes.subarray(sample.offset, sample.offset + sample.size), expected[i], `sample ${i}`);
    assert.equal(sample.cts, i * 1024);
    assert.equal(sample.duration, 1024);
  });
}

test('isMp4Header recognizes ISO BMFF starts', () => {
  const header = (type) => Uint8Array.from([0, 0, 0, 32, ...new TextEncoder().encode(type)]);
  for (const type of ['ftyp', 'moov', 'mdat', 'wide', 'free']) assert.equal(isMp4Header(header(type)), true, type);
  assert.equal(isMp4Header(new TextEncoder().encode('RIFF\0\0\0\0WAVE')), false);
  assert.equal(isMp4Header(new TextEncoder().encode('ID3\x04\0\0\0\0')), false);
  assert.equal(isMp4Header(header('ftyp').subarray(0, 7)), false);
});

test('readMp4AudioTrack describes the audio track of a video with moov first', async () => {
  const { bytes, audioSamples } = buildMp4({ moovFirst: true });
  const movie = await readMp4AudioTrack(new Blob([bytes]));
  assert.equal(movie.audio.codec, 'mp4a.40.2');
  assert.equal(movie.audio.sampleRate, 48000);
  assert.equal(movie.audio.numberOfChannels, 2);
  assert.deepEqual(movie.audio.description, AAC_LC_48K_STEREO);
  assert.equal(movie.audio.timescale, 48000);
  assert.equal(movie.audio.duration, (40 * 1024) / 48000);
  assertSamplesMatch(bytes, movie.audio, audioSamples);
});

test('readMp4AudioTrack skips the media data to find moov at the end', async () => {
  // About 20 MB of video before the metadata.
  const { bytes, audioSamples } = buildMp4({ moovFirst: false, video: { chunks: 10, sampleSize: 2 * 1024 * 1024 } });
  const blob = recordingBlob(bytes);
  const movie = await readMp4AudioTrack(blob);
  assertSamplesMatch(bytes, movie.audio, audioSamples);
  const bytesRead = blob.reads.reduce((sum, [start, end]) => sum + (end - start), 0);
  assert.ok(bytesRead < bytes.length / 2, `read ${bytesRead} of ${bytes.length} bytes`);
});

test('readMp4AudioTrack finds audio-only files (M4A)', async () => {
  const { bytes, audioSamples } = buildMp4({ video: null });
  const movie = await readMp4AudioTrack(new Blob([bytes]));
  assertSamplesMatch(bytes, movie.audio, audioSamples);
});

test('readMp4AudioTrack reports a video without an audio track', async () => {
  const { bytes } = buildMp4({ audio: null });
  assert.deepEqual(await readMp4AudioTrack(new Blob([bytes])), { audio: null });
});

test('readMp4AudioTrack returns null for other files', async () => {
  assert.equal(await readMp4AudioTrack(new Blob([new TextEncoder().encode('RIFF\0\0\0\0WAVEfmt ')])), null);
  assert.equal(await readMp4AudioTrack(new Blob([])), null);
  // An MP4 cut off before its metadata.
  const { bytes } = buildMp4({ moovFirst: false });
  assert.equal(await readMp4AudioTrack(new Blob([bytes.subarray(0, 100)])), null);
});

// ---- decodeMp4AudioTrack, with a stand-in for WebCodecs ----

/**
 * Installs fake WebCodecs globals. Each decoded chunk becomes 1024 stereo
 * frames at 48 kHz: the left channel is chunk.data[0] / 100 and the right
 * channel is 0, so the mono mix is chunk.data[0] / 200.
 */
function installFakeWebCodecs({ supported = true, failAt = -1 } = {}) {
  const decoded = [];
  class FakeEncodedAudioChunk {
    constructor({ type, timestamp, duration, data }) {
      Object.assign(this, { type, timestamp, duration, data: new Uint8Array(data) });
    }
  }
  class FakeAudioDecoder extends EventTarget {
    static async isConfigSupported(config) {
      return { supported, config };
    }

    #output;
    #error;
    #pending = 0;
    state = 'unconfigured';

    constructor({ output, error }) {
      super();
      this.#output = output;
      this.#error = error;
    }

    get decodeQueueSize() {
      return this.#pending;
    }

    configure(config) {
      this.config = config;
      this.state = 'configured';
    }

    decode(chunk) {
      if (this.state !== 'configured') throw new Error('InvalidStateError: decode() while ' + this.state);
      const index = decoded.length;
      decoded.push(chunk);
      this.#pending++;
      setTimeout(() => {
        this.#pending--;
        this.dispatchEvent(new Event('dequeue'));
        if (this.state !== 'configured') return;
        if (index === failAt) {
          this.state = 'closed';
          this.#error(new Error('EncodingError: bad chunk'));
          return;
        }
        const level = chunk.data[0] / 100;
        this.#output({
          sampleRate: 48000,
          numberOfFrames: 1024,
          numberOfChannels: 2,
          copyTo(destination, { planeIndex, format }) {
            assert.equal(format, 'f32-planar');
            destination.fill(planeIndex === 0 ? level : 0);
          },
          close() {},
        });
      }, 0);
    }

    async flush() {
      while (this.#pending) await new Promise((resolve) => setTimeout(resolve, 1));
      if (this.state === 'closed') throw new Error('InvalidStateError: flush() while closed');
    }

    close() {
      this.state = 'closed';
    }
  }
  globalThis.AudioDecoder = FakeAudioDecoder;
  globalThis.EncodedAudioChunk = FakeEncodedAudioChunk;
  return decoded;
}

function removeFakeWebCodecs() {
  delete globalThis.AudioDecoder;
  delete globalThis.EncodedAudioChunk;
}

test('decodeMp4AudioTrack returns null without WebCodecs', async () => {
  const { bytes } = buildMp4();
  const blob = new Blob([bytes]);
  const { audio } = await readMp4AudioTrack(blob);
  assert.equal(await decodeMp4AudioTrack(blob, audio, 16000), null);
});

test('decodeMp4AudioTrack returns null when the codec is unsupported', async (t) => {
  t.after(removeFakeWebCodecs);
  installFakeWebCodecs({ supported: false });
  const { bytes } = buildMp4();
  const blob = new Blob([bytes]);
  const { audio } = await readMp4AudioTrack(blob);
  assert.equal(await decodeMp4AudioTrack(blob, audio, 16000), null);
});

test('decodeMp4AudioTrack decodes every sample in order and resamples to mono', async (t) => {
  t.after(removeFakeWebCodecs);
  const decoded = installFakeWebCodecs();
  const { bytes, audioSamples } = buildMp4({ moovFirst: false, audio: { samples: 200, samplesPerChunk: 5, sampleRate: 48000, channels: 2 } });
  const blob = new Blob([bytes]);
  const { audio } = await readMp4AudioTrack(blob);
  const progress = [];
  const pcm = await decodeMp4AudioTrack(blob, audio, 16000, { onProgress: (fraction) => progress.push(fraction) });

  assert.equal(decoded.length, 200);
  decoded.forEach((chunk, i) => {
    assert.equal(chunk.type, 'key');
    assert.equal(chunk.timestamp, Math.round((i * 1024 * 1e6) / 48000));
    assert.equal(chunk.duration, Math.round((1024 * 1e6) / 48000));
    assert.deepEqual(chunk.data, audioSamples[i]);
  });

  assert.equal(pcm.length, Math.ceil((200 * 1024 * 16000) / 48000));
  // In the middle of sample 100's frames, the mono level is its first byte / 200.
  const middle = Math.round(((100 * 1024 + 512) * 16000) / 48000);
  assert.ok(Math.abs(pcm[middle] - audioSamples[100][0] / 200) < 1e-3, `level ${pcm[middle]}`);

  assert.equal(progress.at(-1), 1);
  assert.ok(progress.every((fraction, i) => fraction >= 0 && fraction <= 1 && (i === 0 || fraction >= progress[i - 1])));
});

test('decodeMp4AudioTrack rejects with the decoder error', async (t) => {
  t.after(removeFakeWebCodecs);
  installFakeWebCodecs({ failAt: 5 });
  const { bytes } = buildMp4();
  const blob = new Blob([bytes]);
  const { audio } = await readMp4AudioTrack(blob);
  await assert.rejects(decodeMp4AudioTrack(blob, audio, 16000), /EncodingError: bad chunk/);
});
