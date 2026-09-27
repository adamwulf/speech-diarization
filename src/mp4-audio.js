// Audio-track decoding for MP4-family files (MP4, M4A, MOV), used when
// decodeAudioData rejects a file, for example a long video recording.
// mp4box.js reads the file's sample tables and skips its media data; then only
// the audio track's bytes are read, decoded with WebCodecs, down-mixed, and
// resampled as they arrive. Neither the whole file nor its full-rate audio is
// ever held in memory.

import { MP4BoxBuffer, createFile } from 'mp4box';
import { StreamingResampler } from './resample.js';

const HEADER_READ_BYTES = 4 * 1024 * 1024;
const SAMPLE_READ_BYTES = 8 * 1024 * 1024;
const MAX_DECODE_QUEUE = 64;
const PROGRESS_INTERVAL = 256; // encoded samples between progress reports
const DEQUEUE_WAIT_MS = 50; // for browsers without the decoder's 'dequeue' event

// ISO BMFF files start with one of these top-level boxes (QuickTime files may
// predate 'ftyp').
const LEADING_BOX_TYPES = new Set(['ftyp', 'moov', 'mdat', 'wide', 'free']);
const DECODER_CONFIG_TAG = 4;
const DECODER_SPECIFIC_INFO_TAG = 5;

/** True when the first 8 bytes look like the start of an ISO BMFF file. */
export function isMp4Header(bytes) {
  if (bytes.length < 8) return false;
  return LEADING_BOX_TYPES.has(String.fromCharCode(bytes[4], bytes[5], bytes[6], bytes[7]));
}

/**
 * Reads the movie metadata of an MP4-family file. Resolves to null when the
 * file isn't one (or has no movie metadata), otherwise to `{ audio }`, where
 * `audio` is null when the file has no audio track, or describes the first one:
 * `{ codec, sampleRate, numberOfChannels, description, timescale, duration, samples }`.
 * `samples` lists `{ offset, size, cts, duration }` in decode order; times are
 * in `timescale` units, and `duration` is in seconds.
 */
export async function readMp4AudioTrack(blob) {
  if (!isMp4Header(new Uint8Array(await blob.slice(0, 8).arrayBuffer()))) return null;

  const file = createFile();
  let info = null;
  let parseError = null;
  file.onReady = (value) => { info = value; };
  file.onError = (error) => { parseError = error; };

  // appendBuffer returns the next file position it needs, jumping over media
  // data, so this reads little more than the metadata. A fragmented file keeps
  // its sample tables in 'moof' boxes throughout, so it is read to the end.
  let position = 0;
  while (position < blob.size && !(info && !info.isFragmented)) {
    const data = await blob.slice(position, position + HEADER_READ_BYTES).arrayBuffer();
    const next = file.appendBuffer(MP4BoxBuffer.fromArrayBuffer(data, position));
    if (parseError) throw new Error(`the file could not be parsed (${parseError})`);
    if (!(next > position)) break;
    position = next;
  }
  if (!info) return null;

  const trackInfo = info.audioTracks[0];
  if (!trackInfo) return { audio: null };
  const trak = file.getTrackById(trackInfo.id);
  const entry = trak.mdia.minf.stbl.stsd.entries[0];
  const esds = entry?.esds ?? entry?.wave?.esds;
  const specificInfo = esds?.esd?.findDescriptor(DECODER_CONFIG_TAG)?.findDescriptor(DECODER_SPECIFIC_INFO_TAG)?.data;
  const timescale = trak.mdia.mdhd.timescale;
  const samples = trak.samples.map(({ offset, size, cts, duration }) => ({ offset, size, cts, duration }));
  const totalDuration = samples.reduce((sum, sample) => sum + sample.duration, 0);
  return {
    audio: {
      codec: trackInfo.codec,
      sampleRate: trackInfo.audio.sample_rate,
      numberOfChannels: trackInfo.audio.channel_count,
      description: specificInfo ? new Uint8Array(specificInfo) : undefined,
      timescale,
      duration: totalDuration / timescale,
      samples,
    },
  };
}

/** Averages an AudioData's channels into one Float32Array. */
function toMono(data) {
  const frames = data.numberOfFrames;
  const channels = data.numberOfChannels;
  const mono = new Float32Array(frames);
  if (channels === 1) {
    data.copyTo(mono, { planeIndex: 0, format: 'f32-planar' });
    return mono;
  }
  const plane = new Float32Array(frames);
  for (let channel = 0; channel < channels; channel++) {
    data.copyTo(plane, { planeIndex: channel, format: 'f32-planar' });
    for (let i = 0; i < frames; i++) mono[i] += plane[i];
  }
  for (let i = 0; i < frames; i++) mono[i] /= channels;
  return mono;
}

/** A Float32Array that grows as chunks are appended. */
class PcmBuffer {
  #data;
  #length = 0;

  constructor(capacity) {
    this.#data = new Float32Array(Math.max(1, capacity));
  }

  append(chunk) {
    const needed = this.#length + chunk.length;
    if (needed > this.#data.length) {
      const grown = new Float32Array(Math.max(needed, Math.ceil(this.#data.length * 1.5)));
      grown.set(this.#data.subarray(0, this.#length));
      this.#data = grown;
    }
    this.#data.set(chunk, this.#length);
    this.#length = needed;
  }

  result() {
    return this.#data.subarray(0, this.#length);
  }
}

/** Resolves when the decoder takes a chunk from its queue (or after a short wait). */
function dequeued(decoder) {
  return new Promise((resolve) => {
    const done = () => {
      clearTimeout(timer);
      decoder.removeEventListener('dequeue', done);
      resolve();
    };
    const timer = setTimeout(done, DEQUEUE_WAIT_MS);
    decoder.addEventListener('dequeue', done);
  });
}

/**
 * Decodes an audio track from readMp4AudioTrack to mono PCM at `outputRate`.
 * Resolves to null when this browser can't decode the track's codec (or has no
 * WebCodecs), and rejects when decoding fails. `onProgress(fraction)` reports
 * how much of the track has been read.
 */
export async function decodeMp4AudioTrack(blob, track, outputRate, { onProgress } = {}) {
  if (typeof AudioDecoder === 'undefined' || typeof EncodedAudioChunk === 'undefined') return null;
  const config = { codec: track.codec, sampleRate: track.sampleRate, numberOfChannels: track.numberOfChannels };
  if (track.description) config.description = track.description;
  try {
    if (!(await AudioDecoder.isConfigSupported(config)).supported) return null;
  } catch {
    return null; // an invalid config, such as a zero sample rate
  }

  const pcm = new PcmBuffer(Math.ceil((track.duration + 1) * outputRate));
  let resampler = null;
  let inputRate = 0;
  let failure = null;
  const decoder = new AudioDecoder({
    output(data) {
      try {
        if (data.sampleRate !== inputRate) {
          if (resampler) pcm.append(resampler.flush());
          inputRate = data.sampleRate;
          resampler = new StreamingResampler(inputRate, outputRate);
        }
        pcm.append(resampler.push(toMono(data)));
      } catch (error) {
        failure ??= error;
      } finally {
        data.close();
      }
    },
    error(error) {
      failure ??= error;
    },
  });

  try {
    decoder.configure(config);
    const { samples, timescale } = track;
    let window = null;
    let windowStart = 0;
    for (let i = 0; i < samples.length && !failure; i++) {
      const { offset, size, cts, duration } = samples[i];
      if (!window || offset < windowStart || offset + size > windowStart + window.length) {
        // Read ahead in large windows: audio samples are interleaved with video.
        windowStart = offset;
        window = new Uint8Array(await blob.slice(offset, offset + Math.max(SAMPLE_READ_BYTES, size)).arrayBuffer());
        if (window.length < size) throw new Error('the file ends before its audio track does');
        if (failure) break; // the decoder failed (and closed) during the read
      }
      decoder.decode(new EncodedAudioChunk({
        type: 'key',
        timestamp: Math.round((cts / timescale) * 1e6),
        duration: Math.round((duration / timescale) * 1e6),
        data: window.subarray(offset - windowStart, offset - windowStart + size),
      }));
      while (decoder.decodeQueueSize > MAX_DECODE_QUEUE && decoder.state === 'configured') {
        await dequeued(decoder);
      }
      if (i % PROGRESS_INTERVAL === 0) onProgress?.(i / samples.length);
    }
    if (!failure) {
      try {
        await decoder.flush();
      } catch (error) {
        failure ??= error;
      }
    }
  } finally {
    if (decoder.state !== 'closed') decoder.close();
  }
  if (failure) throw failure;
  if (resampler) pcm.append(resampler.flush());
  onProgress?.(1);
  return pcm.result();
}
