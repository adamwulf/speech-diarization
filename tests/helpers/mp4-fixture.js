// Builds small, non-fragmented MP4 files for tests: an optional video track
// and an optional AAC audio track, interleaved chunk by chunk in one 'mdat',
// with the 'moov' box either before or after it. Video payloads are always
// placeholder bytes; audio payloads are placeholders unless real encoded AAC
// frames are passed in.

const encoder = new TextEncoder();

function concat(parts) {
  const bytes = new Uint8Array(parts.reduce((sum, part) => sum + part.length, 0));
  let offset = 0;
  for (const part of parts) {
    bytes.set(part, offset);
    offset += part.length;
  }
  return bytes;
}

const u8 = (...values) => Uint8Array.from(values);
const u16 = (value) => u8(value >>> 8 & 255, value & 255);
const u32 = (value) => u8(value >>> 24 & 255, value >>> 16 & 255, value >>> 8 & 255, value & 255);
const zeros = (length) => new Uint8Array(length);
const text = (value) => encoder.encode(value);

function box(type, ...payload) {
  const body = concat(payload);
  return concat([u32(8 + body.length), text(type), body]);
}

function fullBox(type, version, flags, ...payload) {
  return box(type, u8(version, flags >>> 16 & 255, flags >>> 8 & 255, flags & 255), ...payload);
}

/** An MPEG-4 descriptor with a one-byte length (enough for these tests). */
function descriptor(tag, ...payload) {
  const body = concat(payload);
  return concat([u8(tag, body.length), body]);
}

const IDENTITY_MATRIX = concat([u32(0x10000), u32(0), u32(0), u32(0), u32(0x10000), u32(0), u32(0), u32(0), u32(0x40000000)]);

function mvhd(timescale, duration, nextTrackId) {
  return fullBox('mvhd', 0, 0, u32(0), u32(0), u32(timescale), u32(duration), u32(0x10000), u16(0x100), zeros(10),
    IDENTITY_MATRIX, zeros(24), u32(nextTrackId));
}

function tkhd(trackId, duration, { volume = 0, width = 0, height = 0 }) {
  return fullBox('tkhd', 0, 3, u32(0), u32(0), u32(trackId), u32(0), u32(duration), zeros(8), u16(0), u16(0),
    u16(volume), u16(0), IDENTITY_MATRIX, u32(width << 16), u32(height << 16));
}

function mdhd(timescale, duration) {
  return fullBox('mdhd', 0, 0, u32(0), u32(0), u32(timescale), u32(duration), u16(0x55c4), u16(0));
}

function hdlr(handler) {
  return fullBox('hdlr', 0, 0, u32(0), text(handler), zeros(12), text('fixture'), u8(0));
}

const dinf = () => box('dinf', fullBox('dref', 0, 0, u32(1), fullBox('url ', 0, 1)));

function stbl(sampleEntry, track, chunkOffsets) {
  const { sizes, samplesPerChunk, delta } = track;
  return box('stbl',
    fullBox('stsd', 0, 0, u32(1), sampleEntry),
    fullBox('stts', 0, 0, u32(1), u32(sizes.length), u32(delta)),
    fullBox('stsc', 0, 0, u32(1), u32(1), u32(samplesPerChunk), u32(1)),
    fullBox('stsz', 0, 0, u32(0), u32(sizes.length), concat(sizes.map(u32))),
    fullBox('stco', 0, 0, u32(chunkOffsets.length), concat(chunkOffsets.map(u32))));
}

function avc1(width, height) {
  return box('avc1', zeros(6), u16(1), zeros(16), u16(width), u16(height), u32(0x480000), u32(0x480000), u32(0),
    u16(1), zeros(32), u16(24), u16(0xffff));
}

function mp4a(sampleRate, channels, specificInfo) {
  const esds = fullBox('esds', 0, 0, descriptor(3, u16(1), u8(0),
    descriptor(4, u8(0x40, 0x15), zeros(3), u32(128000), u32(128000), descriptor(5, specificInfo)),
    descriptor(6, u8(2))));
  return box('mp4a', zeros(6), u16(1), zeros(8), u16(channels), u16(16), u16(0), u16(0), u32(sampleRate << 16), esds);
}

/** AudioSpecificConfig for AAC-LC at 48 kHz, stereo. */
export const AAC_LC_48K_STEREO = u8(0x11, 0x90);

/**
 * Returns `{ bytes, audioSamples }`. `audioSamples` holds the payload of each
 * audio sample, in order. Options:
 * - `moovFirst` (default true): put 'moov' before 'mdat'.
 * - `video`: `{ chunks, sampleSize }`, or null for no video track.
 * - `audio`: `{ samples, samplesPerChunk, sampleRate, channels }`, or null.
 *   Add `payloads` (encoded AAC frames, which replace `samples`) and
 *   `specificInfo` (their AudioSpecificConfig) to store real audio.
 */
export function buildMp4({
  moovFirst = true,
  video = { chunks: 10, sampleSize: 500 },
  audio = { samples: 40, samplesPerChunk: 4, sampleRate: 48000, channels: 2 },
} = {}) {
  const tracks = [];
  if (video) {
    const videoPayload = new Uint8Array(video.sampleSize).fill(0xee); // shared by every video sample
    tracks.push({
      kind: 'video', timescale: 30, delta: 1, samplesPerChunk: 1,
      sizes: Array.from({ length: video.chunks }, () => video.sampleSize),
      payload: () => videoPayload,
    });
  }
  const audioSamples = [];
  if (audio) {
    const { payloads } = audio;
    tracks.push({
      kind: 'audio', timescale: audio.sampleRate, delta: 1024, samplesPerChunk: audio.samplesPerChunk,
      sizes: payloads ? payloads.map((bytes) => bytes.length) : Array.from({ length: audio.samples }, (_, i) => 20 + (i % 7)),
      payload: (i, size) => {
        const bytes = payloads ? payloads[i] : new Uint8Array(size).fill((i + 1) & 255);
        audioSamples.push(bytes);
        return bytes;
      },
      sampleRate: audio.sampleRate,
      channels: audio.channels,
      specificInfo: audio.specificInfo ?? AAC_LC_48K_STEREO,
    });
  }

  // Interleave: chunk 0 of each track, then chunk 1 of each track, and so on.
  const layout = [];
  for (const track of tracks) track.chunkCount = Math.ceil(track.sizes.length / track.samplesPerChunk);
  const maxChunks = Math.max(0, ...tracks.map((track) => track.chunkCount));
  for (let chunk = 0; chunk < maxChunks; chunk++) {
    for (const track of tracks) {
      if (chunk >= track.chunkCount) continue;
      const first = chunk * track.samplesPerChunk;
      const parts = track.sizes.slice(first, first + track.samplesPerChunk).map((size, j) => track.payload(first + j, size));
      layout.push({ track, parts, length: parts.reduce((sum, part) => sum + part.length, 0) });
    }
  }
  const mdatParts = layout.flatMap((chunk) => chunk.parts);
  const mdatLength = layout.reduce((sum, chunk) => sum + chunk.length, 0);

  const moovFor = (mdatDataStart) => {
    const offsets = new Map(tracks.map((track) => [track, []]));
    let position = mdatDataStart;
    for (const chunk of layout) {
      offsets.get(chunk.track).push(position);
      position += chunk.length;
    }
    const traks = tracks.map((track, index) => {
      const duration = track.sizes.length * track.delta;
      const isVideo = track.kind === 'video';
      const entry = isVideo ? avc1(320, 240) : mp4a(track.sampleRate, track.channels, track.specificInfo);
      return box('trak',
        tkhd(index + 1, 0, isVideo ? { width: 320, height: 240 } : { volume: 0x100 }),
        box('mdia',
          mdhd(track.timescale, duration),
          hdlr(isVideo ? 'vide' : 'soun'),
          box('minf',
            isVideo ? fullBox('vmhd', 0, 1, zeros(8)) : fullBox('smhd', 0, 0, zeros(4)),
            dinf(),
            stbl(entry, track, offsets.get(track)))));
    });
    return box('moov', mvhd(1000, 0, tracks.length + 1), ...traks);
  };

  const ftyp = box('ftyp', text('isom'), u32(0x200), text('isom'), text('mp41'));
  const mdatHeader = concat([u32(8 + mdatLength), text('mdat')]);
  let parts;
  if (moovFirst) {
    const moovSize = moovFor(0).length; // offsets are fixed-width, so the size doesn't depend on them
    parts = [ftyp, moovFor(ftyp.length + moovSize + mdatHeader.length), mdatHeader, ...mdatParts];
  } else {
    parts = [ftyp, mdatHeader, ...mdatParts, moovFor(ftyp.length + mdatHeader.length)];
  }
  // `parts` can build a large Blob without one contiguous copy of the file.
  return {
    parts,
    get bytes() {
      return concat(parts);
    },
    audioSamples,
  };
}
