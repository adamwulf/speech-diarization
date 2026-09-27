import test from 'node:test';
import assert from 'node:assert/strict';
import {
  assignSpeakers,
  dropNonSpeechTags,
  groupWords,
  MIN_WORD_SECONDS,
  normalizeTurns,
  normalizeWords,
  relabelSpeakers,
  whisperChunksToWords,
  wordsFromSegments,
} from '../src/alignment.js';

const word = (text, start, end, speaker) => ({ text, start, end, ...(speaker !== undefined && { speaker }) });

function assertValidWords(words, duration) {
  let previousStart = 0;
  for (const w of words) {
    assert.ok(Number.isFinite(w.start) && Number.isFinite(w.end), `finite ${JSON.stringify(w)}`);
    assert.ok(w.start >= 0 && w.end > w.start && w.end <= duration, `bounded ${JSON.stringify(w)}`);
    assert.ok(w.start >= previousStart, `ordered ${JSON.stringify(w)}`);
    previousStart = w.start;
  }
}

test('whisperChunksToWords reads Transformers.js word chunks', () => {
  assert.deepEqual(
    whisperChunksToWords([{ text: ' And', timestamp: [0, 0.5] }, { text: ' so', timestamp: [0.5, null] }, { text: ' x' }]),
    [word(' And', 0, 0.5), word(' so', 0.5, null), word(' x', undefined, undefined)],
  );
  assert.deepEqual(whisperChunksToWords(undefined), []);
});

test('normalizeWords clamps timestamps into the audio and keeps end after start', () => {
  const words = normalizeWords([
    word(' early', -0.3, 0.4),
    word(' open', 1, null),
    word(' zero', 2, 2),
    word(' late', 9.5, 12),
    word(' after', 11, 13),
  ], 10);
  assertValidWords(words, 10);
  assert.deepEqual(words.map((w) => [w.start, w.end]), [
    [0, 0.4],
    [1, 1 + MIN_WORD_SECONDS],
    [2, 2 + MIN_WORD_SECONDS],
    [9.5, 10 - MIN_WORD_SECONDS],
    [10 - MIN_WORD_SECONDS, 10],
  ]);
});

test('normalizeWords orders words and trims an overlapping previous word', () => {
  const words = normalizeWords([
    word(' one', 1, 2.5),
    word(' two', 2, 3),
    word(' back', 1.5, 3.2),
    word(' missing', undefined, NaN),
  ], 10);
  assertValidWords(words, 10);
  assert.deepEqual(words.map((w) => [w.text, w.start, w.end]), [
    [' one', 1, 2],
    [' two', 2, 2 + MIN_WORD_SECONDS],
    [' back', 2, 3.2],
    [' missing', 3.2, 3.2 + MIN_WORD_SECONDS],
  ]);
});

test('normalizeWords keeps Whisper spacing and drops blank words', () => {
  const words = normalizeWords([word('\n  well', 0, 0.3), word('-known', 0.3, 0.6), word('  ', 0.6, 0.7), word(' fact. ', 0.7, 1)], 5);
  assert.deepEqual(words.map((w) => w.text), [' well', '-known', ' fact.']);
});

test('normalizeWords returns nothing for audio shorter than one word step', () => {
  assert.deepEqual(normalizeWords([word(' hi', 0, 0.01)], MIN_WORD_SECONDS), []);
  assert.deepEqual(normalizeWords(null, 5), []);
});

test('dropNonSpeechTags removes Whisper tags, also when split over words', () => {
  const words = [
    word(' [BLANK', 0, 1), word('_AUDIO]', 1, 2),
    word(' Hello', 2, 2.5),
    word(' [MUSIC', 3, 3.5), word(' PLAYING]', 3.5, 4),
    word(' [sic]', 4, 4.2),
    word(' [unclosed', 4.2, 4.4), word(' words', 4.4, 4.6),
  ];
  assert.deepEqual(dropNonSpeechTags(words).map((w) => w.text), [' Hello', ' [sic]', ' [unclosed', ' words']);
});

test('normalizeTurns drops unusable turns, clamps, and sorts', () => {
  assert.deepEqual(normalizeTurns([
    { start: 5, end: 12, speaker: 'B' },
    { start: -1, end: 2, speaker: 'A' },
    { start: 3, end: 3, speaker: 'A' },
    { start: NaN, end: 4, speaker: 'A' },
    { start: 1, end: 2, speaker: '' },
    { start: 1, end: 2 },
    null,
  ], 10), [
    { start: 0, end: 2, speaker: 'A' },
    { start: 5, end: 10, speaker: 'B' },
  ]);
});

test('assignSpeakers picks the speaker with the most total overlap', () => {
  const turns = normalizeTurns([
    { start: 0, end: 1.2, speaker: 'A' },
    { start: 1.2, end: 3, speaker: 'B' },
    { start: 3, end: 3.3, speaker: 'A' },
    { start: 3.3, end: 3.6, speaker: 'A' },
  ], 10);
  const labeled = assignSpeakers([
    word(' mostly-a', 0.8, 1.4), // A 0.4 s, B 0.2 s
    word(' mostly-b', 1.5, 2.5), // B only
    word(' split', 2.6, 3.6), // B 0.4 s; A 0.3 + 0.3 s over two turns
  ], turns);
  assert.deepEqual(labeled.map((w) => w.speaker), ['A', 'B', 'A']);
});

test('assignSpeakers snaps words in short gaps and leaves far words null', () => {
  const turns = normalizeTurns([{ start: 0, end: 1, speaker: 'A' }, { start: 4, end: 5, speaker: 'B' }], 10);
  const labeled = assignSpeakers([
    word(' near-a', 1.2, 1.4),
    word(' near-b', 3.6, 3.8),
    word(' alone', 2.2, 2.6),
    word(' end', 8, 9),
  ], turns);
  assert.deepEqual(labeled.map((w) => w.speaker), ['A', 'B', null, null]);
  assert.deepEqual(assignSpeakers([word(' x', 0, 1)], []).map((w) => w.speaker), [null]);
});

test('relabelSpeakers numbers speakers by first appearance and keeps null', () => {
  const words = relabelSpeakers([
    word(' a', 0, 1, 'SPEAKER_03'),
    word(' b', 1, 2, null),
    word(' c', 2, 3, 'SPEAKER_00'),
    word(' d', 3, 4, 'SPEAKER_03'),
  ]);
  assert.deepEqual(words.map((w) => w.speaker), ['SPEAKER_1', null, 'SPEAKER_2', 'SPEAKER_1']);
});

test('groupWords splits cues at speaker changes, pauses, and sentence ends', () => {
  const segments = groupWords([
    word(' Hello', 0, 0.4, 'S1'), word(' there.', 0.4, 0.8, 'S1'),
    word(' How', 0.9, 1.1, 'S1'), word(' are', 1.1, 1.3, 'S1'),
    word(' you?', 1.3, 1.6, 'S2'),
    word(' Well', 3, 3.3, 'S2'), word('-known', 3.3, 3.8, 'S2'),
  ]);
  assert.deepEqual(segments.map((s) => [s.start, s.end, s.speaker, s.text]), [
    [0, 0.8, 'S1', 'Hello there.'],
    [0.9, 1.3, 'S1', 'How are'],
    [1.3, 1.6, 'S2', 'you?'],
    [3, 3.8, 'S2', 'Well-known'],
  ]);
  assert.deepEqual(segments[3].words, [word(' Well', 3, 3.3), word('-known', 3.3, 3.8)]);
});

test('groupWords keeps cues within the length limits', () => {
  const many = Array.from({ length: 40 }, (_, i) => word(' word', i * 0.3, i * 0.3 + 0.25, null));
  const segments = groupWords(many, { maxSeconds: 3, maxChars: 20, maxPauseSeconds: 1 });
  for (const s of segments) {
    assert.ok(s.text.length <= 20, s.text);
    assert.ok(s.end - s.start <= 3, `${s.start}-${s.end}`);
  }
  assert.equal(segments.flatMap((s) => s.words).length, 40);
  assert.deepEqual(groupWords([]), []);
});

test('wordsFromSegments round-trips grouped words and handles segments without words', () => {
  const words = normalizeWords([word(' One', 0, 0.5), word(' two.', 0.5, 1), word(' Three', 1.2, 1.6)], 5)
    .map((w) => ({ ...w, speaker: null }));
  const segments = groupWords(words);
  assert.deepEqual(normalizeWords(wordsFromSegments(segments), 5), words.map(({ speaker, ...rest }) => rest));

  const fromText = normalizeWords(wordsFromSegments([{ start: 2, end: 3, text: 'Typed text', speaker: null }]), 5);
  assert.deepEqual(fromText, [word(' Typed text', 2, 3)]);
  assert.deepEqual(wordsFromSegments(null), []);
});

test('alignment keeps speaker identity across diarization windows', () => {
  // Two speakers alternate over 30 s, so each speaker appears in several
  // 10 s segmentation windows. The diarizer's global labels must map to one
  // stable ID per speaker for the whole transcript.
  const turns = normalizeTurns([
    { start: 0.5, end: 4, speaker: 'SPEAKER_01' },
    { start: 4.2, end: 9, speaker: 'SPEAKER_00' },
    { start: 9.1, end: 16, speaker: 'SPEAKER_01' },
    { start: 16.2, end: 24, speaker: 'SPEAKER_00' },
    { start: 24.3, end: 29.5, speaker: 'SPEAKER_01' },
  ], 30);
  const words = normalizeWords([
    word(' Hi', 0.6, 1), word(' there.', 1, 3.8),
    word(' Hello', 4.3, 5), word(' back.', 5, 8.8),
    word(' Second', 9.2, 10), word(' turn.', 10, 15.9),
    word(' Yes', 16.3, 17), word(' indeed.', 17, 23),
    word(' Bye', 24.4, 25), word(' now.', 25, 29.4),
  ], 30);
  const segments = groupWords(relabelSpeakers(assignSpeakers(words, turns)));
  assert.deepEqual(segments.map((s) => [s.speaker, s.text]), [
    ['SPEAKER_1', 'Hi there.'],
    ['SPEAKER_2', 'Hello back.'],
    ['SPEAKER_1', 'Second turn.'],
    ['SPEAKER_2', 'Yes indeed.'],
    ['SPEAKER_1', 'Bye now.'],
  ]);
});
