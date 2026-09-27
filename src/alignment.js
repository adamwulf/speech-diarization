// Pure helpers that turn Whisper word timestamps and diarization turns into
// TranscriptSegment cues. No model or DOM code here, so Node tests cover it.

/** Shortest duration a word may have, in seconds (Whisper's timestamp step). */
export const MIN_WORD_SECONDS = 0.02;

/** A word with no overlapping speaker turn takes the nearest turn within this gap. */
export const MAX_SPEAKER_SNAP_SECONDS = 0.5;

export const CUE_LIMITS = Object.freeze({
  maxSeconds: 7,
  maxChars: 84,
  maxPauseSeconds: 1,
});

const SENTENCE_END = /[.!?]["')\]]*$/;

/** Whisper's non-speech tags, such as [BLANK_AUDIO] or [MUSIC]. */
const NON_SPEECH_TAG = /^\[[A-Z_ ]+\]$/;
const MAX_TAG_WORDS = 6;

function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max);
}

/**
 * Convert Transformers.js word chunks (`{text, timestamp: [start, end]}`) to
 * raw words. Timestamps are not validated here; use `normalizeWords`.
 */
export function whisperChunksToWords(chunks) {
  if (!Array.isArray(chunks)) return [];
  return chunks.map((chunk) => ({
    text: chunk?.text,
    start: chunk?.timestamp?.[0],
    end: chunk?.timestamp?.[1],
  }));
}

/**
 * Flatten transcript segments back to words for re-diarization. A segment
 * without a `words` array counts as one word that spans the segment.
 */
export function wordsFromSegments(segments) {
  if (!Array.isArray(segments)) return [];
  return segments.flatMap((segment) =>
    Array.isArray(segment?.words) && segment.words.length > 0
      ? segment.words
      : [{ text: ` ${segment?.text ?? ''}`, start: segment?.start, end: segment?.end }],
  );
}

/**
 * Make word timestamps finite, bounded to [0, duration], and ordered.
 * Guarantees for each word: 0 <= start < end <= duration, and start values
 * never decrease. A word's end is trimmed to the next word's start when the
 * two overlap. Words with blank text are dropped.
 *
 * Word text keeps one leading space if Whisper put a space before the word;
 * some words attach to the previous word with no space (for example "-known"
 * after "well"). Concatenate word texts to rebuild the text.
 */
export function normalizeWords(rawWords, duration, minWord = MIN_WORD_SECONDS) {
  if (!Array.isArray(rawWords) || !(duration > minWord)) return [];
  const latestStart = duration - minWord;
  const words = [];
  for (const raw of rawWords) {
    if (typeof raw?.text !== 'string' || !raw.text.trim()) continue;
    const text = raw.text.replace(/^\s+/, ' ').trimEnd();
    const prev = words.at(-1);
    let start = Number.isFinite(raw.start) ? raw.start : prev ? prev.end : 0;
    start = clamp(start, prev ? prev.start : 0, latestStart);
    if (prev && prev.end > start) {
      prev.end = Math.max(start, prev.start + minWord);
    }
    const rawEnd = Number.isFinite(raw.end) ? raw.end : start;
    const end = clamp(rawEnd, start + minWord, duration);
    words.push({ start, end, text });
  }
  return words;
}

/**
 * Remove Whisper non-speech tags such as [BLANK_AUDIO]. A tag can be split
 * over more than one word ("[BLANK" + "_AUDIO]").
 */
export function dropNonSpeechTags(words) {
  const kept = [];
  for (let i = 0; i < words.length; i++) {
    let tag = words[i].text.trim();
    if (tag.startsWith('[')) {
      let last = i;
      while (!tag.endsWith(']') && last + 1 < words.length && last - i + 1 < MAX_TAG_WORDS) {
        last += 1;
        tag += words[last].text;
      }
      if (NON_SPEECH_TAG.test(tag)) {
        i = last;
        continue;
      }
    }
    kept.push(words[i]);
  }
  return kept;
}

/**
 * Keep only usable diarization turns, clamped to [0, duration], sorted by start.
 */
export function normalizeTurns(turns, duration) {
  if (!Array.isArray(turns)) return [];
  return turns
    .filter((turn) => typeof turn?.speaker === 'string' && turn.speaker !== ''
      && Number.isFinite(turn.start) && Number.isFinite(turn.end))
    .map((turn) => ({
      start: clamp(turn.start, 0, duration),
      end: clamp(turn.end, 0, duration),
      speaker: turn.speaker,
    }))
    .filter((turn) => turn.end > turn.start)
    .sort((a, b) => a.start - b.start);
}

/**
 * Give each word the speaker with the most total overlap in time. A word that
 * overlaps no turn takes the nearest turn within `maxSnap` seconds, otherwise
 * its speaker is null. Ties go to the speaker whose turn starts first.
 * `turns` must come from `normalizeTurns`.
 */
export function assignSpeakers(words, turns, maxSnap = MAX_SPEAKER_SNAP_SECONDS) {
  return words.map((word) => {
    const overlapBySpeaker = new Map();
    let nearest = null;
    let nearestDistance = Infinity;
    for (const turn of turns) {
      const overlap = Math.min(word.end, turn.end) - Math.max(word.start, turn.start);
      if (overlap > 0) {
        overlapBySpeaker.set(turn.speaker, (overlapBySpeaker.get(turn.speaker) ?? 0) + overlap);
      } else {
        const distance = -overlap;
        if (distance < nearestDistance) {
          nearestDistance = distance;
          nearest = turn.speaker;
        }
      }
    }
    let speaker = null;
    let best = 0;
    for (const [candidate, overlap] of overlapBySpeaker) {
      if (overlap > best) {
        best = overlap;
        speaker = candidate;
      }
    }
    if (speaker === null && nearestDistance <= maxSnap) speaker = nearest;
    return { ...word, speaker };
  });
}

/**
 * Replace diarizer labels with `SPEAKER_1`, `SPEAKER_2`, ... in order of first
 * appearance in the transcript. Null stays null.
 */
export function relabelSpeakers(words) {
  const ids = new Map();
  return words.map((word) => {
    if (word.speaker === null || word.speaker === undefined) return { ...word, speaker: null };
    if (!ids.has(word.speaker)) ids.set(word.speaker, `SPEAKER_${ids.size + 1}`);
    return { ...word, speaker: ids.get(word.speaker) };
  });
}

/**
 * Group consecutive words into subtitle cues. A new cue starts when the
 * speaker changes, after a long pause, after sentence-final punctuation, or
 * when the cue would pass the length limits. Each cue keeps its words so a
 * later diarization pass can realign at word boundaries.
 */
export function groupWords(words, limits = CUE_LIMITS) {
  const segments = [];
  let current = null;
  for (const word of words) {
    const speaker = word.speaker ?? null;
    const startNew = !current
      || speaker !== current.speaker
      || word.start - current.end > limits.maxPauseSeconds
      || word.end - current.start > limits.maxSeconds
      || (current.text + word.text).trim().length > limits.maxChars
      || SENTENCE_END.test(current.text);
    if (startNew) {
      current = { start: word.start, end: word.end, text: '', speaker, words: [] };
      segments.push(current);
    }
    current.text += word.text;
    current.end = Math.max(current.end, word.end);
    current.words.push({ start: word.start, end: word.end, text: word.text });
  }
  for (const segment of segments) segment.text = segment.text.trim();
  return segments;
}
