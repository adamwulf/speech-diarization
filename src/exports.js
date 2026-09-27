// Pure transcript helpers shared by the page and the Markdown / WebVTT exports.
// No DOM access, so this module runs unchanged under `node --test`.
//
// A TranscriptSegment is `{start, end, text, speaker}` (seconds; speaker is a
// worker-assigned id or null). Speaker *names* never live in segments: the page
// keeps a Map of id -> custom name, and anything unnamed falls back to
// "Speaker N", numbered by first appearance.

export const UNKNOWN_SPEAKER_LABEL = 'Unknown speaker';
export const MAX_SPEAKER_NAME_LENGTH = 60;

export const EXPORT_FORMATS = Object.freeze({
  markdown: Object.freeze({ extension: 'md', mimeType: 'text/markdown;charset=utf-8' }),
  webvtt: Object.freeze({ extension: 'vtt', mimeType: 'text/vtt;charset=utf-8' }),
});

function collapseWhitespace(value) {
  return String(value ?? '').replace(/\s+/g, ' ').trim();
}

/**
 * Keeps only well-formed segments (finite, nonnegative, end > start, non-empty
 * text), collapses whitespace so text is always one line, and sorts by start.
 * Returns new objects; the input is not modified.
 */
export function normalizeSegments(segments) {
  if (!Array.isArray(segments)) return [];
  const out = [];
  for (const seg of segments) {
    if (!seg || typeof seg !== 'object') continue;
    const start = Number(seg.start);
    const end = Number(seg.end);
    if (!Number.isFinite(start) || !Number.isFinite(end) || start < 0 || end <= start) continue;
    const text = collapseWhitespace(seg.text);
    if (!text) continue;
    const speaker = typeof seg.speaker === 'string' && seg.speaker !== '' ? seg.speaker : null;
    out.push({ start, end, text, speaker });
  }
  // Array#sort is stable, so segments sharing a start keep the worker's order.
  return out.sort((a, b) => a.start - b.start);
}

/** Distinct non-null speaker ids in order of first appearance. */
export function speakerOrder(segments) {
  const seen = new Set();
  for (const seg of segments) {
    if (seg.speaker !== null && seg.speaker !== undefined) seen.add(seg.speaker);
  }
  return [...seen];
}

export function defaultSpeakerName(index) {
  return `Speaker ${index + 1}`;
}

/** Single-line, trimmed, length-capped name; '' means "use the default". */
export function cleanSpeakerName(value) {
  // Array.from splits by code point so the cap never cuts an emoji in half.
  return Array.from(collapseWhitespace(value)).slice(0, MAX_SPEAKER_NAME_LENGTH).join('').trim();
}

/**
 * Returns `id => display name`. With speakers hidden it always returns null so
 * callers cannot leak names into the display or exports by accident.
 */
export function createSpeakerNamer(segments, customNames = new Map(), showSpeakers = true) {
  if (!showSpeakers) return () => null;
  const index = new Map(speakerOrder(segments).map((id, i) => [id, i]));
  return (id) => {
    if (id === null || id === undefined || !index.has(id)) return UNKNOWN_SPEAKER_LABEL;
    return cleanSpeakerName(customNames.get(id)) || defaultSpeakerName(index.get(id));
  };
}

/**
 * Groups segments into display turns. With speakers shown, consecutive
 * segments from the same speaker merge into one turn; with speakers hidden,
 * every segment stays its own timestamped turn and carries no speaker.
 */
export function groupTurns(segments, { showSpeakers = true } = {}) {
  const turns = [];
  for (const seg of segments) {
    const last = turns[turns.length - 1];
    if (showSpeakers && last && last.speaker === seg.speaker) {
      last.end = Math.max(last.end, seg.end);
      last.text += ` ${seg.text}`;
    } else {
      turns.push({
        start: seg.start,
        end: seg.end,
        text: seg.text,
        speaker: showSpeakers ? seg.speaker : null,
      });
    }
  }
  return turns;
}

function pad(value, width = 2) {
  return String(value).padStart(width, '0');
}

/** "MM:SS", or "H:MM:SS" from one hour on. Fractions are truncated. */
export function formatClock(seconds) {
  const total = Math.max(0, Math.floor(Number(seconds) || 0));
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  return h > 0 ? `${h}:${pad(m)}:${pad(s)}` : `${pad(m)}:${pad(s)}`;
}

function formatVttMilliseconds(ms) {
  const h = Math.floor(ms / 3_600_000);
  const m = Math.floor((ms % 3_600_000) / 60_000);
  const s = Math.floor((ms % 60_000) / 1000);
  return `${pad(h)}:${pad(m)}:${pad(s)}.${pad(ms % 1000, 3)}`;
}

/** WebVTT timestamp "HH:MM:SS.mmm", rounded to the nearest millisecond. */
export function formatVttTimestamp(seconds) {
  return formatVttMilliseconds(Math.max(0, Math.round(Number(seconds) * 1000) || 0));
}

/**
 * Escapes WebVTT cue text (and voice annotations). Escaping `<`, `>` and `&`
 * also rules out a stray "-->" or cue tag; collapsing whitespace rules out the
 * blank line that would end a cue early.
 */
export function escapeVttText(text) {
  return collapseWhitespace(text)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}

/**
 * Escapes inline Markdown so transcript text and names render literally.
 * Output is always a single line, so block syntax (#, -, 1.) can't start here.
 */
export function escapeMarkdown(text) {
  return collapseWhitespace(text)
    .replace(/[\\`*_[\]<>|~$]/g, '\\$&')
    // Only an entity-like "&name;" would be decoded; a lone "&" stays readable.
    .replace(/&(?=#?[a-z0-9]+;)/gi, '\\&');
}

/** One cue per segment; speaker names become WebVTT voice spans when shown. */
export function toWebVTT(segments, { speakerNames = new Map(), showSpeakers = true } = {}) {
  const segs = normalizeSegments(segments);
  const nameOf = createSpeakerNamer(segs, speakerNames, showSpeakers);
  const blocks = ['WEBVTT'];
  segs.forEach((seg, i) => {
    const startMs = Math.round(seg.start * 1000);
    // Millisecond rounding must never produce a zero-length cue.
    const endMs = Math.max(Math.round(seg.end * 1000), startMs + 1);
    const text = escapeVttText(seg.text);
    // Unattributed segments get no voice span rather than an invented voice.
    const name = showSpeakers && seg.speaker !== null ? nameOf(seg.speaker) : null;
    const payload = name ? `<v ${escapeVttText(name)}>${text}</v>` : text;
    blocks.push(`${i + 1}\n${formatVttMilliseconds(startMs)} --> ${formatVttMilliseconds(endMs)}\n${payload}`);
  });
  return `${blocks.join('\n\n')}\n`;
}

/** Timestamped Markdown: one paragraph per turn, same grouping as the page. */
export function toMarkdown(segments, { speakerNames = new Map(), showSpeakers = true, sourceName = '', duration } = {}) {
  const segs = normalizeSegments(segments);
  const nameOf = createSpeakerNamer(segs, speakerNames, showSpeakers);
  const lines = ['# Transcript', ''];

  const meta = [];
  if (collapseWhitespace(sourceName)) meta.push(`- Source: ${escapeMarkdown(sourceName)}`);
  if (Number.isFinite(duration) && duration > 0) meta.push(`- Duration: ${formatClock(duration)}`);
  if (meta.length) lines.push(...meta, '');

  if (!segs.length) lines.push('_No speech detected._', '');
  for (const turn of groupTurns(segs, { showSpeakers })) {
    const time = `[${formatClock(turn.start)} – ${formatClock(turn.end)}]`;
    const who = showSpeakers ? ` ${escapeMarkdown(nameOf(turn.speaker))}:` : '';
    lines.push(`**${time}${who}** ${escapeMarkdown(turn.text)}`, '');
  }
  return `${lines.join('\n').trimEnd()}\n`;
}

/** "<source name without extension>-transcript.<ext>", safe on common filesystems. */
export function exportFileName(sourceName, extension) {
  const base = Array.from(
    collapseWhitespace(sourceName)
      .replace(/\.[^./\\]{1,8}$/, '')
      .replace(/[\\/:*?"<>|\u0000-\u001f\u007f]+/g, '_')
      .replace(/^[\s._]+|[\s.]+$/g, ''),
  ).slice(0, 80).join('');
  return base ? `${base}-transcript.${extension}` : `transcript.${extension}`;
}
