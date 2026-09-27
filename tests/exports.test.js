import { test } from 'node:test';
import assert from 'node:assert/strict';

import {
  UNKNOWN_SPEAKER_LABEL,
  MAX_SPEAKER_NAME_LENGTH,
  normalizeSegments,
  speakerOrder,
  cleanSpeakerName,
  createSpeakerNamer,
  groupTurns,
  formatClock,
  formatVttTimestamp,
  escapeMarkdown,
  toWebVTT,
  toMarkdown,
  exportFileName,
} from '../src/exports.js';

// A small two-speaker conversation with one unattributed segment. The worker
// also sends word timings; exports must ignore them.
const CONVERSATION = [
  {
    start: 0,
    end: 2.5,
    text: 'Hello there.',
    speaker: 'spk_a',
    words: [{ start: 0, end: 1, text: 'Hello' }, { start: 1.1, end: 2.5, text: 'there.' }],
  },
  { start: 2.5, end: 4, text: 'How are you?', speaker: 'spk_a' },
  { start: 4.2, end: 6, text: 'Fine, thanks.', speaker: 'spk_b' },
  { start: 6.5, end: 7, text: 'Mm-hmm.', speaker: null },
  { start: 7.1, end: 9.75, text: 'Great to hear.', speaker: 'spk_a' },
];

// --- A minimal WebVTT validator (the parts of the spec our exporter emits). ---

const VTT_TIMING = /^(\d{2,}):([0-5]\d):([0-5]\d)\.(\d{3}) --> (\d{2,}):([0-5]\d):([0-5]\d)\.(\d{3})$/;

function vttSeconds(h, m, s, ms) {
  return Number(h) * 3600 + Number(m) * 60 + Number(s) + Number(ms) / 1000;
}

function parseVtt(source) {
  assert.ok(source.endsWith('\n'), 'file ends with a newline');
  const blocks = source.trimEnd().split('\n\n');
  assert.equal(blocks.shift(), 'WEBVTT', 'first block is the bare WEBVTT header');
  let previousStart = -Infinity;
  return blocks.map((block) => {
    const lines = block.split('\n');
    assert.equal(lines.length, 3, `cue has id, timing, and one payload line: ${JSON.stringify(block)}`);
    const [id, timing, payload] = lines;
    assert.ok(!id.includes('-->'), 'cue id must not contain -->');
    const match = VTT_TIMING.exec(timing);
    assert.ok(match, `valid timing line: ${timing}`);
    const start = vttSeconds(...match.slice(1, 5));
    const end = vttSeconds(...match.slice(5, 9));
    assert.ok(end > start, `cue end after start: ${timing}`);
    assert.ok(start >= previousStart, 'cues are ordered by start time');
    previousStart = start;
    assert.ok(payload.trim().length > 0, 'payload is not blank');
    assert.ok(!payload.includes('-->'), 'payload must not contain -->');
    // Outside of our own <v ...> / </v> tags, no raw < or > may remain.
    const inner = payload.replace(/^<v [^<>]+>/, '').replace(/<\/v>$/, '');
    assert.ok(!/[<>]/.test(inner), `payload text is escaped: ${payload}`);
    const voice = /^<v ([^<>]+)>/.exec(payload)?.[1] ?? null;
    return { id, start, end, payload, voice, text: inner };
  });
}

function decodeVtt(text) {
  return text.replace(/&lt;/g, '<').replace(/&gt;/g, '>').replace(/&amp;/g, '&');
}

// --- Markdown helpers: CommonMark backslash escapes apply to ASCII punctuation. ---

function unescapeMarkdown(text) {
  return text.replace(/\\([!-/:-@[-`{-~])/g, '$1');
}

const MARKDOWN_SPECIALS = new Set(['\\', '`', '*', '_', '[', ']', '<', '>', '|', '~', '$']);

function assertNoUnescapedSpecials(text) {
  for (let i = 0; i < text.length; i++) {
    if (text[i] === '\\') { i++; continue; }
    assert.ok(!MARKDOWN_SPECIALS.has(text[i]), `unescaped ${text[i]} at ${i} in ${JSON.stringify(text)}`);
  }
}

// ---------------------------------------------------------------------------

test('normalizeSegments drops malformed segments, cleans text, and sorts', () => {
  const input = [
    { start: 5, end: 6, text: '  second\n\tline  ', speaker: 'b' },
    { start: 1, end: 2, text: 'first', speaker: '' },
    { start: 3, end: 3, text: 'zero length', speaker: 'a' },
    { start: -1, end: 2, text: 'negative', speaker: 'a' },
    { start: Number.NaN, end: 2, text: 'nan', speaker: 'a' },
    { start: 2, end: Infinity, text: 'infinite', speaker: 'a' },
    { start: 7, end: 8, text: '   ', speaker: 'a' },
    null,
    'not a segment',
    { start: 5, end: 5.5, text: 'same start keeps order', speaker: 'c' },
  ];
  const snapshot = structuredClone(input);
  assert.deepEqual(normalizeSegments(input), [
    { start: 1, end: 2, text: 'first', speaker: null },
    { start: 5, end: 6, text: 'second line', speaker: 'b' },
    { start: 5, end: 5.5, text: 'same start keeps order', speaker: 'c' },
  ]);
  assert.deepEqual(input, snapshot, 'input is not mutated');
  assert.deepEqual(normalizeSegments(undefined), []);
});

test('speakers get default names by first appearance, custom names win, null is unknown', () => {
  assert.deepEqual(speakerOrder(CONVERSATION), ['spk_a', 'spk_b']);

  const defaults = createSpeakerNamer(CONVERSATION);
  assert.equal(defaults('spk_a'), 'Speaker 1');
  assert.equal(defaults('spk_b'), 'Speaker 2');
  assert.equal(defaults(null), UNKNOWN_SPEAKER_LABEL);
  assert.equal(defaults('not_in_transcript'), UNKNOWN_SPEAKER_LABEL);

  const named = createSpeakerNamer(CONVERSATION, new Map([['spk_b', '  Dr.   Bob '], ['spk_a', '   ']]));
  assert.equal(named('spk_a'), 'Speaker 1', 'a blank custom name falls back to the default');
  assert.equal(named('spk_b'), 'Dr. Bob');

  const hidden = createSpeakerNamer(CONVERSATION, new Map([['spk_a', 'Alice']]), false);
  assert.equal(hidden('spk_a'), null);
  assert.equal(hidden(null), null);
});

test('cleanSpeakerName keeps names single-line and caps length by code point', () => {
  assert.equal(cleanSpeakerName('  Ada\nLovelace\t '), 'Ada Lovelace');
  assert.equal(cleanSpeakerName(undefined), '');
  const long = '😀'.repeat(MAX_SPEAKER_NAME_LENGTH + 5);
  const capped = cleanSpeakerName(long);
  assert.equal(Array.from(capped).length, MAX_SPEAKER_NAME_LENGTH);
  assert.ok(capped.isWellFormed(), 'no split surrogate pairs');
});

test('groupTurns merges consecutive same-speaker segments only when speakers are shown', () => {
  const segs = normalizeSegments(CONVERSATION);
  assert.deepEqual(groupTurns(segs, { showSpeakers: true }), [
    { start: 0, end: 4, text: 'Hello there. How are you?', speaker: 'spk_a' },
    { start: 4.2, end: 6, text: 'Fine, thanks.', speaker: 'spk_b' },
    { start: 6.5, end: 7, text: 'Mm-hmm.', speaker: null },
    { start: 7.1, end: 9.75, text: 'Great to hear.', speaker: 'spk_a' },
  ]);
  const hidden = groupTurns(segs, { showSpeakers: false });
  assert.equal(hidden.length, segs.length, 'one turn per segment when hidden');
  assert.ok(hidden.every((turn) => turn.speaker === null), 'no speaker ids leak when hidden');
});

test('formatClock and formatVttTimestamp handle rounding, carries, and hours', () => {
  assert.equal(formatClock(0), '00:00');
  assert.equal(formatClock(65.9), '01:05');
  assert.equal(formatClock(3599.99), '59:59');
  assert.equal(formatClock(3600), '1:00:00');
  assert.equal(formatClock(-3), '00:00');
  assert.equal(formatClock(Number.NaN), '00:00');

  assert.equal(formatVttTimestamp(0), '00:00:00.000');
  assert.equal(formatVttTimestamp(1.2345), '00:00:01.235');
  assert.equal(formatVttTimestamp(59.9996), '00:01:00.000', 'rounding carries into minutes');
  assert.equal(formatVttTimestamp(3723.456), '01:02:03.456');
  assert.equal(formatVttTimestamp(360000), '100:00:00.000', 'hours may exceed two digits');
});

test('toWebVTT emits one valid cue per segment with renamed voice spans', () => {
  const vtt = toWebVTT(CONVERSATION, { speakerNames: new Map([['spk_a', 'Alice']]) });
  const cues = parseVtt(vtt);
  assert.equal(cues.length, CONVERSATION.length);
  assert.deepEqual(cues.map((cue) => cue.id), ['1', '2', '3', '4', '5']);
  assert.deepEqual(cues.map((cue) => cue.voice), ['Alice', 'Alice', 'Speaker 2', null, 'Alice'],
    'rename applies to every matching cue; unattributed cues get no voice');
  assert.deepEqual(cues.map((cue) => cue.text), CONVERSATION.map((seg) => seg.text));
  assert.equal(cues[4].start, 7.1);
  assert.equal(cues[4].end, 9.75);
});

test('toWebVTT escapes markup and cue terminators in text and speaker names', () => {
  const segments = [
    { start: 0, end: 1, text: 'a --> b <b>bold</b> & more\n\nnext', speaker: 's1' },
    { start: 1, end: 1.0002, text: 'tiny', speaker: 's1' },
  ];
  const name = 'Tom <T> & "Jerry"\nJr.';
  const cues = parseVtt(toWebVTT(segments, { speakerNames: new Map([['s1', name]]) }));
  assert.equal(decodeVtt(cues[0].text), 'a --> b <b>bold</b> & more next');
  assert.equal(decodeVtt(cues[0].voice), 'Tom <T> & "Jerry" Jr.');
  assert.ok(cues[1].end > cues[1].start, 'sub-millisecond cue still gets a positive duration');
});

test('toWebVTT omits all speaker information when speakers are hidden', () => {
  const vtt = toWebVTT(CONVERSATION, { speakerNames: new Map([['spk_a', 'Alice']]), showSpeakers: false });
  const cues = parseVtt(vtt);
  assert.equal(cues.length, CONVERSATION.length);
  assert.ok(cues.every((cue) => cue.voice === null));
  assert.ok(!vtt.includes('<v'));
  assert.ok(!vtt.includes('Alice'));
  assert.equal(toWebVTT([]), 'WEBVTT\n', 'an empty transcript is still a valid file');
});

test('toMarkdown writes timestamped turns and applies renames to every matching turn', () => {
  const md = toMarkdown(CONVERSATION, {
    speakerNames: new Map([['spk_a', 'Alice']]),
    sourceName: 'team_sync.m4a',
    duration: 125,
  });
  assert.equal(md, [
    '# Transcript',
    '',
    '- Source: team\\_sync.m4a',
    '- Duration: 02:05',
    '',
    '**[00:00 – 00:04] Alice:** Hello there. How are you?',
    '',
    '**[00:04 – 00:06] Speaker 2:** Fine, thanks.',
    '',
    `**[00:06 – 00:07] ${UNKNOWN_SPEAKER_LABEL}:** Mm-hmm.`,
    '',
    '**[00:07 – 00:09] Alice:** Great to hear.',
    '',
  ].join('\n'));
});

test('toMarkdown hides every speaker name when speakers are hidden', () => {
  const md = toMarkdown(CONVERSATION, { speakerNames: new Map([['spk_a', 'Alice']]), showSpeakers: false });
  for (const name of ['Alice', 'Speaker', UNKNOWN_SPEAKER_LABEL]) {
    assert.ok(!md.includes(name), `no "${name}" in ${md}`);
  }
  const turnLines = md.split('\n').filter((line) => line.startsWith('**['));
  assert.equal(turnLines.length, CONVERSATION.length, 'one timestamped line per segment');
  assert.equal(turnLines[0], '**[00:00 – 00:02]** Hello there.');
});

test('escapeMarkdown preserves text exactly and leaves no live Markdown syntax', () => {
  const samples = [
    'Use *stars*, _underscores_, `code` and [links](http://x.test)',
    '<script>alert(1)</script> | a ~~strike~~ | costs $5',
    'C:\\path\\to\\file and a trailing backslash\\',
    'Entities &amp; &#169; &copy; stay literal; a lone & is fine',
    '# not a heading\n- not a list\n\n> not a quote',
  ];
  for (const sample of samples) {
    const escaped = escapeMarkdown(sample);
    assert.ok(!escaped.includes('\n'), 'always a single line');
    assertNoUnescapedSpecials(escaped);
    assert.ok(!/(^|[^\\])&#?[a-z0-9]+;/i.test(escaped), `entities are escaped in ${escaped}`);
    assert.equal(unescapeMarkdown(escaped), sample.replace(/\s+/g, ' ').trim(), 'round-trips to the original text');
  }
  assert.equal(escapeMarkdown('Tom & Jerry'), 'Tom & Jerry', 'plain ampersands stay readable');
});

test('toMarkdown escapes hostile speaker names and text', () => {
  const md = toMarkdown(
    [{ start: 0, end: 1, text: '**bold** <img src=x onerror=alert(1)>', speaker: 'x' }],
    { speakerNames: new Map([['x', '[Eve](javascript:alert(1))']]) },
  );
  const line = md.split('\n').find((l) => l.startsWith('**['));
  assert.equal(line, '**[00:00 – 00:01] \\[Eve\\](javascript:alert(1)):** \\*\\*bold\\*\\* \\<img src=x onerror=alert(1)\\>');
  assert.match(toMarkdown([]), /_No speech detected\._/);
});

test('exportFileName strips the extension and unsafe characters', () => {
  assert.equal(exportFileName('Team sync.m4a', 'md'), 'Team sync-transcript.md');
  assert.equal(exportFileName('recording-2026-09-27-14-54-11', 'vtt'), 'recording-2026-09-27-14-54-11-transcript.vtt');
  assert.equal(exportFileName('a/b\\c:d*e?"f"<g>|h.wav', 'vtt'), 'a_b_c_d_e_f_g_h-transcript.vtt');
  assert.equal(exportFileName('...', 'md'), 'transcript.md');
  assert.equal(exportFileName('', 'md'), 'transcript.md');
  assert.equal(exportFileName(undefined, 'vtt'), 'transcript.vtt');
});
