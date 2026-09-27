# Inference worker

`src/inference-worker.js` runs all model downloads and inference in a module Web Worker.
The message contract is in `IMPLEMENTATION.md`. This file records the additions to the
contract, the model choices, and the runtime facts that were verified.

## Files

| File | Purpose |
|---|---|
| `src/inference-worker.js` | Loads the models, handles `transcribe`, `diarize`, and `preload` messages, one request at a time. |
| `src/asr-models.js` | The Whisper models that the user can select (repository, revision, download bytes). The page uses it too. |
| `src/inference-tasks.js` | Task flow with injected model calls, silence check, progress helpers, the one-model slot for Whisper, the request queue order. |
| `src/alignment.js` | Word timestamp cleanup, speaker assignment, cue grouping. |
| `tests/inference-tasks.test.js`, `tests/alignment.test.js`, `tests/asr-models.test.js` | Node tests with fake models. |

Create the worker with
`new Worker(new URL('./inference-worker.js', import.meta.url), { type: 'module' })`.
`vite.config.js` must set `worker: { format: 'es' }`.

## Contract additions

- A `transcribe` request can have `asrModel: 'tiny.en' | 'base.en' | 'small.en'`.
  Without it, the worker uses `tiny.en`. An unknown value gives `{type: 'error'}` when
  Whisper runs (silent or too-short audio returns its result before that).
  `diarize` requests ignore it.
- Each result segment also has `words: [{start, end, text}]`. Word `text` keeps one
  leading space when Whisper put a space before the word. Some words attach with no
  space (for example `-known` after `well`). Segment `text` is the word texts joined
  with no separator, then trimmed.
- The UI keeps the segments with `words` and sends them back unchanged in `diarize`.
  Then speakers are aligned at word boundaries. A segment without `words` counts as
  one word.
- Speaker IDs are `SPEAKER_1`, `SPEAKER_2`, ... in order of first appearance in the
  transcript. A word that is not near any speaker turn has speaker `null`.
- `segments: []` with a `warning` means: empty, too short (< 0.1 s), silent, or no speech.
- `diarized: true` with a warning means that diarization ran but found no speaker.
- A `transcribe` request with `speakers: true` returns the transcript with
  `diarized: false` and a warning if diarization fails.
  A `diarize` request that fails sends `{type: 'error'}`; the UI keeps its transcript.
- Progress: `asr-load`, `asr`, `diarization-load`, and `diarization` send `progress`
  0..1 (a few messages, such as "Preparing the speaker detection models", have none). For `asr` it is the fraction of
  30 s windows that are done (see `asrWindowCount`), counted with a `streamer`
  whose `end()` Transformers.js calls once per window. While the audio features
  for all windows are computed, before the first window runs (the streamer's
  first `put()`), `asr` reports "Preparing transcription" with no `progress`.
- Model loads are kept for the life of the worker. A failed load is tried again on the
  next request. The worker keeps one Whisper model: when a request selects a different
  one, the worker disposes the old model, then loads the new one. Dispose frees memory
  only; the model files stay in the `transformers-cache` Cache API store, and nothing
  in the app deletes from it.
- `preload` requests load a model with no audio. The page sends one for the selected
  Whisper model and one for diarization at startup, and one for a new Whisper
  selection after it stays selected for 1 s (arrow keys on a closed select can fire
  `change` for each model they pass).
- Queue order (`enqueueRequest` in `src/inference-tasks.js`): one request runs at a
  time, preloads included, so the Whisper slot never changes models during a task.
  A task goes after the other waiting tasks but before the waiting preloads. A task
  still waits for the request that is running, even a preload of a model that it does
  not use (for example the diarization download when `speakers` is false, if that
  download started first). A preload replaces a waiting preload of the same model, and
  the worker answers the replaced one with an empty result, so only the newest
  selection loads.
- Each load also sends `{type: 'model', ...}` messages (no `id`) for the model lines on
  the page: `loading` with `progress`, `loading` without `progress` while the
  diarization sessions are created, then `ready` or `failed`. A preload also sends
  `ready` when its model was already loaded. Progress comes from the byte reads, so a
  load from the browser cache also shows `loading` with `progress`.

## Models

| Use | Source | Download |
|---|---|---|
| ASR `tiny.en` (default) | `onnx-community/whisper-tiny.en_timestamped` at revision `aeaa1376…` | 43.5 MB, Transformers.js browser cache |
| ASR `base.en` | `onnx-community/whisper-base.en_timestamped` at revision `fa239a41…` | 79.6 MB, Transformers.js browser cache |
| ASR `small.en` | `onnx-community/whisper-small.en_timestamped` at revision `80853938…` | 251.7 MB, Transformers.js browser cache |
| Diarization | `diarization-js@0.1.0` with `briox/diarization-js-community-1` at revision `7f02c43a…` | 33.5 MB, Cache API `diarization-js-community-1` |

- All ASR models load with `dtype: 'q8'` and `device: 'wasm'`. The download is
  `encoder_model_quantized.onnx`, `decoder_model_merged_quantized.onnx`, and the config
  and tokenizer files. Each `_timestamped` export has `alignment_heads` in its
  `generation_config.json`.
- The `_timestamped` export has the cross-attention outputs that word timestamps need.
  Do not pass `language` or `task`: Transformers.js 3.8.1 throws for English-only models.
- The browser checks below used only `tiny.en`. `base.en` and `small.en` use the same
  code path, but they were not run in a browser.
- The ASR call is `{return_timestamps: 'word', chunk_length_s: 30, stride_length_s: 5}`
  (`ASR_CHUNK_OPTIONS`), plus the progress `streamer`.
  Without `chunk_length_s`, audio after 30 s is lost.
- diarization-js is a port of `pyannote/speaker-diarization-community-1`:
  10 s window segmentation, WeSpeaker ResNet34 embeddings, then AHC and VBx clustering
  over all windows. Its speaker labels are global for the recording, not local to a window.
- The artifact URLs are pinned to a commit. `ensureArtifacts` is not used because it has
  no persistent browser cache; the worker fetches the files with progress and stores them
  with the Cache API.
- Licenses: diarization-js code (npm field Apache-2.0, README MIT); segmentation-3.0 MIT;
  WeSpeaker ResNet34 embedding CC-BY-4.0 (attribute pyannote/wespeaker-voxceleb-resnet34-LM);
  PLDA parameters MIT per the artifact repository but CC-BY-4.0 per the diarization-js
  README (pyannote/speaker-diarization-community-1). Attribute both to be safe.

## ONNX Runtime

- Transformers.js and diarization-js use the same `onnxruntime-web` 1.22.0 instance
  (the package overrides dedupe it), so there is one WASM runtime.
- At import, Transformers.js 3.8.1 sets `ort.env.wasm.wasmPaths` to jsDelivr
  `@huggingface/transformers@3.8.1/dist/`. Those files come from the 1.22.0-dev build
  (different sha256 from 1.22.0). The worker replaces this with
  `{wasm: <local ort-wasm-simd-threaded.jsep.wasm URL>}`. ORT then uses its embedded
  1.22.0 glue code with the matching local WASM, and there is no CDN request for runtime
  files. The URL comes from a relative `?url` import because the `onnxruntime-web`
  exports map does not expose `dist/`.
- Both sessions use the WASM execution provider. Without cross-origin isolation, ORT
  uses one thread.
- `segmentationBatchSize: 8` keeps WASM memory low. `embeddingBatchSize: 1`:
  diarization-js zero-pads each embedding batch to its longest crop, and the padding
  changes the embeddings. On the two-speaker pyannote sample, batch 8 merged both
  speakers into one. In that test, batch 1 (no padding) found both speakers and was also
  faster than the larger batches.
- Vite prints "Module node:fs/promises has been externalized" for diarization-js
  `artifacts.js`. Those imports are only in its Node code path; the warning is harmless.

## Alignment

1. Word timestamps are made finite, clamped to `[0, duration]`, given a minimum length of
   0.02 s, and ordered. A word end that overlaps the next word start is trimmed.
2. Whisper non-speech tags such as `[BLANK_AUDIO]` are removed. If no words remain, the
   result is "no speech".
3. Each word takes the speaker with the most total overlap. A word in a gap takes the
   nearest turn within 0.5 s, else `null`.
4. Words are grouped into cues. A new cue starts at a speaker change, a pause > 1 s,
   sentence-final punctuation, more than 7 s, or more than 84 characters.

## Browser validation

These are checks on two short fixtures. They show that the pipeline works end to end.
They do not measure general accuracy or speed.

Conditions: headless Chrome through `websnap` (a fresh profile for each run, so the
first case downloads all models from Hugging Face), WASM execution provider, one thread
(no cross-origin isolation), real models, no mocks. Two builds were tested:

1. Vite dev server.
2. Production: `vite build` with `base: './'` and `worker: { format: 'es' }`, served by
   `vite preview`.

Fixtures:

- `jfk.wav` (11 s, one speaker).
- pyannote `tutorials/assets/sample.wav` (30 s, 16 kHz, two speakers that alternate
  across 10 s windows) with its reference RTTM (speakers `speaker90` and `speaker91`).

Results, the same in both builds:

- JFK: the expected sentence, one speaker, valid timestamps, all progress values in 0..1.
- `sample.wav`: two speakers. From 9 s on, each cue has the reference speaker, and a
  speaker keeps its ID across windows (for example `speaker90` is `SPEAKER_1` at 9 s,
  11 s, 18 s, and 28 s). Exceptions: short words at turn edges, and the overlapping
  "Hello" words in the first 9 s.
- Transcribe without speakers, then `diarize` with the returned segments: the same
  result as transcribe with speakers (dev server).
- 3 s of zeros: "no speech" without a model run. Low noise: Whisper `[BLANK_AUDIO]` is
  removed, so the result is "no speech" (dev server).
- The worker loads only the local hashed `ort-wasm-simd-threaded.jsep` WASM, with a URL
  relative to the worker. The jsDelivr default from Transformers.js is replaced before
  any session starts.

The harness and fixtures are in `/private/tmp/speech-inference-smoke/` (not committed).

### Long audio: ASR chunk stitching

No natural 60 to 90 s recording with more than one speaker was available, so two inputs
were used (production build, same conditions as above, engine commit `746607f`):

- A: `ted_60_16k.wav` from the Transformers.js docs dataset: natural speech, 60.0 s,
  one speaker.
- B: a concatenation of pyannote `sample.wav` (0 to 30 s, two speakers) and
  `ted_60_16k.wav` (30 to 90 s), 90.0 s. The concatenation point is at 30 s.

With `chunk_length_s: 30, stride_length_s: 5`, the ASR windows start every 20 s, so the
stitch regions are 20 to 30 s and 40 to 50 s (A), and 20 to 30, 40 to 50, and 60 to 70 s (B).
All stitch regions are in natural speech.

Checks and results:

| Check | A (60 s) | B (90 s) |
|---|---|---|
| Last word | " up" ends at 60.00 s | " up," ends at 90.00 s |
| Words | 187 | 267 |
| Each word: finite, `0 <= start < end <= duration`; starts never decrease; segment text = words concatenated and trimmed | no problems | no problems |
| Same 3-word group repeated at once | none | only "hello hello ..." at 0 to 9 s, which is in the audio and also in the unstitched 30 s run of `sample.wav` |
| Stitched words vs a baseline clip of 20 s (one ASR window, no stitching) around each stitch region | 40-50 s: equal. 20-30 s: 4 word differences (civil/sivil, wanna/want to, one "and"), no word lost or repeated at the seam | 40-50 s: equal. 60-70 s: one extra "then" (see below). 20-30 s: the baseline clip is not usable (Whisper skipped the phone speech in it) |
| Speakers | `SPEAKER_1` for all words except 2 null ("you know," at 13 s, no speaker turn nearby) | 3 speakers: `SPEAKER_1` and `SPEAKER_2` in 0 to 30 s; `SPEAKER_3` for all 188 words from 30 to 90 s (in every 10 s window) |

Seam diagnosis for B: each 30 s ASR window of B was also transcribed alone (no stitching).

- 60 to 70 s: the stitched words are the same as in the 40 to 70 s window alone, including
  "but then then" at 60.14 s and "I would, I would" at 57.7 s. The 60 to 90 s window
  alone has "But then that's". So Whisper made the extra words in the context of that
  window; the stitching did not add them.
- 20 to 30 s: the stitched words are the words of the 20 to 50 s window ("So I'm like,
  oh, I don't hear ..."). The 0 to 30 s window alone has "So I don't hear ...". The stitch
  removed the partial "So" at 20.00 s, which starts the 20 to 50 s window.

Conclusion: in these runs, no word is lost at a seam and no word is repeated by the
stitch. The words near a seam can be different from the words in a single-window run,
because Whisper output depends on the window context.

Known limit: after long silence, Whisper can put the first word's start too early
(for example 0.0 s when speech starts at 6.7 s). The timestamps stay valid.
