# Inference worker

`src/inference-worker.js` runs all model downloads and inference in a module Web Worker.
The message contract is in `IMPLEMENTATION.md`. This file records the additions to the
contract, the model choices, and the runtime facts that were verified.

## Files

| File | Purpose |
|---|---|
| `src/inference-worker.js` | Loads the models, handles `transcribe` and `diarize` messages, one task at a time. |
| `src/inference-tasks.js` | Task flow with injected model calls, silence check, progress helpers. |
| `src/alignment.js` | Word timestamp cleanup, speaker assignment, cue grouping. |
| `tests/inference-tasks.test.js`, `tests/alignment.test.js` | Node tests with fake models. |

Create the worker with
`new Worker(new URL('./inference-worker.js', import.meta.url), { type: 'module' })`.
`vite.config.js` must set `worker: { format: 'es' }`.

## Contract additions

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
- Progress: `asr-load`, `diarization-load`, and `diarization` send `progress` 0..1.
  `asr` has no `progress` (indeterminate).
- Model loads are kept for the life of the worker. A failed load is tried again on the
  next request.

## Models

| Use | Source | Download |
|---|---|---|
| ASR | `onnx-community/whisper-tiny.en_timestamped` at revision `aeaa1376…`, `dtype: 'q8'`, `device: 'wasm'` | 43.5 MB, Transformers.js browser cache |
| Diarization | `diarization-js@0.1.0` with `briox/diarization-js-community-1` at revision `7f02c43a…` | 33.5 MB, Cache API `diarization-js-community-1` |

- The `_timestamped` export has the cross-attention outputs that word timestamps need.
  Do not pass `language` or `task`: Transformers.js 3.8.1 throws for English-only models.
- The ASR call is `{return_timestamps: 'word', chunk_length_s: 30, stride_length_s: 5}`.
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

Known limit: after long silence, Whisper can put the first word's start too early
(for example 0.0 s when speech starts at 6.7 s). The timestamps stay valid.
