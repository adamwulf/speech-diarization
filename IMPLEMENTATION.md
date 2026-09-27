# Speech recognition and diarization demo

## Success criteria

- Match the speech-ai-demo recognition page colors, typography, panels, buttons, mic meter, and model progress UI; no TTS.
- Record microphone audio or select a browser-decodable audio file, and transcribe in a Web Worker with Whisper tiny.en.
- Optional diarization entirely in the browser, including consistent speaker identities across model windows.
- Checkbox disables speaker detection for new work and hides speaker labels in existing results and exports; re-enabling on a transcript without speaker data runs diarization on retained audio.
- Rename detected speakers, reflected in every matching transcript turn and both exports.
- Export valid timestamped Markdown and WebVTT; preserve text and speaker names safely.
- Model/network/mic/decoding failures give actionable errors and permit retry. No audio uploads.
- Automated meaningful unit tests, production build, browser smoke test with real model inference if environment permits, and two independent reviews.

## Shared module contract

Use vanilla JavaScript modules with Vite. Main thread owns microphone/file decoding to mono 16 kHz PCM and audio playback. Worker owns all model downloads/inference. One active task at a time.

`src/inference-worker.js` incoming messages:

- `{type: 'transcribe', id, audio: Float32Array, speakers: boolean}`
- `{type: 'diarize', id, audio: Float32Array, segments: TranscriptSegment[]}` for an existing transcript

Outgoing:

- `{type: 'progress', id, stage: 'asr-load'|'asr'|'diarization-load'|'diarization', message, progress?: number}` where progress is 0..1, omitted when indeterminate.
- `{type: 'result', id, segments: TranscriptSegment[], diarized: boolean, warning?: string}`
- `{type: 'error', id, message}`

`TranscriptSegment`: `{start: number, end: number, text: string, speaker: string|null}`. Times are seconds, finite, nonnegative, end > start. Speaker IDs are stable strings within one source audio. Text contains words joined naturally. No speaker names stored in worker; names belong to UI map.

Main thread keeps the decoded PCM for re-diarization, so post copies (do not transfer/detach its sole buffer). Results replace the current recording/file; this is explained in UI. Unrecognized speakers stay null, not silently assigned to the first speaker. On diarization failure after ASR, return transcript with `diarized:false` and a warning so text is retained. ASR uses word timestamps for speaker assignment; public result may group adjacent words into useful subtitle cues.

## Ownership

- Inference worker: `src/inference-worker.js`, supporting inference/alignment modules, `tests/inference*.test.js`, `tests/alignment*.test.js`, `INFERENCE.md`. May request dependency/config changes from manager.
- UI worker: `index.html`, `src/main.js`, `src/style.css`, `src/audio.js`, `src/exports.js`, `tests/exports*.test.js`, `README.md`. Follow contract above. Do not change package files.
- Manager: package/build config, lockfile, integration/browser tests, review coordination.

Reference page supplied by speech-helper at `/private/tmp/claude-501/-Users-adamwulf-Developer-html-speech-ai-demo--ittybitty-agents-speech-helper-repo/2df01e01-ceb6-4c45-9b6c-2ea7c8a4d4dd/scratchpad/share/speech-ai-demo-index.html`.

Candidate diarization pipeline is `diarization-js@0.1.0` (`DiarizationPipeline.create`, then `pipeline.run(audio, 16000, {onProgress})` returning `{result,metrics}`), with segmentation, speaker embeddings, and clustering. Inspect installed source/types, do not trust newer model-card API examples. Artifact repo: https://huggingface.co/briox/diarization-js-community-1 . Keep engine on WASM initially for predictable compatibility; no requirement for GPU acceleration.
