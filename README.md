# Speech Recognition & Diarization

A browser demo that transcribes speech and labels who spoke when — entirely on
your device. Record from the microphone or pick an audio file; Whisper tiny.en
transcribes it, and optional speaker detection (pyannote + WeSpeaker via
diarization-js) adds speaker labels. Rename speakers, then export the transcript
as Markdown or WebVTT. Audio is never uploaded; only model files are downloaded.

## Run it

```sh
npm ci --ignore-scripts
npm run dev
```

Then open the printed local address (for example `http://127.0.0.1:5173/`).

`--ignore-scripts` skips the post-install steps of `sharp` and
`onnxruntime-node`, which are Node-only dependencies of Transformers.js. The
browser app does not use these native add-ons, and their install scripts can
fail on some hosts (for example, with a Homebrew `libvips`).

Other scripts:

| Command | What it does |
| --- | --- |
| `npm test` | Runs the unit tests (`node --test tests/*.test.js`). |
| `npm run build` | Builds the static site into `dist/`. |
| `npm run preview` | Serves the production build locally. |

Microphone recording needs a secure page: `http://localhost`, `http://127.0.0.1`,
or HTTPS. Transcribing a file works on any origin.

## Using the demo

- **Start recording**, speak, then **Stop recording** to transcribe.
  **Transcribe file…** accepts any audio format the browser can decode
  (WAV, MP3, M4A, WebM, and so on).
- **Each new recording or file replaces** the current audio, transcript, and
  speaker names. If the new audio fails to decode or transcribe, the previous
  transcript stays on screen.
- **Detect speakers** (on by default):
  - Off: new audio is transcribed without speaker detection, and speaker names
    are hidden in the transcript and in both exports.
  - On again: names come back. If the current transcript has no speaker data
    yet, speaker detection runs on the retained audio. The transcript text is
    not transcribed again.
  - If a new recording or file failed and **Retry transcription** is showing,
    the checkbox only sets what the retry does. It does not start work on the
    older transcript.
  - If speaker detection runs but cannot identify any speaker, the page says
    so and shows the transcript without speaker names.
- **Speakers** get default names (“Speaker 1”, “Speaker 2”, …) in order of
  first appearance. Type a new name to update every matching turn and both
  exports. Clear the field to use the default name again. Segments that could
  not be given to a speaker show as “Unknown speaker”.
- **Export Markdown** and **Export WebVTT** download the transcript as shown,
  including the current speaker names (or no names, if they are hidden).
- **Errors** tell you what to do next. Model download or inference failures
  show a **Retry transcription** or **Retry speaker detection** button.
  Microphone and decoding errors tell you how to fix the problem.
- Buttons that would start other work are disabled while a task runs. Only one
  task runs at a time.

## Models

| Model | Purpose | Download |
| --- | --- | --- |
| Whisper tiny.en (timestamped, q8) | Speech-to-text | ~41 MB |
| pyannote segmentation-3.0, WeSpeaker ResNet34 embeddings, community-1 PLDA | Speaker diarization | ~34 MB |

Nothing downloads when the page opens. The Whisper model downloads on the
first transcription; the speaker models download the first time speaker
detection runs. After that, the browser cache supplies them.

## How it works

- `src/main.js`: page state and controls, the worker client, and
  transcript rendering. It builds all DOM with `textContent`, never HTML
  strings, so transcript text and speaker names cannot inject markup.
- `src/audio.js`: microphone capture (`MediaRecorder`, with every track and
  audio context released on stop, error, or page hide), decoding to 16 kHz mono
  PCM with `OfflineAudioContext`, and the native canvas level meter.
- `src/exports.js`: pure functions for turn grouping, speaker names, and
  Markdown and WebVTT output. These are tested in `tests/exports.test.js`.
- `src/inference-worker.js`: a Web Worker that owns all model downloads and
  inference. The main thread keeps the decoded audio and posts a copy for each
  task. `IMPLEMENTATION.md` defines the message contract.

## Export formats

Markdown: one paragraph per speaker turn. Adjacent segments from the same
speaker are merged. Text is escaped so that it shows literally.

```md
# Transcript

- Source: team-sync.m4a
- Duration: 02:05

**[00:00 – 00:04] Alice:** Hello there. How are you?

**[00:04 – 00:06] Speaker 2:** Fine, thanks.
```

WebVTT: one cue per segment, with speaker names as voice spans. `&`, `<`, and
`>` are escaped in text and names.

```vtt
WEBVTT

1
00:00:00.000 --> 00:00:02.500
<v Alice>Hello there.</v>
```

## Credits and licenses

- [Whisper](https://github.com/openai/whisper) by OpenAI, via the
  [onnx-community/whisper-tiny.en_timestamped](https://huggingface.co/onnx-community/whisper-tiny.en_timestamped)
  conversion and [Transformers.js](https://github.com/huggingface/transformers.js).
- [diarization-js](https://github.com/briox/diarization-js), a port of
  [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1)
  by the [pyannote.audio](https://github.com/pyannote/pyannote-audio) team.
  The [ONNX artifacts](https://huggingface.co/briox/diarization-js-community-1)
  keep their upstream licenses:
  - `pyannote/segmentation-3.0`: MIT.
  - WeSpeaker ResNet34 speaker embeddings
    ([pyannote/wespeaker-voxceleb-resnet34-LM](https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM),
    from the [WeSpeaker](https://github.com/wenet-e2e/wespeaker) project):
    [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
  - community-1 PLDA parameters: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- Inference runs with ONNX Runtime Web (WASM).
