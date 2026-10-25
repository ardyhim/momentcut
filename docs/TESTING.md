# Validation and test notes

## What was actually run

Environment used for validation:

- Windows, Python 3.12.14
- FFmpeg/ffprobe 8.0.1
- faster-whisper 1.2.1, ctranslate2 4.8.4, PyAV 16.0.1, CPU int8
- pytest 9.1.1

Automated suite:

```bash
python -m pytest tests -q
```

Result at the time of writing: **121 passed** (about 85 s, including synthetic
video generation and real FFmpeg renders). The count will grow as tests are
added; the exact number is less important than the coverage below.

### Covered by automated tests

| Area | What is asserted |
|---|---|
| Transcript units | sentence/pause splitting, clamping, CJK spacing, empty garbage |
| Overlap merge | duplicate words from overlapping ASR chunks are removed; real repetition survives |
| Subtitle import | SRT/VTT parsing, tags, notes, cue merging, invalid input |
| JSON transcript | units, Whisper words/segments, top-level words, unsupported plain text |
| Blocks | full coverage, overlap, one-giant-unit behavior, line ranges |
| Prompts | only block lines included, schema/JSON shape, duration band, injection guard, prompt bundle files |
| JSON parsing | plain/fenced/prose/bare-array, trailing comma repair, wrappers, 10 MB cap |
| Candidate parsing | aliases, scores, metadata fallback, one bad candidate does not kill the rest |
| Anchor resolution | valid, repaired, conflicting, missing, ambiguous, out-of-block |
| Duration | minimum extension, maximum rejection, padding trim, source-end clamp, short source |
| Selection | score filter, standalone context floor, temporal/text duplicates, max clips, chronological IDs, no padding, rank |
| Pipeline | manual end-to-end, re-import without AI, prompt archival on settings change, source relocation, source mismatch refusal, invalid response report, partial coverage, API mode against a fake endpoint |
| Rendering | original and 1080x1920 vertical, audio stream, duration verification, resume, wrong-duration rejection |
| Provider | success, auth header, response-format fallback, 429/5xx retry, auth failure not retried, empty content, wrapper import |
| ASR | chunk planning, resumable cache with a fake model, missing dependency message, no-audio error |

### Real ASR smoke tests (not mocked)

1. Generated 11.1 s of Windows TTS speech, built an MP4, transcribed with
   `faster-whisper-tiny` int8 on CPU. Result: 3 units, correct sentence
   boundaries, 2.8 s wall time.
2. Generated 105 s of TTS speech, used two 60 s overlapping ASR chunks. Result:
   24 units, no duplicated boundary units, 7.2 s wall time. ASR made a normal
   recognition error ("week" instead of "weak"), which the engine correctly
   treats as transcript text.
3. Imported a hand-written provider answer for the 11.1 s transcript, rendered a
   10.63 s vertical clip, and verified 1080x1920 plus audio with ffprobe.

The real ASR run exposed the PyAV 19 incompatibility documented in
`requirements.txt`; installing PyAV 16 fixed it.

### Real end-to-end mechanics test

The automated pipeline test:

- generates a 50 s synthetic video;
- imports a 10-line transcript fixture;
- analyzes and exports a prompt;
- imports seven provider candidates, including a duplicate, a hallucinated line,
  an overlong span, a below-threshold entry, and a missing title;
- resolves 5, rejects 2, selects 2, and drops duplicates/below-threshold;
- renders original clips and vertical clips;
- verifies durations/audio/dimensions with ffprobe;
- re-renders and confirms both original and vertical outputs are skipped.

This proves the mechanical contract: AI JSON -> validated line IDs -> safe
timestamps -> playable clips.

## Not validated

Be explicit about these before treating the system as production-ready:

- **Real external provider quality.** The API client was tested against a local
  fake HTTP server. No paid provider account was used here. Real providers may
  differ in JSON discipline, rate limits, context limits, and parameter support.
- **Editorial quality.** No human-labelled dataset or blind rating study was
  performed. The prompt's "cold-open", "standalone", and "payoff" tests are
  reasoned heuristics, not measured accuracy. A prompt can still select a
  semantically out-of-context moment.
- **Free Colab execution.** The mechanics were tested on Windows with the same
  libraries. The exact Colab image, GPU availability, RAM, disk, and package
  resolution were not exercised live. The notebook is written for Colab and its
  code cells are syntax-checked, but that is not the same as a live run.
- **GPU ASR.** Only CPU int8 was exercised. CUDA load fallback is implemented
  but not measured on Colab hardware.
- **Multilingual accuracy.** Unit construction and copy-language handling are
  language-general; only English ASR was smoke-tested. Indonesian, Arabic,
  Chinese, code-switching, and heavy accents need their own tests.
- **Long/large video.** The longest real ASR test was 105 s. Multi-hour
  behavior, disk pressure, and rendering time are reasoned, not measured.
- **Visual-only footage.** Not supported by the baseline.
- **YouTube download, social upload, captions, face tracking, speaker
  diarization.** Out of scope for v0.1.
- **Concurrency.** Rendering and API calls are sequential. There is no
  multi-job stress test.
- **Prompt-injection resistance.** The prompt labels the transcript as data;
  a model can still be manipulated by content. The engine only consumes
  structured fields and never executes model text, which limits the blast
  radius, but it cannot make the model obey.
- **Score calibration across blocks.** The rubric is explicit, but scores from
  different chapters/providers are not statistically normalized.

## Quality evaluation that should happen next

A credible evaluation needs:

1. 20-30 long videos across at least podcast, interview, tutorial, and comedy.
2. Human-selected "acceptable" moments per video (start/end + reason).
3. Blind ratings of returned clips on: standalone context, hook, payoff,
   duplication, and overall publish-worthiness.
4. Comparison against a simple baseline (e.g., random/transcript-position
   clips) and against OpenShorts-style fixed-window selection.
5. Inter-rater agreement and failure analysis, not just an average score.

Only then can we claim the approach is better than OpenShorts, rather than
better at enforcing timing and provider independence.

## Running the tests

```bash
pip install -r requirements-dev.txt
python -m pytest tests -q
```

FFmpeg tests generate their own synthetic media and skip when `ffmpeg` or
`ffprobe` is unavailable. No copyrighted or user video is part of the test
suite.

## Interpreting failures

- A failure in `test_resolve.py` usually means boundary validation changed.
  Check whether the change is intentional and update the documented contract.
- A failure in `test_pipeline.py` may leave a run directory under pytest's temp
  directory; inspect `analysis/report.md`.
- A rendering test failing with "Invalid argument" often means an FFmpeg filter
  or codec option is unavailable in the installed FFmpeg.
- If all ASR tests pass but real transcription fails, check the `av` version and
  the model download/cache path first.







