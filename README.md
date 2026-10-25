# MomentCut

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ardyhim/momentcut/blob/main/notebooks/momentcut_colab.ipynb)

Turn long-form spoken video into short clips without trusting an AI model with
the final cut times.

MomentCut transcribes a video (or imports SRT/VTT/JSON), builds sentence-level
transcript lines, exports portable chapter prompts, validates AI answers
against those lines, and renders clips with FFmpeg. It works with any AI
provider through copy/paste, or with any OpenAI-compatible chat endpoint through
an optional API mode.

This is a working v0.1 baseline, not a hosted product. The 121-test suite and
the real FFmpeg/ASR smoke runs cover the mechanics; editorial quality still has
to be evaluated on your own material. The investigation that led here is in
[`docs/FINDINGS.md`](docs/FINDINGS.md).

## Why this design

OpenShorts' useful insight is to treat the transcript as timing ground truth and
use the model for editorial judgment. Its fixed 90-second windows, forced clip
floors, and trust in float timestamps are not copied.

The core invariant:

> The AI chooses meaning. The engine chooses seconds.

The model returns transcript line IDs (`L0001`) plus exact quote anchors. The
engine resolves those to real word timings, expands to complete sentence lines,
checks the configured duration and source bounds, removes duplicates, and only
then calls FFmpeg. A model that returns `L9999` or a start after its end cannot
produce a clip.

## Quick start (manual, no API key)

Requirements: Python 3.10+, FFmpeg/ffprobe on `PATH`. On Linux/Colab, FFmpeg is
usually preinstalled.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install -r requirements.txt

python -m momentcut analyze video.mp4 \
  --run runs/my-video \
  --asr-model small \
  --min-duration 15 \
  --max-duration 90 \
  --max-clips 8
```

`analyze` writes:

```text
runs/my-video/
  run.json
  transcript.json
  analysis/
    blocks.json
    prompts/block_001.md ...
    responses/
```

Paste each prompt into the AI provider of your choice. Save each answer as a
`.json`, `.txt`, or `.md` file in `analysis/responses/` (one answer per block),
then:

```bash
python -m momentcut import runs/my-video
python -m momentcut render runs/my-video --aspect vertical
```

Rendered files go to `runs/my-video/clips/`; validation details are in
`analysis.json`, `analysis/report.md`, and `render_manifest.json`.

If you already have a transcript, skip ASR:

```bash
python -m momentcut analyze video.mp4 --run runs/my-video --transcript captions.srt
```

## Optional API mode

Any endpoint that implements `POST {base_url}/chat/completions` can be used.
The key is read from the environment and is never written to the run directory.

```bash
export MOMENTCUT_API_BASE_URL="https://provider.example/v1"
export MOMENTCUT_API_MODEL="provider-model-name"
export MOMENTCUT_API_KEY="..."

python -m momentcut analyze video.mp4 --run runs/my-video --api
python -m momentcut render runs/my-video --aspect vertical
```

The prompt files are still exported in API mode. Completed block responses are
reused on retry, so a rate limit or a disconnected session does not repeat paid
calls. See [`docs/COLAB.md`](docs/COLAB.md) for provider/cost notes.

## Commands

| Command | Purpose |
|---|---|
| `analyze` | Probe, transcribe (or import), build line-level blocks, export prompts; optionally call an API |
| `import` | Parse provider answers, resolve line anchors, validate durations, dedupe, select clips |
| `render` | Cut and verify selected ranges with FFmpeg; resumable |

Useful options:

- `--asr-model tiny|base|small|medium|large-v3`, or a local model directory.
- `--asr-chunk-seconds 600` for resumable transcription; `0` disables chunking.
- `--transcript file.json|.srt|.vtt` to skip ASR.
- `--min-duration`, `--max-duration`, `--max-clips`, `--min-score`.
- `--content-type auto|podcast|interview|tutorial|comedy|vlog|debate|other`.
- `--focus "funny moments"` to bias editorial selection without changing the
  validation rules.
- `--source path` on `import`/`render` if the video moved since analysis.
- `--aspect original|vertical` on `render` (vertical = 1080x1920 blurred fill).
- `--force-transcribe` to rebuild the transcript; `--force` on render to redo
  existing files.
- `python -m momentcut import runs/my-video --min-score 80 --max-clips 4` applies
  a new selection without repeating AI work.

## What is enforced

- Every selected range is derived from transcript line timings, not model
  floats.
- `0 <= start < end <= source duration` is re-checked before rendering.
- Duration is repaired at line boundaries or the candidate is rejected.
- Invalid line IDs, conflicting quotes, start-after-end, overlong spans, and
  ambiguous quote matches are rejected with reasons.
- A candidate whose self-reported `standalone` score is below 4 is dropped even
  if its overall score is high; the prompt asks for 7 or better.
- Overlapping/near-duplicate clips are dropped in favor of the higher-scored
  one.
- No clip count is padded to satisfy a target.
- Rendered files are verified with ffprobe (duration, video, audio), not only by
  FFmpeg's exit code.
- Analysis is reusable independently from rendering.

## What is not guaranteed

- **Editorial taste.** "Stands alone" and "has a payoff" are semantic properties.
  The prompt and richer block context improve the odds; they do not prove the
  clip is good.
- **Score calibration.** Scores are model estimates. Cross-block scores may not
  be perfectly comparable.
- **Semantic duplicates.** Dedupe is temporal plus transcript similarity, not
  meaning-level. The same story retold in different words can survive.
- **Visual-only video.** The baseline is transcript-first. Silent or
  visual-dominant footage is not analyzed.
- **Speaker diarization, burned captions, smart face tracking.** Not in v0.1.
- **Real API quality.** Mechanical provider handling is tested against a local
  fake endpoint; no paid external provider is called by the tests.
- **Every language/borderline ASR case.** See [`docs/TESTING.md`](docs/TESTING.md).

## Google Colab

Open the notebook above. The first cell clones this public repository to
`/content/momentcut`; on a later run it uses `git pull --ff-only` to update the
checkout. There is no project zip to upload. Set `UPDATE_REPO = False` to skip
the pull or change `REPO_URL` for a fork.

It uses only faster-whisper locally; the AI step can be manual or API-based. No
local LLM or large vision model is required. Setup notes:
[`docs/COLAB.md`](docs/COLAB.md).

Important: free Colab has variable GPU availability and can disconnect. The
pipeline checkpoints ASR chunks, prompts, responses, and renders. Use a Drive
run directory for longer jobs, or download the run artifacts between stages.

## Costs

- Running MomentCut itself is free. Colab's free tier is free within its limits.
- Manual prompt/import can use a free provider web tier; there is no API bill
  from this project.
- API mode bills the provider by tokens. A 60-minute talk is roughly 5-7 blocks
  and tens of thousands of input/output tokens; the exact cost depends on the
  provider's current per-token price and retries. Check your provider's pricing.
- faster-whisper is local and free, but model download/storage/time are real
  costs on Colab.

## Documentation

- [`docs/FINDINGS.md`](docs/FINDINGS.md): OpenShorts investigation, verified
  limitations, alternatives, recommendation.
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md): pipeline, modules, decisions,
  and trade-offs.
- [`docs/JSON_CONTRACT.md`](docs/JSON_CONTRACT.md): provider output contract and
  validation rules.
- [`docs/COLAB.md`](docs/COLAB.md): free-Colab setup, performance, persistence,
  API requirements.
- [`docs/TESTING.md`](docs/TESTING.md): what was tested and what remains
  unverified.

## Repository layout

```text
momentcut/          application package
  asr.py            chunked faster-whisper + resume cache
  transcript.py     units, SRT/VTT/JSON import, timing
  prompting.py      chapter blocks and portable prompts
  contract.py       provider JSON parsing + candidate normalization
  resolve.py        line/quote -> validated seconds
  select.py         quality filter, duplicate handling, final selection
  pipeline.py       analyze/import lifecycle and manifests
  render.py         FFmpeg rendering + ffprobe verification
  provider.py       optional OpenAI-compatible client
  cli.py            command-line entry points
tests/              121 tests (unit + synthetic FFmpeg end-to-end)
docs/               investigation, architecture, contract, Colab, testing
notebooks/          Colab notebook
```

## Development

```bash
pip install -r requirements-dev.txt
python -m pytest tests -q
```

The FFmpeg tests generate their own synthetic video and are skipped when
FFmpeg/ffprobe are unavailable. See [`docs/TESTING.md`](docs/TESTING.md).






