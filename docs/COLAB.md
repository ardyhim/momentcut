# Running MomentCut on free Google Colab

The notebook is [`notebooks/momentcut_colab.ipynb`](../notebooks/momentcut_colab.ipynb).
This page explains the assumptions behind it.

## What runs in Colab, and what does not

Runs in Colab:

- FFmpeg/ffprobe probing and rendering;
- faster-whisper transcription (CPU or GPU if one is assigned);
- prompt generation and JSON validation;
- the optional HTTP client for an OpenAI-compatible provider.

Does **not** run locally in the baseline:

- an LLM;
- a video-understanding model;
- a face tracker or smart reframer.

That is deliberate. Free Colab cannot be assumed to have a GPU, and downloading
a local LLM would make the baseline fragile and slow. The AI provider is
external or manual.

## 1. Get the project into the runtime

Any one of these is fine:

**A. Clone from your own Git remote**

```python
!git clone <your-repo-url> /content/clip-generator
```

**B. Upload a zip of this project**

Use the notebook's upload cell; it extracts the zip under `/content`. The
notebook auto-detects a folder containing `momentcut/`.

**C. Put it on Google Drive**

```python
from google.colab import drive
drive.mount("/content/drive")
# then set PROJECT_DIR to /content/drive/MyDrive/.../clip-generator
```

Drive is persistent but slower for heavy reads/writes. A good compromise is to
keep code and `runs/` on Drive for long jobs, or download the important
artifacts after each stage.

## 2. Install dependencies

```bash
pip install -r requirements.txt
```

The important pin is `av>=12,<17`. `faster-whisper==1.2.1` calls
`av.open(..., metadata_errors="ignore")`; PyAV 19 removed that keyword, and a
fresh environment can otherwise install a combination that fails at the first
transcription. This was reproduced and fixed during development.

FFmpeg is normally preinstalled in Colab. Verify:

```bash
ffmpeg -version | head -n 1
ffprobe -version | head -n 1
```

If installation changed `av` in a runtime where it was already imported,
restart the runtime before transcribing.

## 3. Choose an ASR model

| Situation | Suggested model | Notes |
|---|---|---|
| GPU assigned (T4 or better) | `small` | Better punctuation/accuracy; still local |
| CPU only, short video | `base` | Reasonable compromise |
| CPU only, long video or slow session | `tiny` | Fastest, lower accuracy |
| Any, but transcript exists | skip ASR | Use `--transcript file.srt/.vtt/.json` |

`small` is the CLI default. The notebook selects `small` on a detected GPU and
`base` on CPU, but you can change it.

Model downloads (approximate, model-dependent): tiny ~75 MB, base ~145 MB,
small ~460 MB. The cache normally lives in the Hugging Face cache. To persist
it on Drive or avoid a read-only home directory:

```python
import os
os.environ["MOMENTCUT_MODEL_DIR"] = "/content/drive/MyDrive/momentcut-models"
```

You can also pass a local model directory directly:

```bash
python -m momentcut analyze video.mp4 --asr-model /content/models/faster-whisper-small
```

## 4. Recommended run shape

Keep the video, run directory, prompts, and responses in predictable paths:

```text
/content/clip-generator/
  video.mp4
  runs/colab-demo/
    transcript.json
    analysis/prompts/
    analysis/responses/
    clips/
```

Run analysis:

```bash
python -m momentcut analyze video.mp4 \
  --run runs/colab-demo \
  --asr-model small \
  --asr-chunk-seconds 600 \
  --min-duration 15 \
  --max-duration 90 \
  --max-clips 8
```

For a long video, 600-second ASR chunks create checkpoints. If the runtime is
lost, a later run reuses every completed chunk. The source fingerprint must
match, so do not edit the video between runs. If the upload path changed, pass
`--source /new/path/video.mp4` to `import` and `render`.

## 5. Manual AI flow (no API bill)

1. Download the prompt bundle from the notebook. It contains one
   `block_XXX.md` per chapter, plus `blocks.json` and `README.md`.
2. Paste each prompt into the provider of your choice. It is self-contained;
   no system prompt or conversation history is needed.
3. Save each answer as one file under `runs/colab-demo/analysis/responses/`.
   The filename can be anything (for example `block_001.json`); the JSON's
   `block_id` is what matters, and the filename is used as a fallback hint.
4. Run:

```bash
python -m momentcut import runs/colab-demo
```

The import validates every candidate and writes `analysis.json` and
`analysis/report.md`. No AI call happens at this stage.

5. Render:

```bash
python -m momentcut render runs/colab-demo --aspect vertical
```

6. Download `clips/`, `analysis.json`, `analysis/report.md`, and
   `render_manifest.json`.

If you use a free web tier, its own rate limits and content policies apply.
MomentCut cannot bypass them, and a provider's free tier is not a guarantee of
availability.

## 6. API flow (optional, paid)

Set the provider's OpenAI-compatible base URL and model:

```python
import os, getpass

os.environ["MOMENTCUT_API_BASE_URL"] = "https://api.provider.example/v1"
os.environ["MOMENTCUT_API_MODEL"] = "provider-model-name"
os.environ["MOMENTCUT_API_KEY"] = getpass.getpass("API key: ")  # not printed
```

Then:

```bash
python -m momentcut analyze video.mp4 --run runs/colab-demo --api
```

The client uses `POST {base}/chat/completions` and asks for a JSON object. It
falls back to simpler request shapes if a provider rejects optional parameters.
It retries 429/5xx with backoff; authentication and other 4xx errors fail
immediately.

The key is never stored in `run.json`, prompts, or analysis. It exists only in
the Colab runtime's environment.

Completed block responses are saved individually. Re-running the same command
retries only missing or unusable blocks. This is how an interrupted API run
avoids paying twice for finished chapters.

### Cost

Pricing is provider-specific and changes. Estimate it yourself:

- A 10-minute chapter's prompt is roughly 2,000-5,000 input tokens depending on
  speech density and prompt wording.
- A chapter response with several candidates is roughly 500-1,500 output tokens.
- A 60-minute talk is usually 5-7 chapters.
- Add retries and any reasoning tokens your provider bills.

At low-cost model rates this is usually cents; at premium rates it can be much
more. Check the provider's current prices and free-tier limits. Running
MomentCut itself and using free Colab do not remove provider charges.

## 7. Runtime, RAM, disk, and interruption

Free Colab resources are variable and not guaranteed. Treat these as practical
guidance, not quotas:

- **GPU** may not be assigned. The code automatically uses CUDA if
  `ctranslate2` reports an available device, and falls back to CPU int8 on a
  load failure.
- **RAM:** faster-whisper `tiny`/`base` are modest; `small` needs more. There is
  no PyTorch/vision stack in this baseline, which keeps headroom.
- **Disk:** a 16 kHz mono WAV chunk is much larger than the encoded audio. Chunks
  are deleted after transcription. Rendered 1080x1920 clips and the source video
  are the main consumers.
- **Time:** ASR dominates on CPU. For long videos, a GPU or an imported
  transcript is strongly preferred. Rendering re-encodes every clip; sequential
  rendering avoids fighting the free tier's limited CPUs.
- **Session loss:** prompts, responses, chunk checkpoints, analysis, and renders
  are separate files. The stage-level resume behavior is described in
  `docs/ARCHITECTURE.md`.

Use **Runtime > Disconnect and delete runtime** intentionally when you are done,
or download the artifacts first.

## 8. Troubleshooting

**`TypeError: open() got an unexpected keyword argument 'metadata_errors'`**

PyAV 19 is installed. Run `pip install -r requirements.txt` from this project;
it constrains PyAV below 17.

**`ffmpeg was not found`**

Install FFmpeg or use a Colab runtime that has it. The pipeline does not bundle
a binary.

**Hugging Face cache is not writable**

Set `MOMENTCUT_MODEL_DIR` to a writable folder (for example a Drive folder), or
download the model to a local path and pass that path to `--asr-model`.

**CUDA OOM / CUDA error**

The code falls back to CPU when the model load fails on CUDA. If the error
happens later, restart the runtime and use `--asr-model base` or `tiny` on CPU.

**`source video not found` after a restart**

Colab loses `/content` when the runtime is deleted. Re-upload the same video and
pass `--source /content/video.mp4` to `import` and `render`. The fingerprint must
match; a different file is refused.

**API 401/403**

The key/model/base URL is wrong, or the provider does not implement an
OpenAI-compatible chat endpoint. Fix the provider configuration; the client
does not retry authentication failures.

**API 429**

Re-run the same command. Completed blocks are reused. Respect the provider's
rate limits; do not remove retry delays.

**No valid candidates**

Read `analysis/report.md`. Common causes: invented line IDs, quotes that do not
match, clip spans beyond the duration band, or the model returning a
multi-block wrapper. The app rejects unsafe timestamps rather than guessing.

**No clips met the quality threshold**

This is a valid result. Review the report. If you want weaker candidates for
inspection, re-run `import` with a lower `--min-score`; this repeats no AI work.
