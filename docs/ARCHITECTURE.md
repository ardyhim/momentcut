# Architecture

MomentCut is deliberately a small, local-first pipeline. The model is an
untrusted advisor; the application owns timing, validation, and rendering.

## 1. Flow

```text
local video
   |
   v
ffprobe (duration, streams, source fingerprint)
   |
   v
transcript: faster-whisper word timestamps (chunked, resumable)
            or .json/.srt/.vtt import
   |
   v
sentence-like units (L0001, start, end, text)
   |
   v
overlapping chapter blocks (~10 min)
   |
   v
portable prompts / optional OpenAI-compatible API calls
   |
   v
provider JSON  ->  structural parse  ->  line/quote resolution
                                          |
                                          v
                                   duration + source bounds
                                          |
                                          v
                                   dedupe + quality filter
                                          |
                                          v
                                   analysis.json + report.md
                                          |
                                          v
                                   FFmpeg cut + ffprobe verification
                                          |
                                          v
                                   clips/*.mp4 + render_manifest.json
```

Every arrow after the video is a durable stage in the run directory. A lost
notebook session does not require repeating completed AI calls or renders.

## 2. Responsibilities

| Module | Responsibility |
|---|---|
| `media.py` | ffprobe, source fingerprints, audio chunk extraction, FFmpeg commands, output verification |
| `transcript.py` | words -> units, SRT/VTT/JSON import, text assembly, line IDs |
| `asr.py` | faster-whisper loading, overlapping chunk plan, per-chunk checkpoints, merge |
| `prompting.py` | chapter blocks, portable prompt, prompt bundle and hashes |
| `contract.py` | JSON extraction, candidate normalization, metadata/scores |
| `resolve.py` | line IDs + quotes -> unit indices -> safe times; duration repair |
| `select.py` | score threshold, duplicate suppression, top-N, chronological IDs |
| `pipeline.py` | run state, analyze/import orchestration, coverage, manifests, report |
| `render.py` | render plan, resume, FFmpeg verification, render manifest |
| `provider.py` | optional generic `/chat/completions` client with retry/fallback |
| `cli.py` | argparse entry points and actionable exit behavior |

The modules have one-way dependencies: the CLI and pipeline depend on the core;
the core does not depend on the CLI or provider.

## 3. Transcript units

LLM timing arithmetic is unreliable, so the model should not own cut seconds.
`transcript.build_units` creates sentence-like units from word timestamps:

- split after sentence punctuation (`.!?…`, CJK equivalents);
- split after a pause of at least 1.2 s once a unit has enough words;
- at a soft limit of 30 s / 450 characters, wait for a small pause before
  closing; a hard limit of 60 s / 900 characters prevents an unpunctuated
  monologue from becoming one enormous line;
- preserve Whisper's leading-space conventions so languages with and without
  spaces assemble correctly;
- clamp every boundary to the real source duration;
- assign sequential global IDs (`L0001`, `L0002`, ...).

SRT/VTT files become cue groups, merged until sentence punctuation or 20 s.
They are inherently less precise than word-level ASR; this is stated in the
docs rather than hidden.

`transcript.json` is the canonical timing artifact. It contains the source
duration, language, backend, and units. It is not re-derived from the model.

## 4. Chapter blocks

`build_blocks` groups units into blocks targeting `--block-seconds` (default
600). Consecutive blocks overlap at least the maximum clip duration plus 10
seconds. The next block starts at the last unit whose start is at or before the
overlap target, not the first after it; otherwise a complete moment could fall
into a gap. A tail under 60 s is merged into the previous block.

Each block prompt contains:

- source duration and block time range;
- the global line range;
- all lines in the block, with IDs and absolute times;
- language, target platforms, clip duration band;
- content-type guidance and optional user focus;
- editorial tests: cold-open hook, standalone setup, complete payoff, one idea,
  specificity, value;
- metadata and score rules;
- the exact JSON shape;
- an instruction that the transcript is data, not instructions.

Prompts are plain Markdown files. This is the provider-neutral interface:
anything that can read text and return JSON can participate.

## 5. Provider contract

The canonical response is a JSON object with `schema_version`, `block_id`, and
`clips`. Each candidate supplies *line IDs* and *exact quotes*, never
authoritative timestamps. See `docs/JSON_CONTRACT.md`.

The parser is deliberately tolerant of presentation and strict about meaning:

Accepted:
- a JSON object, a bare array, fenced JSON, or JSON surrounded by commentary;
- `shorts` as an alias for `clips`;
- our API envelope with `content`, or a raw chat-completions `choices` wrapper;
- a small set of common metadata aliases (for example
  `video_title_for_youtube_short`).

Rejected:
- unsupported schema major versions;
- candidate arrays with no usable structure;
- candidates with no line or quote anchor;
- candidates with no score signal;
- oversized response files (>10 MB).

One malformed candidate does not discard the other candidates in the same
answer. Candidate-level parse errors go into the rejection report.

## 6. Anchor resolution

For each boundary (`start`, `end`):

1. Resolve the declared line reference against the transcript. IDs, numeric
   forms, and `line 12` are accepted.
2. If the line is outside the declared block, reject it: the model was not
   shown that line.
3. Compare the quote to the line text using normalized, ordered token matching.
   Exact substrings score 1.0.
4. If the declared line and quote agree sufficiently (>=0.70), use the line.
5. If they disagree, search at most three neighbouring lines for a clear quote
   match (>=0.85). A clean local repair is accepted and warned; otherwise the
   candidate is rejected (or used with a warning only for a weak 0.50-0.70
   quote).
6. If the declared line is missing, search the block scope by quote. Require
   >=0.85 and reject ambiguous ties. A repair is warned.
7. A missing quote with a valid line ID is accepted with a warning because the
   line ID is still an exact transcript reference. A missing line and quote is
   rejected.

After both boundaries resolve:

- `start_index <= end_index` is required;
- the first/last complete lines define the content span;
- 0.20 s lead and 0.35 s tail are added (clipped to source bounds);
- if the clip is below the minimum, adjacent lines are added only across a
  gap of at most 1.5 s and only while the maximum is respected;
- if the span is above the maximum, reject;
- if only padding pushes it over the maximum, trim padding, never a line;
- a source shorter than the configured minimum is handled as the whole source,
  not rejected for being short.

Times are rounded to milliseconds. FFmpeg never sees a float that was not
derived from the transcript and checked against the source.

## 7. Selection

1. Candidates whose self-reported `standalone` score is below 4 are dropped as
   likely context-dependent, even if their overall score is high. The prompt
   asks for 7 or better; 4 is a mechanical safety floor, not an editorial
   target.
2. Candidates whose overall score is below `--min-score` are dropped.
3. Survivors are ranked by overall score, then standalone, hook, payoff, and
   start time.
4. A candidate is a duplicate if it shares >=35% of the shorter clip's seconds
   with a higher-ranked survivor, or its transcript similarity is >=0.82.
5. The best `--max-clips` survive.
6. Survivors are numbered chronologically (`C001`) for filenames while keeping
   `selection_rank` from the score order.

No quota or floor is enforced. A 60-minute video may produce zero clips if the
material or provider output does not meet the bar. That is a valid outcome.

The score is the model's overall estimate. It is not calibrated against a
human-labelled dataset. Dedupe detects near-duplicate transcript spans, not
semantic repetition.

## 8. Rendering

`analysis.json` is the only input to rendering. `render.py` re-probes the
source and refuses a file whose size/duration/quick hash differs from the file
that was analyzed.

For each selected clip:

- FFmpeg uses `-ss` before `-i` and re-encodes with libx264/AAC for accurate
  boundaries (stream copy would snap to keyframes and break the validation
  contract).
- Original framing is the default.
- `--aspect vertical` produces a 1080x1920 blurred-pillarbox frame. It does not
  perform face tracking or smart cropping.
- Output duration, video stream, and (when applicable) audio stream are checked
  with ffprobe.
- A failed clip is marked in `render_manifest.json`; other clips continue.
- Existing verified renders are reused; `--force` redoes them.
- Original and vertical outputs get distinct filenames (`_vertical`), so both
  can coexist.

## 9. Run state and resume

```text
run.json                  source fingerprint, settings, transcript/bundle refs
transcript.json           normalization output
asr/chunks-<key>/         per-chunk checkpoints
analysis/blocks.json      block ranges + prompt hashes
analysis/prompts/         portable prompts
analysis/responses/       provider answers
analysis/api_errors.json  optional API call failures
analysis.json             resolved/selected/rejected results
analysis/report.md        human-readable report
render_manifest.json      output verification and resume state
clips/                    rendered MP4s
```

- Re-running `analyze` reuses a compatible transcript and prompt bundle.
- Prompt-affecting settings changes archive old responses instead of silently
  mixing answers produced for another prompt.
- Re-running `import` re-reads saved responses; it never calls an AI provider.
- Re-running `render` skips verified outputs.
- `--source` relocates a source whose path changed (for example after a Colab
  restart); the fingerprint must still match.

## 10. Error handling

- Missing FFmpeg/ffprobe produces an installation message.
- Missing audio with no transcript fails with a clear visual-analysis scope
  message.
- An unparseable provider answer writes `analysis.json` and a report, then
  fails with a reason.
- A candidate failure is isolated; it does not abort other candidates.
- API 401/403-style errors are not retried; 429/5xx are retried with backoff.
- Optional provider parameters (`response_format`, `max_tokens`, `temperature`)
  get compatibility fallbacks; each variant is recorded only by its outcome.
- Every selected/rejected/dropped candidate has a reason in the report.

## 11. Security and privacy

- No API keys are written to run state, logs, prompts, or analysis manifests.
- Transcripts and model answers are treated as untrusted data, never code.
- FFmpeg is invoked with argument arrays, not a shell.
- Output filenames are a fixed clip ID plus an ASCII slug; provider text cannot
  choose a path.
- The source is local; the baseline does not download URLs or upload video.
- Prompts warn the model that transcript text is data, not instructions.

## 12. Deliberate non-goals (v0.1)

- Web UI, job server, database, queue.
- YouTube download/upload.
- Local LLM or multimodal vision models.
- Smart vertical reframing, face tracking, burned captions.
- Speaker diarization.
- A provider plugin registry.
- Forced clip counts or a retention-driven floor.

These are excluded to keep the core reliable and within free-Colab limits. The
extension seams are clear: add a transcript importer, a prompt strategy, a
provider call, or a render filter without changing the timing contract.
