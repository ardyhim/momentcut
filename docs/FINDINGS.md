# OpenShorts investigation and design decision

Status: investigation complete. This document separates what was verified in
source code from what is an inference or proposal.

Reference inspected: [`mutonby/openshorts`](https://github.com/mutonby/openshorts)
at commit `9474289c09332f10afd8bdcad1f32acfa050f588` (2026-10-25). The clone is
kept locally under `_reference/openshorts` and is not part of the product.

## 1. What OpenShorts actually does

### 1.1 Pipeline

The spoken-video path is:

```text
ingest -> faster-whisper words -> build ~90 s transcript windows
       -> LLM scores windows -> global shortlist
       -> LLM returns 15-60 s clips with copy
       -> snap cuts to words -> dedupe overlaps
       -> FFmpeg cut -> reframe/captions/hooks -> publish
```

Verified entry points and helpers:

- [`main.get_viral_clips()`](https://github.com/mutonby/openshorts/blob/9474289c09332f10afd8bdcad1f32acfa050f588/main.py#L2021-L2188)
- [`clip_selection.py`](https://github.com/mutonby/openshorts/blob/9474289c09332f10afd8bdcad1f32acfa050f588/clip_selection.py)
- [`gemini_worker.py`](https://github.com/mutonby/openshorts/blob/9474289c09332f10afd8bdcad1f32acfa050f588/gemini_worker.py)
- [`llm_backend.py`](https://github.com/mutonby/openshorts/blob/9474289c09332f10afd8bdcad1f32acfa050f588/llm_backend.py)

The default model is `gemini-3.1-flash-lite` and can be overridden with
`GEMINI_MODEL` (`main.py:2043`). This is a Gemini-first design; an
OpenAI-compatible local endpoint is a secondary path for the two text passes
only.

### 1.2 Window construction

`build_transcript_windows()` (`clip_selection.py:203-258`) grows windows from
Whisper segment boundaries until roughly `window_seconds` (default 90 s, up to
1.25x), then starts the next window around 30 s before the previous end. This
gives overlapping, segment-aligned windows.

The reason stated in the code is that a single call over a whole transcript
clusters picks near the start (`main.py:2023-2027`). This is a real design
constraint for long videos.

### 1.3 Two LLM passes

**Pass 1: score windows**

- Batches of 8 with Gemini, or 3 with a local OpenAI-compatible model
  (`main.score_batch_size`, `main.py:2009-2018`).
- The prompt scores every supplied window from 0 to 100, with the "2-second
  test" as the main criterion (`gemini_worker.SCORE_PROMPT_TEMPLATE`,
  `gemini_worker.py:254-290`).
- The shortlist is the global top `shortlist_target(duration)`, which is
  `min(10, duration // 90 + 2)`, floored at 3 (`clip_selection.py:70-81`).
- All windows are batched, but the shortlist is chosen globally by ID, not per
  batch (`main.py:2085-2097`). This was a deliberate fix: the older
  "choose up to three per batch" cap starved the shortlist.

**Pass 2: detail clips**

- The shortlist windows are sent in one call (up to 10 windows).
- The prompt asks for 15-60 s clips, anchored in absolute seconds, with rules:
  the 2-second rule, "stands alone", don't cut mid-word, return a count band,
  diversity, concrete metadata (`gemini_worker.DETAIL_PROMPT_TEMPLATE`,
  `gemini_worker.py:292-365`).
- The model is asked for a `source_window_id`, `predicted_score`, TikTok and
  Instagram descriptions, a YouTube title, a hook, and a `why`.
- If the result is below the calculated clip floor, unused shortlist windows
  get one more detail call (`main.py:2117-2137`). An empty second result is
  accepted rather than padded.

### 1.4 Post-processing

- **Word snapping**: `snap_clip_to_words()` (`clip_selection.py:303-365`) finds
  the nearest word start/end within 1.5 s, adds up to 0.35 s lead and 0.45 s
  tail, and repairs minimum/maximum duration using word ends.
- **Trim by score**: if the model returns more than the maximum, the list is
  trimmed by `predicted_score`, not by transcript position
  (`clip_selection.py:132-164`).
- **Dedupe**: overlapping clips sharing at least 25% of the shorter clip are
  dropped, keeping the higher score (`clip_selection.py:261-300`).
- Clips are then cut by FFmpeg with `-ss`/`-to`, reframed, captioned, and
  optionally hooked. The reframe stage uses scene detection, but scene
  boundaries are not passed to the moment picker.

### 1.5 Silent / non-speech footage

`transcribe_video` raises `NoAudioError`, or `speech_is_sparse()` detects fewer
than 8 words or fewer than 5 words/minute. The video is then uploaded whole to
Gemini, which returns the same `{"shorts"}` shape
(`main.get_visual_clips`, `main.py:2209-2289`).

The code's own architecture note is explicit that this is the only stage that
sends the whole video, that it is Gemini-only, and that nothing guards the
length; roughly one hour is already ~1.08M tokens, beyond a 1M window
(`docs/architecture/clip-selection.md:42-48`). This is not a viable
provider-neutral or Colab-friendly baseline.

### 1.6 Provider handling

- Gemini uses server-enforced `response_schema` with pydantic models
  (`gemini_worker.py:22-51`), plus JSON text recovery for the fallback path.
- The OpenAI-compatible path uses the same pydantic schemas after asking for
  `json_schema`, then `json_object`, then no response format
  (`llm_backend.py:86-153`). This is a good compatibility pattern.
- There is no exported prompt / imported answer workflow. Manual use of another
  provider is not a supported path; the two live backends are Gemini and an
  OpenAI-compatible HTTP endpoint.
- Vision analysis (silent videos) remains Gemini-only even when the text
  backend is local.

## 2. Why the approach works

These are the parts worth learning from, not necessarily copying:

1. **Transcript windowing prevents positional bias.** One giant call tends to
   pick near the beginning; overlapping windows give every part of a long video
   a chance.
2. **Cheap retrieval, expensive detail.** Scoring many windows first keeps the
   detailed generation pass focused. This is a sensible cost/quality trade-off.
3. **Global shortlist, not per-batch winners.** Scores are compared across
   batches instead of letting every batch contribute a quota.
4. **Transcript is the timing ground truth.** Word-level timestamps are used to
   repair LLM arithmetic. This is the right direction, even though the
   implementation is incomplete.
5. **The prompt encodes editorial tests.** The 2-second test, "stands alone",
   complete payoff, and diversity rules are concrete and useful.
6. **Deterministic cleanup after the model.** Word snapping, score-based
   trimming, and overlap dedupe are simpler and more reliable than asking the
   model to guarantee them.
7. **Structured output with recovery.** Schema enforcement plus JSON recovery
   and retries is a pragmatic response to provider variability.

## 3. Verified limitations and failure modes

### 3.1 Timestamps are not actually guaranteed to be valid

The pydantic schema only requires floats. There is no final check that
`0 <= start < end <= video_duration`. `snap_clip_to_words` returns the original
proposal when it has no words (or cannot repair duration), and it does not
clamp an unsnapped original (`clip_selection.py:316-317, 349-361, 363-364`).
`get_viral_clips` writes the snapped result back without a final range check
(`main.py:2147-2156`). The renderer then passes it to FFmpeg.

Consequence: a provider can return an out-of-range or inverted range and the
job may fail late, or produce a shortened clip, rather than rejecting the
candidate with a precise reason.

### 3.2 Line/segment anchors are not validated

`sources_window_id` is required by the schema but never checked against the
supplied window IDs. `start`/`end` are not checked against the candidate window
either. The prompt says "stay within the window"; the code does not enforce it.
The scoring pass's returned scores are mapped by ID (unknown IDs are discarded),
but the code does not require every input window to be scored.

Consequence: a model can produce a plausible-looking clip that is outside the
material it actually saw, or skip a window silently.

### 3.3 Fixed windows cap context and can split stories

A clip must fit inside a ~90 s candidate window. A story that needs 2 minutes of
setup cannot be represented, even if it would make the best clip. Overlapping
windows reduce missed coverage, but they also create duplicate candidates and
can cut an arc at a window wall. The detail prompt is instructed to "stay within
the candidate window boundaries", so this is a structural limit, not just a
prompt issue.

### 3.4 Count floor is a retention optimization, not a quality guarantee

`clip_count_targets()` explicitly derives a floor from production retention
data (jobs with 4-9 clips retained better than jobs with 1-3;
`clip_selection.py:29-67`). The code then re-calls unused windows to reach the
floor. This is reasonable for a consumer product, but it is in tension with
"quality over quantity": a floor creates pressure to find clips the material
does not support. The prompt says not to pad, and the code accepts an empty
retry, but the incentive remains.

### 3.5 Scores are self-reported and not calibrated

`predicted_score` drives trimming and dedupe, but it is a single model estimate
per clip. Scores from different batches/windows are not normalized or
cross-checked. A model can score generously, and the highest reported number
wins even if another clip is better. There is no pairwise comparison or
holdout validation in the pipeline.

### 3.6 Dedupe is temporal only

`dedupe_overlapping()` catches clips sharing seconds, but not the same story
told twice at different points in a long video. The diversity instruction is
the only defense against semantic duplicates. It is a prompt rule, not a
verified property.

### 3.7 "Stands alone" is prompt-only

The pipeline does not verify that the opening actually contains its setup.
Starting at a pronoun, an answer to an earlier question, or a trailing
"so anyway" is a semantic failure that a word-snap cannot detect. The prompt
tries to prevent it; the code does not check it. This is the hardest quality
problem and should not be overclaimed.

### 3.8 Spoken-video selection is transcript-only

For normal talking videos, the model never sees frames or hears audio prosody.
That is a reasonable cost trade-off for podcasts and interviews, but it misses
visual reveals, physical comedy, demonstrations where the words are generic,
and emotional delivery. The silent-video fallback does the opposite
(video-only, no transcript) and is Gemini-only and length-unguarded.

### 3.9 The full stack is not a free-Colab baseline

The reference requirements include PyTorch 2.11, torchvision, Ultralytics,
MediaPipe, TransNetV2, faster-whisper, Google GenAI, OpenCV, and more
(`requirements.txt`). Its README targets Docker with 8 GB+ RAM; it also supports
GPU acceleration. The clip-selection core itself is lighter, but the complete
application is not a reasonable "free Colab, no local large models" target.

### 3.10 No portable prompt/answer contract

The prompts are embedded in Python and provider calls are live. There is no
first-class way to take the analysis prompt to Claude, ChatGPT, DeepSeek, a
local model, or a free web tier, then import the answer. The OpenAI-compatible
path helps for API endpoints, but it is still an API integration and does not
cover manual copy/paste.

### 3.11 Quality claims are not testable from this repository alone

The repository contains unit tests for windowing, trimming, dedupe, snapping,
and floor behavior. Those tests verify mechanics. They do not establish that
selected clips are editorially good, and the retention figures in comments are
product analytics that cannot be independently verified from source. We should
treat the prompt principles as hypotheses to be evaluated, not proven quality.

## 4. Facts vs. inferences

### Verified in source

- Window defaults, scoring/detail passes, model defaults, structured schemas,
  word snapping, score-based trim, temporal dedupe, count floor, silent-video
  fallback, and provider branching described above.
- Scene detection is used by reframing, not by `get_viral_clips`.
- Final timestamp bounds and anchor IDs are not validated in the selection
  layer.
- There is no prompt export/import path.
- The vision path is Gemini-only and unguarded by duration.

### Inference / judgment

- "Long transcript calls cluster near the start" is the project's stated
  rationale; I did not run an attention experiment. The window design is still a
  prudent way to bound context.
- The 25% overlap dedupe threshold is a product choice, not an objectively
  correct number.
- The retention-based clip floor is plausible as product analytics, but I did
  not verify the underlying data.
- The quality of OpenShorts' clips was not measured on a labelled dataset. The
  prompt rules are sensible heuristics, not evidence of superiority.
- The proposed alternative below is a reasoned design; its editorial quality
  also remains to be measured, not assumed.

## 5. Options considered

| Option | Strengths | Costs / risks | Verdict |
|---|---|---|---|
| **A. Reuse OpenShorts wholesale** | Proven mechanics, all features | Gemini-first, heavy Docker/GPU stack, hard to run free Colab, keeps validation gaps | Reject as the product base |
| **B. Improve its fixed-window two-pass flow** | Familiar, bounded context, cheap | Still window-capped, timestamp trust remains, forced count, many calls | Partial reuse only |
| **C. Chapter-level editorial pass with sentence anchors** | Full context, provider-neutral prompt, timestamps derived from transcript, no fixed window wall, one call per chapter | Chunking/duplicate handling; scores still uncalibrated; transcript-only | **Recommended baseline** |
| **D. Multimodal video/audio analysis** | Can catch visual/emotional moments | Provider lock or large local models, huge token/upload cost, hard on free Colab, hard to validate | Later extension, not baseline |
| **E. Local embeddings / semantic segmentation** | Cheap after model download; deterministic | Adds a local model, doesn't understand complete arcs, threshold tuning, Colab resource cost | Not justified for v1 |
| **F. Whole-transcript single call** | Maximum global context, simplest | Context limits, cost, positional bias, weak on 2-3 h sources | Fallback for short videos, not default |

## 6. Recommended architecture

Working name: **MomentCut** (the package name can be changed without affecting
the design).

### 6.1 Core principle: the model chooses meaning, the engine chooses seconds

The AI returns **transcript line IDs and exact quote anchors**, not authoritative
timestamps. The engine resolves those anchors against the transcript, expands
them to complete sentence boundaries, applies duration/source bounds, and only
then hands ranges to FFmpeg.

This makes invalid timestamps structurally unlikely and works identically for
any provider.

### 6.2 Transcript units

- Transcribe with faster-whisper word timestamps, or import SRT/VTT/JSON.
- Build sentence-like units from words, with:
  - global IDs (`L0001`),
  - start/end from real word timings,
  - text reconstructed with source-language spacing.
- These units are the only legal clip boundaries.

Why units instead of fixed windows:

- the model can select any complete arc, not one squeezed into 90 s;
- every boundary is already a real timestamp;
- the prompt can show line IDs, making hallucinated arithmetic obvious;
- clips can span chapter boundaries without inventing a new cut point.

### 6.3 Chapter-level analysis

- Group units into overlapping blocks of ~10 minutes (configurable), with
  overlap at least the maximum clip duration so no moment can fall between
  blocks.
- Each block gets one self-contained prompt with all its lines, timestamps,
  language, focus, duration band, and editorial rubric.
- The model returns zero or more candidates, not a forced count.
- No fixed 90 s window is used as a selection unit.

Why blocks rather than one call:

- bounded prompt cost and context, even with 2-3 h videos;
- focused attention and local context;
- can run on models with modest context windows;
- each block is independently retryable/resumable.

The trade-off: scores from different blocks are not perfectly calibrated. We
mitigate this with a strict rubric and deterministic cross-block dedupe/ranking,
and we document the residual risk rather than hiding it.

### 6.4 Provider strategy

Two paths, one contract:

1. **Manual (primary, zero API coupling):** export prompt files, the user pastes
   them into any AI provider, saves the JSON answer, imports it.
2. **API (optional convenience):** one minimal OpenAI-compatible
   `/chat/completions` adapter. This covers OpenAI, DeepSeek, Groq, OpenRouter,
   Together, local Ollama/LM Studio/vLLM, and other compatible endpoints. No SDK,
   no provider-specific prompt path.

The manual path is what guarantees provider freedom, including free web tiers.
The API path is deliberately one adapter, not a provider abstraction framework.

### 6.5 Stable JSON contract

Provider response (schema version 1.0):

```json
{
  "schema_version": "1.0",
  "block_id": "B001",
  "clips": [
    {
      "start_line": "L0002",
      "end_line": "L0004",
      "start_quote": "exact words at the start",
      "end_quote": "exact words at the end",
      "title": "concrete title",
      "hook": "short overlay text",
      "caption": "1-2 social sentences",
      "hashtags": ["#tag"],
      "summary": "one sentence",
      "why": "specific editorial reason",
      "moment_type": "story",
      "scores": {
        "hook": 8,
        "standalone": 9,
        "payoff": 8,
        "overall": 84
      }
    }
  ]
}
```

Rules:

- The parser accepts fenced JSON, surrounding prose, a bare array, and
  `shorts` as an alias for `clips`. It does not invent missing structure.
- Line IDs are resolved against the block's actual line range.
- Quotes are normalized and fuzzy-matched. A conflicting anchor is rejected;
  a missing quote can be repaired from a valid line ID, with a warning.
- Duration is expanded at unit boundaries to satisfy the minimum. A candidate
  that still cannot satisfy the band, or exceeds the maximum, is rejected with
  a reason rather than silently rendered.
- Source bounds are checked against the probed duration.
- Metadata is normalized to safe lengths; missing cosmetic fields get
  deterministic fallbacks and warnings. A clip without a trustworthy timestamp
  anchor is never rescued.
- Every candidate, selected or rejected, is reported, so the user can see why a
  provider's output did not survive validation.

### 6.6 Selection and dedupe

- Candidates are resolved first, then deduped by:
  - temporal overlap of at least 35% of the shorter clip, or
  - high text similarity of the covered transcript plus temporal proximity.
- The higher `overall` score wins; ties prefer stronger standalone/hook/payoff.
- Apply a minimum score (default 60) and a maximum number of clips (default 8).
- Never pad to hit a count. If only four clips survive, four clips are returned.
- Keep final output in chronological order, with the selection rank retained.
- Do not force per-chapter quotas; a single chapter may legitimately hold all
  the best moments.

### 6.7 Quality rubric

The prompt asks for four signals:

- **Hook (0-10):** would the first two seconds make a cold viewer stop?
- **Standalone (0-10):** can someone who saw nothing else follow the setup?
- **Payoff (0-10):** does the clip end on a complete reveal, punchline, insight,
  decision, or emotional beat?
- **Overall (0-100):** publish-worthiness, with explicit anchors. 80+ is
  exceptional; 60-79 is solid but not required; below 60 should not be returned.

The prompt also includes content-type guidance (podcast/interview/tutorial/
comedy/vlog/debate or auto) and an optional user focus, because a single
definition of "interesting" does not fit every format.

The model is told to:

- start at the first line a cold viewer needs, not on a pronoun or an answer;
- end after the payoff, not on a trailing filler line;
- return one clip per distinct idea;
- skip a moment if the complete arc cannot fit the duration band;
- write concrete, non-generic copy in the transcript language;
- never invent a claim, timestamp, or visual detail.

These are still prompt-level guarantees. The engine enforces boundary validity,
not editorial truth. We will state that distinction in the docs.

### 6.8 Rendering

- FFmpeg re-encodes each selected range (`libx264`, AAC) for accurate cuts and
  fast start.
- Original aspect ratio by default; optional 9:16 blurred-pillarbox output.
  Deliberately no AI face tracking/reframing in v1.
- Validate the output with ffprobe (duration, video stream, audio stream), not
  just the FFmpeg exit code.
- Store analysis and rendering separately. Re-rendering does not call the AI
  again.
- Use sequential rendering; free Colab has few CPUs and parallel x264 encodes
  usually make the session slower, not faster.

### 6.9 Colab behavior

- Only local model required: faster-whisper. No local LLM, no diffusion, no
  torch-based vision stack.
- Default to word-level transcription; allow SRT/VTT/JSON import to skip ASR
  entirely.
- Checkpoint ASR per chunk and resume completed chunks after a disconnect.
- Persist every stage in the run directory; prompts, raw responses, analysis,
  and rendered clips are all restartable.
- Keep the AI API optional. Manual copy/paste is the baseline and has no API
  bill.
- Document Colab's free-tier variability: GPU not guaranteed, RAM/disk/time
  limits, session loss, and the difference between Colab cost and provider cost.

## 7. What we deliberately do not build in v1

- A web UI or job server.
- YouTube downloading/uploading. Users can bring a local file; downloading is a
  separate, policy-sensitive integration.
- AI face tracking / smart cropping / burned captions.
- Speaker diarization.
- Full multimodal analysis of arbitrary visual-only video.
- A plugin registry for AI providers.
- A local LLM.

These are not rejected forever; they are excluded because they do not improve
the core promise enough for their cost and failure modes.

## 8. Implementation plan

1. **Media + transcript foundation**
   - ffprobe validation, source fingerprint, durable run directory.
   - faster-whisper integration, chunked/resumable ASR, SRT/VTT/JSON import.
   - sentence-unit builder and unit tests.
2. **Prompt and contract**
   - overlapping chapter blocks, prompt export, prompt manifest.
   - JSON parser, contract validation, anchor resolution, metadata normalization.
   - unit tests for malformed/partial provider outputs.
3. **Selection**
   - duration/source bounds, score filtering, dedupe, chronological selection.
   - analysis manifest with selected and rejected candidates.
4. **Optional API**
   - minimal OpenAI-compatible client, retries, response-format fallback,
     raw usage capture, resumable per-block calls.
5. **Rendering**
   - FFmpeg cut/encode, optional vertical, ffprobe verification, resume,
     render manifest.
6. **Colab + docs**
   - notebook for both manual and API flows, prompt/response download cells.
   - README, JSON contract, cost/runtime notes, limitations.
7. **Validation**
   - unit tests; synthetic end-to-end video cut with real FFmpeg; fake HTTP
     provider; malformed-output fixtures.
   - clearly report what remains unverified (real provider quality, actual Colab
     runtime, ASR accuracy across languages).

## 9. Acceptance criteria

A run is successful only if:

- every selected clip has a start/end within the probed source duration,
  `start < end`, and a duration inside the configured band;
- the cut files exist, are playable, have the expected duration, and preserve
  an audio stream when the source has one;
- malformed or out-of-range AI candidates are rejected with reasons, not passed
  to FFmpeg;
- the same provider response can be imported and rendered without any provider
  SDK;
- the manual prompt path works with no API key;
- the pipeline survives restart at stage boundaries and does not repeat
  completed AI calls/renders;
- documentation states what was tested and what was not.

## 10. Remaining risk

The biggest unsolved problem is not parsing; it is editorial judgment.
"Stands alone" and "has a payoff" are semantic properties that neither line IDs
nor fuzzy matching can prove. This design reduces the mechanical failure rate
and gives the model much better context than a 90 s window, but it does not
guarantee good taste. A proper evaluation would need a labelled set of long
videos, blind human ratings, and comparison across prompts/models. That is a
separate experiment and is not claimed here.

## 11. Implementation outcome (v0.1)

The investigation above was turned into a working baseline rather than left as
a design document.

- **Package:** `momentcut/` with `analyze`, `import`, and `render` commands, a
  provider-neutral JSON contract, and a Colab notebook.
- **Transcript:** real faster-whisper transcription with overlapping,
  resumable chunks; SRT/VTT/JSON import; sentence-unit construction.
- **Provider freedom:** the manual prompt export/import path is the primary
  interface; one optional OpenAI-compatible HTTP adapter covers API users.
  No Gemini SDK is present.
- **Timing safety:** line IDs and quotes are resolved against transcript
  timings; duration/source bounds are enforced before FFmpeg. Unsafe
  candidates are rejected with reasons.
- **Rendering:** original and 1080x1920 blurred-pillarbox output, verified with
  ffprobe; original/vertical outputs coexist and resume.
- **Resume:** ASR chunks, prompts, provider answers, analysis, and clips are
  durable stage artifacts; moving a source is supported via `--source` with a
  fingerprint check.
- **Dependency finding:** faster-whisper 1.2.1 is incompatible with PyAV 19
  (`metadata_errors` was removed from `av.open`). The project pins
  `av>=12,<17`; PyAV 16.0.1 was used for the real ASR runs.
- **Validation:** 121 automated tests pass. Real CPU ASR was exercised on an
  11.1 s clip (3 units) and a 105 s clip split into two overlapping chunks
  (24 units). A hand-written provider answer was imported and rendered into a
  verified 10.63 s vertical clip. Details are in `docs/TESTING.md`.

Not established by v0.1:

- editorial quality on a labelled corpus;
- live Colab/GPU execution;
- real external provider behavior and cost;
- multilingual accuracy beyond English smoke tests;
- visual-only content.

Those are stated as limitations in the docs and should be the next experiments,
not silently claimed as solved.


