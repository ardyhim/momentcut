# Provider JSON contract (schema 1.0)

The AI provider returns one JSON object per chapter prompt. The importer
accepts it as a file (`.json`, `.txt`, `.md`), a fenced block, or JSON with
surrounding commentary. It then validates the content against
`transcript.json` and the block manifest.

The key rule: **the model never provides the authoritative cut timestamps.**
It selects line IDs and quotes; the engine resolves seconds.

## Canonical response

```json
{
  "schema_version": "1.0",
  "block_id": "B001",
  "clips": [
    {
      "start_line": "L0002",
      "end_line": "L0005",
      "start_quote": "the first exact words of the opening line",
      "end_quote": "the last exact words of the closing line",
      "title": "A concrete title under 100 characters",
      "hook": "Short overlay hook, at most 10 words",
      "caption": "One or two sentences that tease the payoff without spoiling it.",
      "hashtags": ["#relevant", "#specific"],
      "summary": "One sentence describing what happens in the clip.",
      "why": "Why this moment is complete and worth publishing.",
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

`clips` may be an empty array. An empty answer is valid and preferred over
padding.

## Fields

### Envelope

| Field | Required | Notes |
|---|---|---|
| `schema_version` | recommended | Must be a `1.x` version. Missing is accepted as 1.0 with a note. |
| `block_id` | recommended | `B001`-style. Missing is inferred from line IDs or the filename and warned. |
| `clips` | required | Array, max 100 candidates per file (larger is truncated with a note). `shorts` is an alias. |

### Candidate anchors

| Field | Required | Notes |
|---|---|---|
| `start_line` | one of line/quote | Global line ID from the prompt's `BLOCK_LINES`, e.g. `L0002`. |
| `end_line` | one of line/quote | Must not precede `start_line`. |
| `start_quote` | one of line/quote | Exact contiguous phrase from the start line, about 5-12 words. No ellipsis, no paraphrase. |
| `end_quote` | one of line/quote | Exact phrase from the end line. |

The engine accepts a valid line ID without a quote (warning), or a quote
without a line ID if the quote resolves unambiguously (repair warning). It does
not accept a boundary with neither.

### Copy

| Field | Required | Rule |
|---|---|---|
| `title` | recommended | Concrete, <=100 characters. Missing is derived from the transcript and warned. |
| `hook` | recommended | Overlay text, <=10 words / 80 characters. Missing derives from title. |
| `caption` | recommended | 1-2 social sentences, <=1000 characters. Missing derives from summary. |
| `hashtags` | recommended | 3-6 relevant tags. A list is preferred. Caption hashtags are extracted only if this field is absent. |
| `summary` | recommended | One sentence. Missing derives from the clip transcript. |
| `why` | recommended | Specific rationale, <=250 characters. Missing gets a generic note. |
| `moment_type` | optional | `story`, `insight`, `humor`, `reaction`, `emotion`, `conflict`, `demo`, or `other`. Unknown values become `other`. |

Copy is expected in the transcript language. The importer does not machine-
translate it.

### Scores

| Field | Type | Meaning |
|---|---|---|
| `scores.hook` | integer 0-10 | Would the first two seconds stop a cold viewer? |
| `scores.standalone` | integer 0-10 | Is the setup understandable without prior context? |
| `scores.payoff` | integer 0-10 | Does it land a complete payoff? |
| `scores.overall` | integer 0-100 | Publish-worthiness. |

At least one score signal is required. If `overall` is absent but dimensions
exist, `overall = mean(dimensions) * 10` with a warning. If dimensions are
missing but `overall` exists, dimensions default to `round(overall / 10)` with a
warning. Values are clamped to their ranges.

`overall` drives ranking. The other dimensions are tie-breakers and review
signals. Scores are estimates, not calibrated probabilities.

## Accepted aliases

The prompt asks for the canonical names above. To import older
OpenShorts-style answers without a transformation step, these aliases are
accepted:

- `title` <- `video_title_for_youtube_short`, `video_title`
- `hook` <- `viral_hook_text`
- `caption` <- `video_description_for_tiktok`, `video_description_for_instagram`,
  `description`
- `summary` <- `summary`
- `why` <- `reason`
- `scores.overall` <- top-level `predicted_score`
- `start_line` <- `start_line_id`, `start_unit`, `start_id`
- `end_line` <- `end_line_id`, `end_unit`, `end_id`
- `start_quote` <- `start_text`
- `end_quote` <- `end_text`

The aliases do not relax timing validation. Old answers with only
`start`/`end` floats, no line anchors, and no quotes are rejected; they cannot
be verified against the transcript.

## Parsing behavior

Accepted wrappers:

1. the canonical object;
2. a bare array of candidates;
3. `{"clips": [...]}` or `{"shorts": [...]}`;
4. a fenced or prose-wrapped object;
5. the API wrapper written by this project:
   `{"block_id": ..., "content": "<model text>"}`;
6. a chat-completions response with `choices[0].message.content`.

Not accepted:

- a multi-block `{"results": [...]}` wrapper (import one answer per block);
- unsupported major schema versions;
- files larger than 10 MB;
- a response with no `clips`/`shorts` array.

A malformed candidate is skipped and recorded; other candidates in the same
response continue.

## Resolution rules

For each boundary:

1. Resolve the line reference to a transcript unit. `L0012`, `12`, and
   `line 12` are understood.
2. If a `block_id` is known, the line must be inside that block.
3. Compare the quote to the candidate line using normalized, ordered token
   matching. An exact substring scores 1.0.
4. If the quote and line agree (>=0.70), accept the line.
5. If they conflict, search up to three neighboring lines for a clear match
   (>=0.85). A clear local match is a repair with a warning. A weak 0.50-0.70
   match uses the declared line with a warning. Otherwise reject.
6. With no valid line ID, search the block (or whole transcript) by quote.
   Require >=0.85 and reject ambiguous ties. A repair gets a warning.
7. If start line index > end line index, reject.
8. Add 0.20 s lead / 0.35 s tail, clipped to the source.
9. If below minimum duration, add adjacent lines only across a <=1.5 s gap and
   only while under maximum. If that cannot reach the minimum, reject.
10. If above maximum duration, reject. If only padding is over, trim padding
    and keep every line.
11. Clamp to the probed source duration; reject non-positive ranges.

## Rejection examples

| Input | Outcome |
|---|---|
| `start_line: L9999`, quote matches L0002 | Repair to L0002 with warning |
| `start_line: L9999`, invented quote | Reject: quote not found |
| `start_line: L0006`, `end_line: L0002` | Reject: start after end |
| Both lines valid, span 50 s, maximum 30 s | Reject: above maximum |
| Valid line ID, no quotes | Accept with warning |
| Same exact quote on two lines, no IDs | Reject: ambiguous |
| Candidate from B002 while `block_id` is B001 | Reject: outside block |
| Valid anchors but no score | Reject: missing score |
| Good anchors, missing title/hook | Accept; metadata derived from transcript with warnings |

Exact reasons for every candidate are written to `analysis.json` and
`analysis/report.md`.

## Internal resolved clip

After import, `analysis.json` contains the selected clips. A clip looks like:

```json
{
  "id": "C001",
  "selection_rank": 1,
  "start": 4.9,
  "end": 30.15,
  "duration": 25.25,
  "start_line": "L0002",
  "end_line": "L0006",
  "block_id": "B001",
  "anchors": {
    "start_line": "L0002",
    "end_line": "L0006",
    "start_quote": "exact opening words",
    "end_quote": "exact closing words",
    "proposed_start_line": "L0002",
    "proposed_end_line": "L0006"
  },
  "title": "...",
  "hook": "...",
  "caption": "...",
  "hashtags": ["#tag"],
  "summary": "...",
  "why": "...",
  "moment_type": "story",
  "scores": {"hook": 8, "standalone": 9, "payoff": 8, "overall": 84},
  "selection_score": 84,
  "transcript": "full transcript text covered by the clip",
  "warnings": []
}
```

`analysis.json` also contains `rejected`, `dropped`, `coverage`, `responses`,
`counts`, and `warnings`. `render.py` uses only the validated `clips` ranges.

## Selection consequences

- A `standalone` score below 4 drops the candidate before ranking, even when
  `overall` is high. This is a context safety floor, not a quality target.
- An `overall` score below the run's `min_score` (default 60) is dropped.
- Temporal overlap >=35% of the shorter clip, or transcript similarity >=0.82,
  is treated as a duplicate; the higher-ranked candidate wins.
- The highest-ranked `max_clips` survive. Scores are model estimates, so a
  provider that is systematically generous or harsh can shift the selection.
- Re-run `import` with a different `--min-score` or `--max-clips` to change the
  selection without repeating AI calls.

## Versioning

- Provider contract version: `1.0`.
- Run state version: `1.0`.
- Internal analysis version: `1.0`.

Additive fields may appear in 1.x. Unknown candidate fields are ignored.
Breaking field/meaning changes require a new major version and an explicit
migration; the engine rejects an unsupported major rather than guessing.
