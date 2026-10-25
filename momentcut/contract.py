"""Provider-neutral response parser and candidate structural validation.

Canonical provider output is documented in docs/JSON_CONTRACT.md. A small set
of aliases is accepted so a provider that returns the older OpenShorts-style
field names can still be imported; timestamps are never taken from the model.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .util import (
    MomentCutError,
    clean_text,
    coerce_int,
    normalize_for_match,
    tokens_for_match,
    truncate_chars,
)

MOMENT_TYPES = (
    "story",
    "insight",
    "humor",
    "reaction",
    "emotion",
    "conflict",
    "demo",
    "other",
)
MAX_RESPONSE_BYTES = 10 * 1024 * 1024
MAX_CANDIDATES_PER_RESPONSE = 100


class ContractError(MomentCutError):
    pass


class CandidateError(ContractError):
    pass


@dataclass
class Candidate:
    raw: dict[str, Any]
    block_id: str | None
    start_ref: Any
    end_ref: Any
    start_quote: str
    end_quote: str
    title: str
    hook: str
    caption: str
    hashtags: list[str]
    summary: str
    why: str
    moment_type: str
    scores: dict[str, int]
    warnings: list[str] = field(default_factory=list)
    source_file: str = ""

    def anchor_summary(self) -> dict[str, Any]:
        return {
            "start_line": _display_ref(self.start_ref),
            "end_line": _display_ref(self.end_ref),
            "start_quote": self.start_quote,
            "end_quote": self.end_quote,
        }


@dataclass
class ParsedResponse:
    source_file: str
    block_id: str | None
    candidates: list[Candidate]
    notes: list[str]
    raw_payload: Any
    errors: list[str] = field(default_factory=list)


def _display_ref(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def strip_code_fences(text: str) -> str:
    text = (text or "").replace("\ufeff", "").strip()
    fenced = re.findall(r"```(?:json|JSON)?\s*(.*?)```", text, flags=re.DOTALL)
    if fenced:
        # Prefer the largest fenced block: providers sometimes show a tiny
        # example before the real answer.
        return max(fenced, key=len).strip()
    return text


def parse_json_payload(text: str) -> Any:
    """Parse JSON surrounded by code fences or provider commentary."""
    cleaned = strip_code_fences(text)
    if not cleaned:
        raise ContractError("Response is empty.")
    attempts: list[str] = [cleaned]
    if cleaned.startswith(("{", "[")):
        attempts.append(cleaned)
    start_positions = [pos for pos in (cleaned.find("{"), cleaned.find("[")) if pos >= 0]
    if start_positions:
        start = min(start_positions)
        decoder = json.JSONDecoder()
        try:
            value, _ = decoder.raw_decode(cleaned[start:])
            return value
        except json.JSONDecodeError:
            pass
        # Fall back to the outer braces/brackets, which handles trailing prose.
        end_chars = [pos for pos in (cleaned.rfind("}"), cleaned.rfind("]")) if pos >= 0]
        if end_chars:
            end = max(end_chars)
            if end > start:
                attempts.append(cleaned[start : end + 1])
    for attempt in attempts:
        try:
            return json.loads(attempt)
        except json.JSONDecodeError:
            continue
    # One narrow repair: trailing commas before a close. Do not attempt to
    # repair quotes, truncation, or arbitrary prose.
    repaired = re.sub(r",\s*([}\]])", r"\1", attempts[-1])
    if repaired != attempts[-1]:
        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            pass
    raise ContractError(
        "Could not find a JSON object or array in the response. The prompt asks "
        "for JSON only; save the model's JSON answer without commentary."
    )


def _looks_like_provider_payload(payload: dict[str, Any]) -> bool:
    return any(key in payload for key in ("clips", "shorts", "content", "choices", "results"))


def _extract_chat_content(payload: dict[str, Any]) -> str | None:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    first = choices[0]
    if not isinstance(first, dict):
        return None
    message = first.get("message")
    if not isinstance(message, dict):
        return None
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                value = part.get("text") or part.get("content")
                if isinstance(value, str):
                    parts.append(value)
        return "".join(parts)
    return None


def _parse_nested(payload: Any, depth: int = 0) -> Any:
    if depth > 2:
        return payload
    if isinstance(payload, str):
        return parse_json_payload(payload)
    if not isinstance(payload, dict):
        return payload
    if "choices" in payload:
        content = _extract_chat_content(payload)
        if content is not None:
            return _parse_nested(content, depth + 1)
    if "content" in payload and not any(key in payload for key in ("clips", "shorts")):
        content = payload.get("content")
        if isinstance(content, (str, dict, list)):
            inner = _parse_nested(content, depth + 1)
        else:
            inner = payload
        if isinstance(inner, dict) and payload.get("block_id") and not inner.get("block_id"):
            inner = dict(inner)
            inner["block_id"] = payload["block_id"]
        return inner
    return payload


def _normalise_block_id(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().upper()
    match = re.fullmatch(r"B?0*(\d{1,4})", text)
    if match:
        return f"B{int(match.group(1)):03d}"
    return text or None


def _filename_block_hint(path: Path) -> str | None:
    match = re.search(r"(?:^|[_-])B?(\d{1,4})(?:[_.-]|$)", path.stem, flags=re.IGNORECASE)
    if not match:
        return None
    return f"B{int(match.group(1)):03d}"


def _first(candidate: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in candidate and candidate[key] not in (None, ""):
            return candidate[key]
    return default


def normalise_line_ref(value: Any) -> str | None:
    """Return a canonical 'L0001'-like reference, without checking existence."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        if value <= 0:
            return None
        return f"L{value:04d}"
    text = str(value).strip().upper().replace("LINE", "L").replace(" ", "")
    match = re.fullmatch(r"L?0*(\d{1,6})", text)
    if not match:
        return None
    number = int(match.group(1))
    if number <= 0:
        return None
    return f"L{number:04d}"


def normalise_hashtags(value: Any) -> list[str]:
    raw_items: list[str] = []
    if isinstance(value, str):
        # A hashtag field given as one string usually looks like "#ai #video".
        # Only explicit hash tokens are trusted; never turn caption prose into
        # hashtags.
        raw_items = re.findall(r"#([\w\u00c0-\uffff]+)", value, flags=re.UNICODE)
        if not raw_items and "," not in value and len(value.split()) <= 8:
            raw_items = [part for part in re.split(r"[\s,]+", value) if part]
    elif isinstance(value, list):
        for item in value:
            if not isinstance(item, str):
                continue
            explicit = re.findall(r"#([\w\u00c0-\uffff]+)", item, flags=re.UNICODE)
            if explicit:
                raw_items.extend(explicit)
            else:
                raw_items.extend(part for part in re.split(r"[\s,]+", item) if part)
    result: list[str] = []
    seen: set[str] = set()
    for item in raw_items:
        tag = re.sub(r"[^\w\u00c0-\uffff]", "", item, flags=re.UNICODE)
        if not tag:
            continue
        tag = "#" + tag[:40]
        key = tag.casefold()
        if key not in seen:
            seen.add(key)
            result.append(tag)
    return result[:10]


def hashtags_from_caption(caption: str) -> list[str]:
    return normalise_hashtags(" ".join(re.findall(r"#[\w\u00c0-\uffff]+", caption, flags=re.UNICODE)))


def _normalise_moment_type(value: Any) -> str:
    text = normalize_for_match(str(value or ""))
    aliases = {
        "funny": "humor",
        "comedy": "humor",
        "joke": "humor",
        "educational": "insight",
        "lesson": "insight",
        "advice": "insight",
        "tutorial": "demo",
        "demonstration": "demo",
        "emotional": "emotion",
        "argument": "conflict",
        "disagreement": "conflict",
        "reaction": "reaction",
        "story": "story",
        "insight": "insight",
        "humor": "humor",
        "emotion": "emotion",
        "conflict": "conflict",
        "demo": "demo",
        "other": "other",
    }
    return aliases.get(text, "other")


def _normalise_scores(candidate: dict[str, Any]) -> tuple[dict[str, int], list[str]]:
    warnings: list[str] = []
    raw_scores = candidate.get("scores")
    if not isinstance(raw_scores, dict):
        raw_scores = {}
    overall = coerce_int(raw_scores.get("overall"))
    if overall is None:
        overall = coerce_int(candidate.get("predicted_score"))
    dimensions = {
        "hook": coerce_int(raw_scores.get("hook")),
        "standalone": coerce_int(raw_scores.get("standalone")),
        "payoff": coerce_int(raw_scores.get("payoff")),
    }
    present = [value for value in dimensions.values() if value is not None]
    if overall is None:
        if present:
            overall = int(round((sum(present) / len(present)) * 10))
            warnings.append("overall score was missing; derived from dimension scores")
        else:
            raise CandidateError("missing scores.overall")
    overall = max(0, min(100, int(overall)))
    result = {"overall": overall}
    for name, value in dimensions.items():
        if value is None:
            result[name] = max(0, min(10, int(round(overall / 10))))
            warnings.append(f"scores.{name} was missing; derived from overall")
        else:
            result[name] = max(0, min(10, int(value)))
    return {
        "hook": result["hook"],
        "standalone": result["standalone"],
        "payoff": result["payoff"],
        "overall": overall,
    }, warnings


def parse_candidate(raw: Any, *, default_block_id: str | None = None) -> Candidate:
    if not isinstance(raw, dict):
        raise CandidateError("candidate is not a JSON object")
    start_ref = _first(raw, "start_line", "start_line_id", "start_unit", "start_id")
    end_ref = _first(raw, "end_line", "end_line_id", "end_unit", "end_id")
    start_quote = clean_text(_first(raw, "start_quote", "start_text", default=""))
    end_quote = clean_text(_first(raw, "end_quote", "end_text", default=""))
    start_tokens = tokens_for_match(start_quote)
    end_tokens = tokens_for_match(end_quote)
    if start_ref is None and len(start_tokens) < 3:
        raise CandidateError("missing or too-short start anchor")
    if end_ref is None and len(end_tokens) < 3:
        raise CandidateError("missing or too-short end anchor")
    if start_ref is None and end_ref is None:
        raise CandidateError("no line anchors")

    title = clean_text(
        _first(raw, "title", "video_title_for_youtube_short", "video_title", default="")
    )
    hook = clean_text(_first(raw, "hook", "viral_hook_text", default=""))
    caption = clean_text(
        _first(
            raw,
            "caption",
            "video_description_for_tiktok",
            "video_description_for_instagram",
            "description",
            default="",
        )
    )
    summary = clean_text(_first(raw, "summary", default=""))
    why = clean_text(_first(raw, "why", "reason", default=""))
    hashtags = normalise_hashtags(raw.get("hashtags"))
    if not hashtags and caption:
        hashtags = hashtags_from_caption(caption)
    moment_type = _normalise_moment_type(_first(raw, "moment_type", "type", default="other"))
    scores, score_warnings = _normalise_scores(raw)
    warnings: list[str] = list(score_warnings)

    block_id = _normalise_block_id(_first(raw, "block_id", default=default_block_id))
    language = clean_text(raw.get("language", ""))
    if language:
        pass  # Preserved in raw; the transcript language is authoritative downstream.

    return Candidate(
        raw=raw,
        block_id=block_id,
        start_ref=start_ref,
        end_ref=end_ref,
        start_quote=start_quote,
        end_quote=end_quote,
        title=title,
        hook=hook,
        caption=caption,
        hashtags=hashtags,
        summary=summary,
        why=why,
        moment_type=moment_type,
        scores=scores,
        warnings=warnings,
    )


def parse_response_payload(
    payload: Any,
    *,
    source_file: str = "",
    filename_hint: str | None = None,
) -> ParsedResponse:
    notes: list[str] = []
    parsed = _parse_nested(payload)
    if isinstance(parsed, list):
        parsed = {"clips": parsed}
    if not isinstance(parsed, dict):
        raise ContractError("The answer is not a JSON object or array.")

    if "results" in parsed and isinstance(parsed.get("results"), list):
        raise ContractError(
            "This looks like a multi-block wrapper. Import one answer per block "
            "instead; 'results' is not part of schema 1.0."
        )

    schema_version = str(parsed.get("schema_version") or "").strip()
    if schema_version and schema_version.split(".")[0] != "1":
        raise ContractError(
            f"Unsupported response schema_version {schema_version!r}; this importer "
            "expects 1.x."
        )
    if not schema_version:
        notes.append("schema_version was missing; accepted as 1.0")

    clips = parsed.get("clips", parsed.get("shorts"))
    if clips is None:
        raise ContractError(
            "The answer has no 'clips' array. Expected an object with "
            "'schema_version', 'block_id', and 'clips'."
        )
    if not isinstance(clips, list):
        raise ContractError("'clips' must be an array.")
    if len(clips) > MAX_CANDIDATES_PER_RESPONSE:
        notes.append(
            f"response contained {len(clips)} candidates; only the first "
            f"{MAX_CANDIDATES_PER_RESPONSE} were considered"
        )
        clips = clips[:MAX_CANDIDATES_PER_RESPONSE]

    block_id = _normalise_block_id(parsed.get("block_id")) or filename_hint
    if not block_id:
        notes.append("block_id was missing; inferred later from line IDs")
    candidates: list[Candidate] = []
    errors: list[str] = []
    for index, raw in enumerate(clips, start=1):
        try:
            candidate = parse_candidate(raw, default_block_id=block_id)
        except CandidateError as exc:
            errors.append(f"candidate #{index}: {exc}")
            continue
        candidate.source_file = source_file
        candidates.append(candidate)
    return ParsedResponse(
        source_file=source_file,
        block_id=block_id,
        candidates=candidates,
        notes=notes,
        raw_payload=parsed,
        errors=errors,
    )


def load_response_file(path: str | Path) -> ParsedResponse:
    source = Path(path)
    if not source.exists():
        raise ContractError(f"Response file not found: {source}")
    size = source.stat().st_size
    if size > MAX_RESPONSE_BYTES:
        raise ContractError(
            f"Response file {source.name} is {size / 1024 / 1024:.1f} MB; "
            "refusing to parse more than 10 MB."
        )
    text = source.read_text(encoding="utf-8-sig", errors="replace")
    payload = parse_json_payload(text)
    return parse_response_payload(
        payload,
        source_file=str(source),
        filename_hint=_filename_block_hint(source),
    )


def validate_metadata(candidate: Candidate, clip_text: str) -> tuple[dict[str, Any], list[str]]:
    """Normalise social copy after boundaries are known.

    Missing copy is repaired from the transcript, with a warning. Bad timing is
    never repaired from metadata; it is rejected by the resolver instead.
    """
    warnings = list(candidate.warnings)
    title = clean_text(candidate.title)
    hook = clean_text(candidate.hook)
    caption = clean_text(candidate.caption)
    summary = clean_text(candidate.summary)
    why = clean_text(candidate.why)

    if not title:
        title = _sentence_from_text(clip_text) or "Untitled moment"
        warnings.append("title was missing; derived from the clip transcript")
    if len(title) > 100:
        title = truncate_chars(title, 100)
        warnings.append("title was truncated to 100 characters")
    if not hook:
        hook = " ".join(title.split()[:10])
        warnings.append("hook was missing; derived from the title")
    if len(hook) > 80 or len(hook.split()) > 10:
        hook = " ".join(hook.split()[:10])
        hook = truncate_chars(hook, 80)
        warnings.append("hook was truncated to 10 words / 80 characters")
    if not summary:
        summary = _sentence_from_text(clip_text)
        if summary:
            warnings.append("summary was missing; derived from the clip transcript")
    if not caption:
        caption = summary or title
        warnings.append("caption was missing; derived from the summary")
    if len(caption) > 1000:
        caption = truncate_chars(caption, 1000)
        warnings.append("caption was truncated to 1000 characters")
    if not why:
        why = "No editorial rationale was supplied."
        warnings.append("why was missing")
    if len(why) > 250:
        why = truncate_chars(why, 250)
        warnings.append("why was truncated to 250 characters")

    hashtags = list(candidate.hashtags)
    if not hashtags and caption:
        hashtags = hashtags_from_caption(caption)
    return {
        "title": title,
        "hook": hook,
        "caption": caption,
        "hashtags": hashtags,
        "summary": summary,
        "why": why,
        "moment_type": candidate.moment_type,
    }, warnings


def _sentence_from_text(text: str) -> str:
    text = clean_text(text)
    if not text:
        return ""
    parts = re.split(r"(?<=[.!?…。？！])\s+", text)
    result = parts[0] if parts else text
    return truncate_chars(result.strip(), 220)


def candidate_debug(candidate: Candidate) -> dict[str, Any]:
    return {
        "block_id": candidate.block_id,
        "anchors": candidate.anchor_summary(),
        "title": candidate.title,
        "scores": candidate.scores,
        "raw": candidate.raw,
    }


def response_contract_example() -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "block_id": "B001",
        "clips": [
            {
                "start_line": "L0002",
                "end_line": "L0005",
                "start_quote": "exact words at the start",
                "end_quote": "exact words at the end",
                "title": "Concrete title",
                "hook": "Short hook",
                "caption": "One or two sentences.",
                "hashtags": ["#tag"],
                "summary": "One sentence.",
                "why": "Specific reason.",
                "moment_type": "story",
                "scores": {
                    "hook": 8,
                    "standalone": 9,
                    "payoff": 8,
                    "overall": 84,
                },
            }
        ],
    }
