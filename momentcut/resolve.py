"""Resolve model-selected line IDs and quotes to safe source seconds.

This is the validation boundary between untrusted AI output and FFmpeg. A
candidate reaches this module only as structured data; it leaves either as a
clip whose range is built from transcript timings, or as a rejection reason.
"""

from __future__ import annotations

import difflib
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from .config import RunSettings
from .contract import Candidate, CandidateError, ParsedResponse, validate_metadata
from .prompting import Block, block_by_id
from .transcript import Transcript
from .util import (
    MomentCutError,
    clean_text,
    tokens_for_match,
)

START_LEAD_SECONDS = 0.20
END_TAIL_SECONDS = 0.35
QUOTE_DECLARED_THRESHOLD = 0.70
QUOTE_REPAIR_THRESHOLD = 0.85
QUOTE_WEAK_ACCEPT_THRESHOLD = 0.50
MAX_EXTENSION_GAP_SECONDS = 1.5


class ResolutionError(MomentCutError):
    pass


@dataclass
class ResolvedClip:
    candidate: Candidate
    block_id: str | None
    start_index: int
    end_index: int
    start: float
    end: float
    duration: float
    start_line: str
    end_line: str
    transcript_text: str
    metadata: dict[str, Any]
    warnings: list[str] = field(default_factory=list)
    selection_score: int = 0
    clip_id: str = ""
    selection_rank: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.clip_id,
            "selection_rank": self.selection_rank,
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "duration": round(self.duration, 3),
            "start_line": self.start_line,
            "end_line": self.end_line,
            "block_id": self.block_id,
            "anchors": {
                "start_line": self.start_line,
                "end_line": self.end_line,
                "start_quote": self.candidate.start_quote,
                "end_quote": self.candidate.end_quote,
                "proposed_start_line": _ref_text(self.candidate.start_ref),
                "proposed_end_line": _ref_text(self.candidate.end_ref),
            },
            "title": self.metadata["title"],
            "hook": self.metadata["hook"],
            "caption": self.metadata["caption"],
            "hashtags": self.metadata["hashtags"],
            "summary": self.metadata["summary"],
            "why": self.metadata["why"],
            "moment_type": self.metadata["moment_type"],
            "scores": self.candidate.scores,
            "selection_score": self.selection_score,
            "transcript": self.transcript_text,
            "warnings": self.warnings,
        }


@dataclass
class RejectedClip:
    source_file: str
    block_id: str | None
    reason: str
    candidate: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_file": self.source_file,
            "block_id": self.block_id,
            "reason": self.reason,
            "candidate": self.candidate,
        }


def _ref_text(value: Any) -> str:
    return "" if value is None else str(value)


def _line_number(value: str) -> int | None:
    match = re.search(r"(\d+)$", value or "")
    if not match:
        return None
    return int(match.group(1))


def _line_index_maps(transcript: Transcript) -> tuple[dict[str, int], dict[int, int]]:
    exact: dict[str, int] = {}
    numeric: dict[int, int] = {}
    for index, unit in enumerate(transcript.units):
        exact[unit.id.upper()] = index
        number = _line_number(unit.id)
        if number is not None and number not in numeric:
            numeric[number] = index
    return exact, numeric


def find_line_index(transcript: Transcript, reference: Any) -> int | None:
    if reference is None or isinstance(reference, bool):
        return None
    text = str(reference).strip().upper().replace(" ", "")
    text = text.replace("LINE", "L")
    exact, numeric = _line_index_maps(transcript)
    if text in exact:
        return exact[text]
    match = re.fullmatch(r"L?0*(\d{1,6})", text)
    if match:
        number = int(match.group(1))
        if number in numeric:
            return numeric[number]
    if isinstance(reference, int) and reference in numeric:
        return numeric[reference]
    return None


def _quote_match_score(unit_text: str, quote: str) -> tuple[float, str]:
    quote_tokens = tokens_for_match(quote)
    if not quote_tokens:
        return 0.0, "empty"
    unit_tokens = tokens_for_match(unit_text)
    if not unit_tokens:
        return 0.0, "empty-unit"
    normalized_quote = " ".join(quote_tokens)
    normalized_unit = " ".join(unit_tokens)
    if len(quote_tokens) >= 3 and normalized_quote in normalized_unit:
        if len(quote_tokens) < 5:
            return 1.0, "exact-short"
        return 1.0, "exact"
    if len(quote_tokens) < 3:
        return 0.0, "too-short"
    matcher = difflib.SequenceMatcher(None, unit_tokens, quote_tokens, autojunk=False)
    matched = sum(block.size for block in matcher.get_matching_blocks())
    ordered_ratio = matched / max(1, len(quote_tokens))
    # A short quote can occur inside a much longer unit. The ordered match is
    # the right measure; the plain ratio would punish the extra unit context.
    if len(quote_tokens) < 5:
        return (1.0 if ordered_ratio == 1.0 else ordered_ratio * 0.6), "short"
    return ordered_ratio, "fuzzy"


def _scope_for_block(block: Block | None, transcript: Transcript) -> tuple[int, int]:
    if block is None:
        return 0, len(transcript.units) - 1
    return block.start_index, block.end_index


def _search_quote(
    transcript: Transcript,
    quote: str,
    scope: tuple[int, int],
    *,
    preferred: int | None = None,
) -> list[tuple[float, int]]:
    low, high = scope
    scored: list[tuple[float, int]] = []
    for index in range(low, high + 1):
        score, _ = _quote_match_score(transcript.units[index].text, quote)
        if score > 0:
            scored.append((score, index))
    scored.sort(key=lambda item: (-item[0], abs(item[1] - (preferred if preferred is not None else item[1]))))
    return scored


def _resolve_boundary(
    transcript: Transcript,
    block: Block | None,
    reference: Any,
    quote: str,
    *,
    kind: str,
    warnings: list[str],
) -> int:
    scope = _scope_for_block(block, transcript)
    low, high = scope
    ref_index = find_line_index(transcript, reference)
    quote_tokens = tokens_for_match(quote)

    if ref_index is not None and not (low <= ref_index <= high):
        raise ResolutionError(
            f"{kind} line {_ref_text(reference)!r} is outside block "
            f"{block.id if block else '(unknown)'} ({block.start_line if block else '?'}"
            f"–{block.end_line if block else '?'})."
        )

    if ref_index is not None:
        if len(quote_tokens) < 3:
            warnings.append(f"{kind} quote was missing or too short; line ID used")
            return ref_index
        score, how = _quote_match_score(transcript.units[ref_index].text, quote)
        if score >= QUOTE_DECLARED_THRESHOLD:
            if how == "short" and score < 1.0:
                warnings.append(f"{kind} quote only partially matched its line ID")
            return ref_index
        # The line ID and the quote disagree. A small local search can repair a
        # clear off-by-one without accepting a quote from the other side of the
        # video.
        nearby_low = max(low, ref_index - 3)
        nearby_high = min(high, ref_index + 3)
        nearby = _search_quote(transcript, quote, (nearby_low, nearby_high), preferred=ref_index)
        if nearby and nearby[0][0] >= QUOTE_REPAIR_THRESHOLD:
            candidate_index = nearby[0][1]
            if candidate_index != ref_index:
                warnings.append(
                    f"{kind} line ID {transcript.units[ref_index].id} disagreed with the "
                    f"quote; repaired to {transcript.units[candidate_index].id}"
                )
            return candidate_index
        if score >= QUOTE_WEAK_ACCEPT_THRESHOLD:
            warnings.append(f"{kind} quote is a weak match to its line ID; line ID used")
            return ref_index
        raise ResolutionError(
            f"{kind} quote does not match line {transcript.units[ref_index].id} "
            f"(similarity {score:.2f}) and no nearby line matches it."
        )

    if len(quote_tokens) < 3:
        raise ResolutionError(
            f"{kind} has no valid line ID and no usable quote anchor."
        )
    scored = _search_quote(transcript, quote, scope)
    if not scored:
        raise ResolutionError(f"{kind} quote was not found in the block transcript.")
    best_score, best_index = scored[0]
    if best_score < QUOTE_REPAIR_THRESHOLD:
        raise ResolutionError(
            f"{kind} quote did not match a transcript line closely enough "
            f"(best similarity {best_score:.2f})."
        )
    if len(scored) > 1:
        second_score = scored[1][0]
        margin = best_score - second_score
        if margin < 0.06 and second_score >= best_score - 0.05:
            raise ResolutionError(
                f"{kind} quote is ambiguous: two lines match at "
                f"{best_score:.2f}/{second_score:.2f}."
            )
    warnings.append(
        f"{kind} line ID was missing; repaired from quote to {transcript.units[best_index].id}"
    )
    return best_index


def _infer_block(
    candidate: Candidate,
    transcript: Transcript,
    blocks: Sequence[Block],
) -> tuple[Block | None, list[str]]:
    warnings: list[str] = []
    explicit = block_by_id(list(blocks), candidate.block_id) if candidate.block_id else None
    if candidate.block_id and explicit is None:
        warnings.append(f"unknown block_id {candidate.block_id!r}; inferred from line IDs")

    ref_index = find_line_index(transcript, candidate.start_ref)
    end_index = find_line_index(transcript, candidate.end_ref)
    containing: list[Block] = []
    for block in blocks:
        if ref_index is None:
            continue
        if block.start_index <= ref_index <= block.end_index:
            if end_index is None or block.start_index <= end_index <= block.end_index:
                containing.append(block)
    if explicit is not None:
        # Trust the declared block for the search scope; the boundary resolver
        # rejects lines outside it with a precise reason.
        return explicit, warnings
    if containing:
        return containing[0], warnings
    if ref_index is None and end_index is None and candidate.block_id:
        return explicit, warnings
    if not blocks:
        return None, warnings
    warnings.append("candidate could not be mapped to a specific block; searched all lines")
    return None, warnings


def _block_units(transcript: Transcript, block: Block | None) -> tuple[int, int]:
    return _scope_for_block(block, transcript)


def _resolve_duration(
    transcript: Transcript,
    start_index: int,
    end_index: int,
    block: Block | None,
    source_duration: float,
    settings: RunSettings,
) -> tuple[int, int, float, float, list[str]]:
    warnings: list[str] = []
    low, high = _block_units(transcript, block)
    if start_index > end_index:
        raise ResolutionError("start line is after end line.")

    effective_min = min(float(settings.min_duration), source_duration)
    effective_max = min(float(settings.max_duration), source_duration)
    if effective_max <= 0:
        raise ResolutionError("source duration is zero.")

    def times_for(first: int, last: int) -> tuple[float, float]:
        start = max(0.0, transcript.units[first].start - START_LEAD_SECONDS)
        end = min(source_duration, transcript.units[last].end + END_TAIL_SECONDS)
        return round(start, 3), round(end, 3)

    current_start = max(low, start_index)
    current_end = min(high, end_index)
    start, end = times_for(current_start, current_end)
    duration = end - start

    while duration < effective_min and current_end + 1 <= high:
        next_unit = transcript.units[current_end + 1]
        gap = max(0.0, next_unit.start - transcript.units[current_end].end)
        if gap > MAX_EXTENSION_GAP_SECONDS:
            break
        candidate_end = times_for(current_start, current_end + 1)[1]
        if candidate_end - start > effective_max:
            break
        current_end += 1
        end = candidate_end
        duration = end - start
        warnings.append("clip was extended at the end to satisfy the minimum duration")

    while duration < effective_min and current_start - 1 >= low:
        previous_unit = transcript.units[current_start - 1]
        gap = max(0.0, transcript.units[current_start].start - previous_unit.end)
        if gap > MAX_EXTENSION_GAP_SECONDS:
            break
        candidate_start = times_for(current_start - 1, current_end)[0]
        if end - candidate_start > effective_max:
            break
        current_start -= 1
        start = candidate_start
        duration = end - start
        warnings.append("clip was extended at the start to satisfy the minimum duration")

    # A source shorter than the requested minimum has no more material to add.
    # Return the whole source rather than rejecting a valid complete video.
    if (
        duration < effective_min - 0.01
        and current_start == low
        and current_end == high
        and high == len(transcript.units) - 1
        and low == 0
    ):
        start = max(0.0, transcript.units[current_start].start - START_LEAD_SECONDS)
        end = source_duration
        duration = end - start
        warnings.append("source is shorter than the requested minimum; whole source used")

    if duration < effective_min - 0.01:
        raise ResolutionError(
            f"clip is {duration:.2f}s after boundary repair; minimum is "
            f"{effective_min:.2f}s and the next usable line would exceed the maximum."
        )

    unit_span = transcript.units[current_end].end - transcript.units[current_start].start
    if unit_span > effective_max + 0.01:
        raise ResolutionError(
            f"clip content span is {unit_span:.2f}s, above the maximum "
            f"{effective_max:.2f}s."
        )

    if duration > effective_max + 0.001:
        # Only silence/padding around the chosen lines may be trimmed. First
        # remove the tail pad, then the lead pad; never remove a line.
        excess = duration - effective_max
        end = max(transcript.units[current_end].end, end - excess)
        duration = end - start
        if duration > effective_max + 0.001:
            start = max(transcript.units[current_start].start, end - effective_max)
            duration = end - start
        warnings.append("clip padding was trimmed to fit the maximum duration")
    if duration > effective_max + 0.06:
        raise ResolutionError(
            f"clip is {duration:.2f}s, above the maximum {effective_max:.2f}s."
        )
    if duration > effective_max:
        # A numeric epsilon only; the lines themselves already fit.
        duration = effective_max
        end = round(start + duration, 3)

    if start < 0 or end > source_duration + 0.05:
        raise ResolutionError("resolved clip falls outside the source duration.")
    if end <= start:
        raise ResolutionError("resolved clip has a non-positive duration.")
    return current_start, current_end, round(start, 3), round(end, 3), warnings


def resolve_candidate(
    candidate: Candidate,
    transcript: Transcript,
    blocks: Sequence[Block],
    source_duration: float,
    settings: RunSettings,
) -> ResolvedClip:
    warnings = list(candidate.warnings)
    block, block_warnings = _infer_block(candidate, transcript, blocks)
    warnings.extend(block_warnings)
    start_index = _resolve_boundary(
        transcript,
        block,
        candidate.start_ref,
        candidate.start_quote,
        kind="start",
        warnings=warnings,
    )
    end_index = _resolve_boundary(
        transcript,
        block,
        candidate.end_ref,
        candidate.end_quote,
        kind="end",
        warnings=warnings,
    )
    if start_index > end_index:
        raise ResolutionError(
            f"start line {transcript.units[start_index].id} is after end line "
            f"{transcript.units[end_index].id}."
        )

    start_index, end_index, start, end, duration_warnings = _resolve_duration(
        transcript,
        start_index,
        end_index,
        block,
        source_duration,
        settings,
    )
    warnings.extend(duration_warnings)
    text = transcript.text_between(start_index, end_index)
    metadata, metadata_warnings = validate_metadata(candidate, text)
    warnings.extend(metadata_warnings)
    # Warnings do not silently change the model's ranking; they are surfaced
    # for review. The score is the model's overall estimate.
    return ResolvedClip(
        candidate=candidate,
        block_id=block.id if block else candidate.block_id,
        start_index=start_index,
        end_index=end_index,
        start=start,
        end=end,
        duration=round(end - start, 3),
        start_line=transcript.units[start_index].id,
        end_line=transcript.units[end_index].id,
        transcript_text=text,
        metadata=metadata,
        warnings=warnings,
        selection_score=int(candidate.scores["overall"]),
    )


def resolve_responses(
    responses: Sequence[ParsedResponse],
    transcript: Transcript,
    blocks: Sequence[Block],
    source_duration: float,
    settings: RunSettings,
) -> tuple[list[ResolvedClip], list[RejectedClip]]:
    resolved: list[ResolvedClip] = []
    rejected: list[RejectedClip] = []
    for response in responses:
        for error in response.errors:
            rejected.append(
                RejectedClip(
                    source_file=response.source_file,
                    block_id=response.block_id,
                    reason=error,
                )
            )
        for candidate in response.candidates:
            try:
                resolved.append(
                    resolve_candidate(
                        candidate,
                        transcript,
                        blocks,
                        source_duration,
                        settings,
                    )
                )
            except (ResolutionError, CandidateError) as exc:
                rejected.append(
                    RejectedClip(
                        source_file=response.source_file,
                        block_id=candidate.block_id,
                        reason=str(exc),
                        candidate={
                            "anchors": candidate.anchor_summary(),
                            "title": candidate.title,
                            "scores": candidate.scores,
                            "raw": candidate.raw,
                        },
                    )
                )
    return resolved, rejected


def line_span_text(transcript: Transcript, start_index: int, end_index: int) -> str:
    return clean_text(" ".join(unit.text for unit in transcript.units[start_index : end_index + 1]))
