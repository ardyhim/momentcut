"""Quality gate, duplicate suppression, and final clip selection."""

from __future__ import annotations

import difflib
from collections.abc import Sequence
from dataclasses import dataclass, field

from .config import RunSettings
from .resolve import ResolvedClip
from .util import format_clock, tokens_for_match

OVERLAP_DUPLICATE_RATIO = 0.35
TEXT_DUPLICATE_RATIO = 0.82
MIN_STANDALONE_SCORE = 4


@dataclass
class DroppedClip:
    clip: ResolvedClip
    reason: str
    kept_clip: str | None = None

    def to_dict(self) -> dict:
        return {
            "title": self.clip.metadata.get("title", "") if self.clip.metadata else "",
            "start": round(self.clip.start, 3),
            "end": round(self.clip.end, 3),
            "duration": round(self.clip.duration, 3),
            "score": self.clip.selection_score,
            "block_id": self.clip.block_id,
            "reason": self.reason,
            "kept_clip": self.kept_clip,
        }


@dataclass
class SelectionResult:
    selected: list[ResolvedClip] = field(default_factory=list)
    dropped: list[DroppedClip] = field(default_factory=list)
    below_score: int = 0
    low_context: int = 0
    duplicate_count: int = 0
    over_limit: int = 0

    def to_dict(self) -> dict:
        return {
            "selected_count": len(self.selected),
            "below_min_score": self.below_score,
            "low_standalone": self.low_context,
            "duplicates_removed": self.duplicate_count,
            "over_max_clips": self.over_limit,
            "dropped": [item.to_dict() for item in self.dropped],
        }


def _rank_key(clip: ResolvedClip) -> tuple:
    scores = clip.candidate.scores
    return (
        -int(clip.selection_score),
        -int(scores.get("standalone", 0)),
        -int(scores.get("hook", 0)),
        -int(scores.get("payoff", 0)),
        float(clip.start),
        float(clip.end),
    )


def _overlap_ratio(first: ResolvedClip, second: ResolvedClip) -> float:
    overlap = min(first.end, second.end) - max(first.start, second.start)
    if overlap <= 0:
        return 0.0
    shorter = max(1e-6, min(first.duration, second.duration))
    return overlap / shorter


def _text_similarity(first: ResolvedClip, second: ResolvedClip) -> float:
    first_tokens = tokens_for_match(first.transcript_text)
    second_tokens = tokens_for_match(second.transcript_text)
    if not first_tokens or not second_tokens:
        return 0.0
    if len(first_tokens) < 8 or len(second_tokens) < 8:
        # Tiny spans are unreliable for sequence similarity. Temporal overlap
        # is the stronger signal there.
        first_set = set(first_tokens)
        second_set = set(second_tokens)
        union = first_set | second_set
        return len(first_set & second_set) / len(union) if union else 0.0
    matcher = difflib.SequenceMatcher(
        None, first_tokens, second_tokens, autojunk=False
    )
    return matcher.ratio()


def _duplicate_reason(
    first: ResolvedClip, second: ResolvedClip
) -> tuple[bool, str]:
    overlap = _overlap_ratio(first, second)
    if overlap >= OVERLAP_DUPLICATE_RATIO:
        return True, (
            f"overlaps a higher-ranked clip by {overlap:.0%} of the shorter clip "
            f"({format_clock(first.start)}-{format_clock(first.end)})"
        )
    similarity = _text_similarity(first, second)
    if similarity >= TEXT_DUPLICATE_RATIO:
        return True, (
            f"near-duplicate transcript (similarity {similarity:.0%}) of a "
            f"higher-ranked clip"
        )
    return False, ""


def select_clips(
    resolved: Sequence[ResolvedClip],
    settings: RunSettings,
) -> SelectionResult:
    result = SelectionResult()
    qualified: list[ResolvedClip] = []
    for clip in resolved:
        if int(clip.candidate.scores.get("standalone", 0)) < MIN_STANDALONE_SCORE:
            result.low_context += 1
            result.dropped.append(
                DroppedClip(
                    clip=clip,
                    reason=(
                        f"standalone score {clip.candidate.scores.get('standalone')} "
                        f"is below the cold-viewer floor {MIN_STANDALONE_SCORE}"
                    ),
                )
            )
            continue
        if clip.selection_score < settings.min_score:
            result.below_score += 1
            result.dropped.append(
                DroppedClip(
                    clip=clip,
                    reason=(
                        f"overall score {clip.selection_score} is below the "
                        f"minimum {settings.min_score}"
                    ),
                )
            )
        else:
            qualified.append(clip)

    ranked = sorted(qualified, key=_rank_key)
    kept: list[ResolvedClip] = []
    for clip in ranked:
        duplicate_of: ResolvedClip | None = None
        duplicate_reason = ""
        for previous in kept:
            is_duplicate, reason = _duplicate_reason(previous, clip)
            if is_duplicate:
                duplicate_of = previous
                duplicate_reason = reason
                break
        if duplicate_of is not None:
            result.duplicate_count += 1
            kept_label = (
                f"{duplicate_of.block_id or '?'} "
                f"{format_clock(duplicate_of.start)}-{format_clock(duplicate_of.end)}"
            )
            result.dropped.append(
                DroppedClip(
                    clip=clip,
                    reason="duplicate: " + duplicate_reason,
                    kept_clip=kept_label,
                )
            )
            continue
        kept.append(clip)

    selected = kept[: settings.max_clips]
    for clip in kept[settings.max_clips :]:
        result.over_limit += 1
        result.dropped.append(
            DroppedClip(
                clip=clip,
                reason=(
                    f"ranked outside the requested maximum of {settings.max_clips} "
                    "clips"
                ),
            )
        )

    # Number chronologically for usable filenames, but preserve the editorial
    # rank so the user can see which clip scored highest.
    selected = sorted(selected, key=lambda clip: (clip.start, clip.end))
    rank_by_identity = {
        id(clip): rank
        for rank, clip in enumerate(sorted(selected, key=_rank_key), start=1)
    }
    for index, clip in enumerate(selected, start=1):
        clip.clip_id = f"C{index:03d}"
        clip.selection_rank = rank_by_identity[id(clip)]
    result.selected = selected
    return result
