from __future__ import annotations

import copy

import pytest

from momentcut.config import RunSettings
from momentcut.contract import parse_candidate, parse_response_payload
from momentcut.prompting import build_blocks
from momentcut.resolve import ResolutionError, resolve_candidate, resolve_responses
from momentcut.select import select_clips
from momentcut.transcript import Transcript, Unit, load_transcript_json

from .conftest import response_payload, transcript_payload


def setup_transcript(payload=None):
    transcript = load_transcript_json(payload or transcript_payload(), 50.0)
    return transcript


def default_settings(**kwargs):
    values = {
        "min_duration": 8.0,
        "max_duration": 30.0,
        "max_clips": 5,
        "min_score": 60,
        "block_seconds": 120.0,
    }
    values.update(kwargs)
    return RunSettings(**values)


def one_candidate(raw=None):
    payload = copy.deepcopy(response_payload()["clips"][0])
    if raw:
        payload.update(raw)
    return parse_candidate(payload, default_block_id="B001")


def resolve_one(raw=None, settings=None, transcript=None):
    transcript = transcript or setup_transcript()
    settings = settings or default_settings()
    blocks = build_blocks(transcript, block_seconds=120.0, min_overlap_seconds=40.0)
    return resolve_candidate(
        one_candidate(raw),
        transcript,
        blocks,
        transcript.source_duration,
        settings,
    )


class TestBoundaryResolution:
    def test_valid_candidate(self):
        clip = resolve_one()
        assert clip.start == 4.9
        assert clip.end == 30.15
        assert clip.duration == 25.25
        assert clip.start_line == "L0002"
        assert clip.end_line == "L0006"
        assert not clip.warnings

    def test_line_id_and_quote_repair(self):
        clip = resolve_one({"start_quote": "The camera does not make the story"})
        assert clip.start_line == "L0003"
        assert any("repaired" in warning for warning in clip.warnings)

    def test_far_conflicting_quote_rejected(self):
        with pytest.raises(ResolutionError):
            resolve_one(
                {
                    "start_quote": "So here is the exact three-step checklist I use now."
                }
            )

    def test_missing_quote_with_valid_line_is_warned(self):
        clip = resolve_one({"start_quote": "", "end_quote": ""})
        assert clip.start_line == "L0002"
        assert any("quote was missing" in warning for warning in clip.warnings)

    def test_invalid_line_repaired_from_quote(self):
        clip = resolve_one(
            {
                "start_line": "L9999",
                "start_quote": "Then I realized something that changed everything.",
            }
        )
        assert clip.start_line == "L0002"
        assert any("repaired" in warning for warning in clip.warnings)

    def test_invalid_both_rejected(self):
        with pytest.raises(ResolutionError):
            resolve_one(
                {
                    "start_line": "L9999",
                    "start_quote": "A sentence that is definitely not here.",
                }
            )

    def test_start_after_end_rejected(self):
        with pytest.raises(ResolutionError):
            resolve_one({"start_line": "L0006", "end_line": "L0002"})

    def test_out_of_block_rejected(self):
        units = []
        for index in range(20):
            units.append(
                Unit(
                    f"L{index + 1:04d}",
                    index * 10.0,
                    index * 10.0 + 9.8,
                    f"Line {index + 1} says something complete.",
                )
            )
        transcript = Transcript("en", units, "fixture", 200.0)
        blocks = build_blocks(transcript, block_seconds=60.0, min_overlap_seconds=20.0)
        # Pick a line that is not in B001.
        target = None
        for index, unit in enumerate(transcript.units):
            if index > blocks[0].end_index:
                target = unit
                break
        assert target is not None
        raw = copy.deepcopy(response_payload()["clips"][0])
        raw.update(
            {
                "block_id": "B001",
                "start_line": target.id,
                "end_line": target.id,
                "start_quote": target.text,
                "end_quote": target.text,
            }
        )
        candidate = parse_candidate(raw)
        with pytest.raises(ResolutionError):
            resolve_candidate(
                candidate, transcript, blocks, 200.0, default_settings()
            )

    def test_ambiguous_quote_rejected(self):
        units = [
            Unit("L0001", 0.0, 9.0, "The exact same line appears here."),
            Unit("L0002", 10.0, 19.0, "The exact same line appears here."),
        ]
        transcript = Transcript("en", units, "fixture", 20.0)
        blocks = build_blocks(transcript, block_seconds=120.0, min_overlap_seconds=10.0)
        raw = copy.deepcopy(response_payload()["clips"][0])
        raw.update(
            {
                "start_line": None,
                "end_line": "L0002",
                "start_quote": "The exact same line appears here.",
                "end_quote": "The exact same line appears here.",
                "scores": {"overall": 80},
            }
        )
        candidate = parse_candidate(raw)
        with pytest.raises(ResolutionError):
            resolve_candidate(candidate, transcript, blocks, 20.0, default_settings())


class TestDuration:
    def test_too_long_rejected(self):
        with pytest.raises(ResolutionError):
            resolve_one({"start_line": "L0001", "end_line": "L0010"})

    def test_minimum_duration_extends_to_previous_line(self):
        clip = resolve_one(
            {
                "start_line": "L0010",
                "end_line": "L0010",
                "start_quote": "That is the whole system.",
                "end_quote": "It is not fancy, but it works.",
            },
            settings=default_settings(min_duration=8.0, max_duration=30.0),
        )
        assert clip.start_line == "L0009"
        assert clip.duration >= 8.0
        assert any("extended" in warning for warning in clip.warnings)

    def test_max_duration_trims_only_padding(self):
        clip = resolve_one(
            settings=default_settings(min_duration=8.0, max_duration=25.2)
        )
        assert clip.duration <= 25.2 + 0.01
        assert clip.end >= 29.8

    def test_short_source_uses_effective_min(self):
        units = [Unit("L0001", 0.0, 3.5, "A short entire video.")]
        transcript = Transcript("en", units, "fixture", 4.0)
        blocks = build_blocks(transcript, block_seconds=120.0, min_overlap_seconds=10.0)
        raw = copy.deepcopy(response_payload()["clips"][0])
        raw.update(
            {
                "start_line": "L0001",
                "end_line": "L0001",
                "start_quote": "A short entire video.",
                "end_quote": "A short entire video.",
                "scores": {"overall": 80},
            }
        )
        candidate = parse_candidate(raw)
        clip = resolve_candidate(
            candidate, transcript, blocks, 4.0, default_settings(min_duration=15.0)
        )
        assert clip.start == 0.0
        assert clip.end == 4.0

    def test_source_end_clamped(self):
        units = [Unit("L0001", 45.2, 49.8, "The end of the source.")]
        transcript = Transcript("en", units, "fixture", 50.0)
        blocks = build_blocks(transcript, block_seconds=120.0, min_overlap_seconds=10.0)
        raw = copy.deepcopy(response_payload()["clips"][0])
        raw.update(
            {
                "start_line": "L0001",
                "end_line": "L0001",
                "start_quote": "The end of the source.",
                "end_quote": "The end of the source.",
                "scores": {"overall": 80},
            }
        )
        clip = resolve_candidate(
            parse_candidate(raw),
            transcript,
            blocks,
            50.0,
            default_settings(min_duration=5.0),
        )
        assert clip.end <= 50.0


class TestMetadataFallback:
    def test_missing_title_and_hook(self):
        raw = copy.deepcopy(response_payload()["clips"][0])
        for key in ("title", "hook"):
            raw.pop(key)
        candidate = parse_candidate(raw, default_block_id="B001")
        transcript = setup_transcript()
        blocks = build_blocks(transcript, block_seconds=120.0, min_overlap_seconds=40.0)
        clip = resolve_candidate(
            candidate, transcript, blocks, 50.0, default_settings()
        )
        assert clip.metadata["title"]
        assert clip.metadata["hook"]
        assert any("title was missing" in warning for warning in clip.warnings)
        assert any("hook was missing" in warning for warning in clip.warnings)

    def test_long_title_truncated(self):
        clip = resolve_one({"title": "A" * 200})
        assert len(clip.metadata["title"]) <= 100


class TestResponseResolution:
    def test_resolves_and_rejects(self):
        transcript = setup_transcript()
        blocks = build_blocks(transcript, block_seconds=120.0, min_overlap_seconds=40.0)
        parsed = parse_response_payload(response_payload(), source_file="x.json")
        resolved, rejected = resolve_responses(
            [parsed], transcript, blocks, 50.0, default_settings()
        )
        assert len(resolved) == 5
        assert len(rejected) == 2
        assert any("above the maximum" in item.reason for item in rejected)
        assert any("did not match" in item.reason for item in rejected)


class TestSelection:
    def _resolved(self):
        transcript = setup_transcript()
        blocks = build_blocks(transcript, block_seconds=120.0, min_overlap_seconds=40.0)
        parsed = parse_response_payload(response_payload(), source_file="x.json")
        resolved, _ = resolve_responses(
            [parsed], transcript, blocks, 50.0, default_settings()
        )
        return resolved

    def test_filters_below_score(self):
        selection = select_clips(
            self._resolved(), default_settings(min_score=75)
        )
        assert all(clip.selection_score >= 75 for clip in selection.selected)
        assert selection.below_score >= 1

    def test_duplicate_keeps_higher_rank(self):
        selection = select_clips(self._resolved(), default_settings())
        starts = [(clip.start, clip.end) for clip in selection.selected]
        assert (4.9, 30.15) in starts
        assert (10.0, 30.15) not in starts
        assert selection.duplicate_count >= 1

    def test_low_standalone_is_dropped_even_with_high_overall(self):
        resolved = self._resolved()
        target = next(clip for clip in resolved if clip.selection_score == 88)
        target.candidate.scores["standalone"] = 2
        selection = select_clips(resolved, default_settings())
        assert all(clip is not target for clip in selection.selected)
        assert selection.low_context == 1
        assert any(
            "cold-viewer floor" in item.reason for item in selection.dropped
        )

    def test_max_clips(self):
        selection = select_clips(self._resolved(), default_settings(max_clips=1))
        assert len(selection.selected) == 1
        assert selection.selected[0].selection_score == 88

    def test_chronological_ids_with_rank(self):
        selection = select_clips(self._resolved(), default_settings())
        assert [clip.clip_id for clip in selection.selected] == ["C001", "C002"]
        assert [clip.selection_rank for clip in selection.selected] == [1, 2]

    def test_never_pads(self):
        selection = select_clips(self._resolved(), default_settings(min_score=99))
        assert selection.selected == []

    def test_no_duplicate_survivors(self):
        selection = select_clips(self._resolved(), default_settings())
        for index, first in enumerate(selection.selected):
            for second in selection.selected[index + 1 :]:
                overlap = min(first.end, second.end) - max(first.start, second.start)
                shorter = min(first.duration, second.duration)
                assert overlap <= 0 or overlap / shorter < 0.35

    def test_rank_is_score_order(self):
        selection = select_clips(self._resolved(), default_settings())
        ranks = [clip.selection_rank for clip in selection.selected]
        scores = [clip.selection_score for clip in selection.selected]
        assert sorted(ranks) == list(range(1, len(ranks) + 1))
        assert [score for _, score in sorted(zip(ranks, scores))] == sorted(
            scores, reverse=True
        )
