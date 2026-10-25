from __future__ import annotations

import pytest

from momentcut.asr import plan_chunks
from momentcut.transcript import (
    TranscriptError,
    Word,
    assemble_words,
    build_units,
    load_transcript_json,
    merge_overlapping_words,
    parse_srt,
    parse_vtt,
    units_from_cues,
)
from tests.conftest import transcript_payload


def w(text, start, end):
    return {"text": text, "start": start, "end": end}


class TestAssemble:
    def test_preserves_whisper_leading_spaces(self):
        words = [
            Word(" Hello", 0.0, 0.5),
            Word(" world", 0.5, 1.0),
            Word(".", 1.0, 1.1),
            Word(" Next", 1.2, 1.5),
        ]
        assert assemble_words(words) == "Hello world. Next"

    def test_joins_unspaced_words(self):
        words = [Word("hello", 0.0, 0.4), Word("world", 0.4, 0.8)]
        assert assemble_words(words) == "hello world"

    def test_cjk_does_not_inject_spaces(self):
        words = [Word("你", 0.0, 0.2), Word("好", 0.2, 0.4)]
        assert assemble_words(words, "zh") == "你好"


class TestBuildUnits:
    def test_splits_on_punctuation_and_keeps_word_times(self):
        words = [
            w("Hello", 0.0, 0.4),
            w(" world", 0.4, 0.9),
            w(".", 0.9, 1.0),
            w(" Second", 1.2, 1.7),
            w(" sentence", 1.7, 2.4),
            w(".", 2.4, 2.5),
        ]
        units = build_units(words, 3.0, language="en")
        assert len(units) == 2
        assert units[0].text == "Hello world."
        assert units[0].start == 0.0
        assert units[0].end == 1.0
        assert units[1].id == "L0002"

    def test_splits_on_long_pause_without_punctuation(self):
        words = [w(f"word{i}", i * 0.4, i * 0.4 + 0.3) for i in range(7)]
        words += [w("after", 6.0, 6.4), w("pause", 6.4, 6.8), w("here", 6.8, 7.2)]
        units = build_units(words, 8.0, language="en")
        assert len(units) == 2
        assert "after pause here" in units[1].text

    def test_soft_limit_waits_for_a_pause(self):
        words = [w(f"word{i}", i * 0.5, i * 0.5 + 0.45) for i in range(80)]
        words += [w("after", 41.5, 41.9), w("the", 41.9, 42.2), w("pause", 42.2, 42.6)]
        units = build_units(words, 50.0, language="en")
        assert len(units) == 2
        assert units[0].end > 35.0

    def test_clamps_to_source_duration(self):
        units = build_units([w("one", 0.0, 5.0), w("two", 9.5, 10.5)], 10.0)
        assert units[-1].end <= 10.0

    def test_empty_words(self):
        assert build_units([], 10.0) == []

    def test_no_letters_dropped(self):
        assert build_units([w("...", 0.0, 1.0)], 10.0) == []


class TestMergeWords:
    def test_removes_overlap_duplicates(self):
        first = [Word(" hello", 0.0, 0.6), Word(" world", 0.6, 1.1)]
        second = [Word(" world", 0.6, 1.1), Word(" again", 1.1, 1.5)]
        merged = merge_overlapping_words([first, second])
        assert [word.text.strip() for word in merged] == ["hello", "world", "again"]

    def test_keeps_genuine_repetition(self):
        words = [Word(" no", 0.0, 0.4), Word(" no", 0.6, 1.0)]
        merged = merge_overlapping_words([words])
        assert len(merged) == 2


class TestSubtitles:
    def test_parse_srt(self):
        raw = """1
00:00:01,000 --> 00:00:03,000
<i>Hello</i> there

2
00:00:03,100 --> 00:00:05,000
This is a test.
"""
        cues = parse_srt(raw, 10.0)
        assert len(cues) == 2
        assert cues[0]["text"] == "Hello there"
        assert cues[0]["start"] == 1.0

    def test_parse_vtt_with_note(self):
        raw = """WEBVTT

NOTE this is a comment

00:00:01.000 --> 00:00:02.500 align:start
Hello world.
"""
        cues = parse_vtt(raw, 10.0)
        assert len(cues) == 1
        assert cues[0]["text"] == "Hello world."

    def test_invalid_subtitle_raises(self):
        with pytest.raises(TranscriptError):
            parse_srt("not a subtitle", 10.0)

    def test_cues_merge_to_sentence(self):
        cues = [
            {"start": 0.0, "end": 1.0, "text": "This is a"},
            {"start": 1.0, "end": 2.0, "text": "complete sentence."},
            {"start": 3.0, "end": 4.0, "text": "Another one."},
        ]
        units = units_from_cues(cues, 10.0)
        assert len(units) == 2
        assert units[0].text == "This is a complete sentence."


class TestJsonTranscript:
    def test_whisper_segments_with_words(self):
        payload = {
            "language": "en",
            "segments": [
                {
                    "start": 0.0,
                    "end": 2.0,
                    "text": "Hello world.",
                    "words": [
                        {"word": " Hello", "start": 0.0, "end": 0.5},
                        {"word": " world", "start": 0.5, "end": 1.0},
                        {"word": ".", "start": 1.0, "end": 1.1},
                    ],
                }
            ],
        }
        transcript = load_transcript_json(payload, 10.0)
        assert len(transcript.units) == 1
        assert transcript.units[0].text == "Hello world."

    def test_units_payload(self):
        transcript = load_transcript_json(transcript_payload(), 50.0)
        assert len(transcript.units) == 10
        assert transcript.units[1].id == "L0002"

    def test_segments_without_words(self):
        payload = {
            "language": "id",
            "segments": [
                {"start": 0.0, "end": 2.0, "text": "Halo semua."},
                {"start": 3.0, "end": 5.0, "text": "Ini tes."},
            ],
        }
        transcript = load_transcript_json(payload, 10.0)
        assert len(transcript.units) == 2
        assert transcript.language == "id"

    def test_plain_text_rejected(self):
        with pytest.raises(TranscriptError):
            load_transcript_json({"text": "no timestamps"}, 10.0)


class TestPlanChunks:
    def test_short_source_one_chunk(self):
        assert plan_chunks(50.0, 600.0) == [(0.0, 50.0)]

    def test_covers_and_overlaps(self):
        chunks = plan_chunks(1250.0, 600.0, overlap_seconds=3.0)
        assert chunks[0][0] == 0.0
        assert chunks[-1][1] == 1250.0
        assert chunks[1][0] < chunks[0][1]
        for previous, current in zip(chunks, chunks[1:]):
            assert current[0] < previous[1]

    def test_avoids_tiny_tail(self):
        chunks = plan_chunks(1200.0, 600.0, overlap_seconds=3.0)
        assert all(end - start >= 30.0 for start, end in chunks)
        assert chunks[-1][1] == 1200.0
