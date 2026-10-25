from __future__ import annotations

import json

import pytest

from momentcut.contract import (
    CandidateError,
    ContractError,
    load_response_file,
    normalise_hashtags,
    normalise_line_ref,
    parse_candidate,
    parse_json_payload,
    parse_response_payload,
)
from momentcut.util import clean_text

from .conftest import response_payload, write_json


class TestJsonExtraction:
    def test_plain_object(self):
        assert parse_json_payload('{"a": 1}') == {"a": 1}

    def test_code_fence(self):
        raw = 'Here is the answer:\n```json\n{"a": 1}\n```\nDone.'
        assert parse_json_payload(raw) == {"a": 1}

    def test_prose_around_object(self):
        raw = 'Sure. {"schema_version": "1.0", "clips": []} Hope that helps.'
        assert parse_json_payload(raw)["clips"] == []

    def test_bare_array(self):
        raw = '[{"a": 1}, {"b": 2}]'
        assert parse_json_payload(raw) == [{"a": 1}, {"b": 2}]

    def test_trailing_comma_narrow_repair(self):
        assert parse_json_payload('{"a": 1,}') == {"a": 1}

    def test_garbage_raises(self):
        with pytest.raises(ContractError):
            parse_json_payload("I cannot help with that.")


class TestHashtagsAndRefs:
    def test_caption_text_is_not_turned_into_hashtags(self):
        assert normalise_hashtags("The gear was never the story. The moment was.") == []

    def test_explicit_hashtags(self):
        assert normalise_hashtags("#AI #video, #AI") == ["#AI", "#video"]

    def test_list_items(self):
        assert normalise_hashtags(["storytelling", "#video", "#video"]) == [
            "#storytelling",
            "#video",
        ]

    def test_line_refs(self):
        assert normalise_line_ref("L0012") == "L0012"
        assert normalise_line_ref("line 12") == "L0012"
        assert normalise_line_ref(3) == "L0003"
        assert normalise_line_ref("abc") is None


class TestResponsePayload:
    def test_canonical_response(self):
        parsed = parse_response_payload(response_payload(), source_file="x.json")
        assert parsed.block_id == "B001"
        assert len(parsed.candidates) == 7
        assert parsed.candidates[0].scores["overall"] == 88

    def test_bare_array(self):
        payload = [response_payload()["clips"][0]]
        parsed = parse_response_payload(payload, filename_hint="B001")
        assert parsed.block_id == "B001"
        assert len(parsed.candidates) == 1

    def test_shorts_alias(self):
        payload = {"schema_version": "1.0", "shorts": response_payload()["clips"]}
        parsed = parse_response_payload(payload)
        assert len(parsed.candidates) == 7

    def test_content_wrapper_keeps_block_id(self):
        inner = {"schema_version": "1.0", "clips": response_payload()["clips"][:1]}
        payload = {"block_id": "B001", "content": json.dumps(inner)}
        parsed = parse_response_payload(payload)
        assert parsed.block_id == "B001"
        assert len(parsed.candidates) == 1

    def test_chat_completion_wrapper(self):
        inner = {"schema_version": "1.0", "clips": response_payload()["clips"][:1]}
        payload = {
            "choices": [
                {"message": {"content": json.dumps(inner)}, "finish_reason": "stop"}
            ]
        }
        parsed = parse_response_payload(payload, filename_hint="B001")
        assert parsed.block_id == "B001"
        assert len(parsed.candidates) == 1

    def test_one_bad_candidate_does_not_destroy_the_rest(self):
        payload = {
            "schema_version": "1.0",
            "block_id": "B001",
            "clips": [
                {"foo": "bar"},
                response_payload()["clips"][0],
            ],
        }
        parsed = parse_response_payload(payload)
        assert len(parsed.candidates) == 1
        assert len(parsed.errors) == 1
        assert "start anchor" in parsed.errors[0] or "end anchor" in parsed.errors[0]

    def test_schema_version_mismatch(self):
        payload = response_payload()
        payload["schema_version"] = "2.0"
        with pytest.raises(ContractError):
            parse_response_payload(payload)

    def test_missing_clips_raises(self):
        with pytest.raises(ContractError):
            parse_response_payload({"schema_version": "1.0", "block_id": "B001"})

    def test_results_wrapper_rejected(self):
        with pytest.raises(ContractError):
            parse_response_payload({"results": []})

    def test_load_file(self, tmp_path):
        path = write_json(tmp_path / "block_001.json", response_payload())
        parsed = load_response_file(path)
        assert parsed.block_id == "B001"
        assert len(parsed.candidates) == 7

    def test_large_file_rejected(self, tmp_path):
        path = tmp_path / "huge.json"
        path.write_text("{" + " " * (10 * 1024 * 1024 + 10) + "}", encoding="utf-8")
        with pytest.raises(ContractError):
            load_response_file(path)


class TestScoresAndMetadata:
    def test_derives_overall_from_dimensions(self):
        raw = response_payload()["clips"][0]
        raw["scores"] = {"hook": 8, "standalone": 8, "payoff": 8}
        candidate = parse_candidate(raw)
        assert candidate.scores["overall"] == 80
        assert any("overall" in warning for warning in candidate.warnings)

    def test_derives_dimensions_from_overall(self):
        raw = response_payload()["clips"][0]
        raw["scores"] = {"overall": 74}
        candidate = parse_candidate(raw)
        assert candidate.scores["hook"] == 7

    def test_no_score_rejected(self):
        raw = response_payload()["clips"][0]
        raw.pop("scores")
        with pytest.raises(CandidateError):
            parse_candidate(raw)

    def test_missing_metadata_is_left_for_resolver(self):
        raw = response_payload()["clips"][0]
        for key in ("title", "hook", "caption", "summary", "why"):
            raw.pop(key, None)
        candidate = parse_candidate(raw)
        assert candidate.title == ""

    def test_clamps_out_of_range_scores(self):
        raw = response_payload()["clips"][0]
        raw["scores"] = {"hook": 99, "standalone": -3, "payoff": 8, "overall": 150}
        candidate = parse_candidate(raw)
        assert candidate.scores == {
            "hook": 10,
            "standalone": 0,
            "payoff": 8,
            "overall": 100,
        }

    def test_unknown_moment_type_becomes_other(self):
        raw = response_payload()["clips"][0]
        raw["moment_type"] = "interpretive-dance"
        assert parse_candidate(raw).moment_type == "other"

    def test_clean_text_removes_control_chars(self):
        assert "\x00" not in clean_text("hello\x00world")
