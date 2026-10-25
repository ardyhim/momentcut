from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "requires_ffmpeg: needs the ffmpeg executable")


def require_ffmpeg() -> None:
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("ffmpeg/ffprobe not available")


def make_video(path: Path, duration: float = 50.0, size: str = "640x360") -> Path:
    require_ffmpeg()
    command = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"testsrc2=size={size}:rate=30",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=440:sample_rate=44100",
        "-t",
        str(duration),
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-crf",
        "30",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-shortest",
        str(path),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)
    return path


@pytest.fixture(scope="session")
def video(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("media") / "fixture.mp4"
    return make_video(path, 50.0)


@pytest.fixture(scope="session")
def other_video(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("media-other") / "other.mp4"
    return make_video(path, 3.0, "320x240")


@pytest.fixture(scope="session")
def long_video(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("media-long") / "long.mp4"
    return make_video(path, 140.0, "320x180")


def transcript_payload() -> dict[str, Any]:
    return {
        "language": "en",
        "backend": "fixture",
        "source_duration": 50.0,
        "units": [
            {
                "id": "L0001",
                "start": 0.2,
                "end": 4.8,
                "text": "I used to think making videos was about buying the best camera.",
            },
            {
                "id": "L0002",
                "start": 5.1,
                "end": 9.8,
                "text": "Then I realized something that changed everything.",
            },
            {
                "id": "L0003",
                "start": 10.2,
                "end": 14.9,
                "text": "The camera does not make the story, and the story does not need a perfect shot.",
            },
            {
                "id": "L0004",
                "start": 15.2,
                "end": 19.8,
                "text": "It needs a moment that a stranger can understand in two seconds.",
            },
            {
                "id": "L0005",
                "start": 20.2,
                "end": 24.9,
                "text": "That is why this one sentence from a stranger changed my whole process.",
            },
            {
                "id": "L0006",
                "start": 25.2,
                "end": 29.8,
                "text": "I stopped chasing gear and started chasing tension.",
            },
            {
                "id": "L0007",
                "start": 30.2,
                "end": 34.9,
                "text": "The result was fewer videos, better clips, and twice the watch time.",
            },
            {
                "id": "L0008",
                "start": 35.2,
                "end": 39.8,
                "text": "So here is the exact three-step checklist I use now.",
            },
            {
                "id": "L0009",
                "start": 40.2,
                "end": 44.9,
                "text": "First, find the conflict. Second, cut before the explanation. Third, end after the payoff.",
            },
            {
                "id": "L0010",
                "start": 45.2,
                "end": 49.8,
                "text": "That is the whole system. It is not fancy, but it works.",
            },
        ],
    }


def response_payload() -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "block_id": "B001",
        "clips": [
            {
                "start_line": "L0002",
                "end_line": "L0006",
                "start_quote": "Then I realized something that changed everything.",
                "end_quote": "I stopped chasing gear and started chasing tension.",
                "title": "Stop Buying Cameras, Start Finding Tension",
                "hook": "Your camera is not the problem",
                "caption": "The gear was never the story. The moment was.",
                "hashtags": ["#storytelling", "#video"],
                "summary": "A creator explains why tension matters more than camera gear.",
                "why": "It has a clear before/after and a concrete payoff about watch time.",
                "moment_type": "insight",
                "scores": {"hook": 9, "standalone": 9, "payoff": 8, "overall": 88},
            },
            {
                "start_line": "L0003",
                "end_line": "L0006",
                "start_quote": "The camera does not make the story",
                "end_quote": "I stopped chasing gear and started chasing tension.",
                "title": "The Camera Is Not the Story",
                "hook": "The story needs no perfect shot",
                "caption": "A perfect shot is not the same thing as a story.",
                "hashtags": ["#video"],
                "summary": "A duplicate framing of the same insight.",
                "why": "Same moment.",
                "moment_type": "insight",
                "scores": {"hook": 7, "standalone": 8, "payoff": 7, "overall": 80},
            },
            {
                "start_line": "L0001",
                "end_line": "L0010",
                "start_quote": "I used to think making videos was about buying the best camera.",
                "end_quote": "That is the whole system. It is not fancy, but it works.",
                "title": "The Whole Video",
                "hook": "Everything I know about video",
                "caption": "This is the entire system.",
                "hashtags": [],
                "summary": "The full transcript.",
                "why": "Too long.",
                "moment_type": "insight",
                "scores": {"hook": 6, "standalone": 9, "payoff": 8, "overall": 90},
            },
            {
                "start_line": "L9999",
                "end_line": "L0010",
                "start_quote": "A line that does not exist.",
                "end_quote": "That is the whole system. It is not fancy, but it works.",
                "title": "Hallucinated Anchor",
                "hook": "This should be rejected",
                "caption": "This should be rejected.",
                "hashtags": [],
                "summary": "Invalid line.",
                "why": "Invalid line.",
                "moment_type": "other",
                "scores": {"hook": 10, "standalone": 10, "payoff": 10, "overall": 99},
            },
            {
                "start_line": "L0007",
                "end_line": "L0010",
                "start_quote": "The result was fewer videos, better clips, and twice the watch time.",
                "end_quote": "That is the whole system. It is not fancy, but it works.",
                "hook": "Fewer videos, better clips",
                "caption": "The system is simple, but it works.",
                "hashtags": ["#workflow"],
                "summary": "A closing thought on a simpler clip workflow.",
                "why": "It closes the arc with a result and a summary.",
                "moment_type": "insight",
                "scores": {"hook": 7, "standalone": 7, "payoff": 8, "overall": 76},
            },
            {
                "start_line": "L0009",
                "end_line": "L0010",
                "start_quote": "First, find the conflict.",
                "end_quote": "That is the whole system. It is not fancy, but it works.",
                "title": "Three Step Checklist",
                "hook": "A three step checklist",
                "caption": "Three steps, nothing fancy.",
                "hashtags": [],
                "summary": "A short checklist.",
                "why": "Useful but below the quality bar.",
                "moment_type": "insight",
                "scores": {"hook": 5, "standalone": 5, "payoff": 5, "overall": 40},
            },
            {
                "start_line": "L0002",
                "end_line": "L0006",
                "start_quote": "The camera does not make the story",
                "end_quote": "I stopped chasing gear and started chasing tension.",
                "title": "Quote Repairs A Bad Start Line",
                "hook": "The camera is not the story",
                "caption": "The quote should repair the start line.",
                "hashtags": [],
                "summary": "A boundary repair test.",
                "why": "The quote is on L0003, not L0002.",
                "moment_type": "insight",
                "scores": {"hook": 7, "standalone": 7, "payoff": 7, "overall": 70},
            },
        ],
    }


def write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
