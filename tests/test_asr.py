from __future__ import annotations

from pathlib import Path

import pytest

from momentcut import asr
from momentcut.asr import AsrError, transcribe_media


class FakeInfo:
    language = "en"
    language_probability = 0.99


class FakeWord:
    def __init__(self, word, start, end):
        self.word = word
        self.start = start
        self.end = end
        self.probability = 0.9


class FakeSegment:
    def __init__(self, text, start, end):
        self.text = text
        self.start = start
        self.end = end
        self.words = [FakeWord(text, start, end)]


class FakeModel:
    def __init__(self, counter):
        self.counter = counter

    def transcribe(self, path, **kwargs):
        self.counter["transcribe"] += 1
        # Each chunk is independent; the wrapper adds the chunk offset.
        return iter([FakeSegment("hello.", 0.2, 1.2)]), FakeInfo()


def test_transcribe_media_is_resumable(tmp_path, monkeypatch):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"fake video")
    cache = tmp_path / "asr"
    fingerprint = {"quick_hash": "abc", "size": source.stat().st_size, "duration": 60.0}
    counter = {"transcribe": 0, "load": 0}
    model = FakeModel(counter)

    def fake_extract(source, start, duration, destination):
        destination = Path(destination)
        destination.write_bytes(b"RIFF fake wav")
        return destination

    def fake_load(name):
        counter["load"] += 1
        return model, "cpu", "int8"

    monkeypatch.setattr(asr, "extract_audio_chunk", fake_extract)
    monkeypatch.setattr(asr, "_load_model", fake_load)

    transcript = transcribe_media(
        source,
        duration=60.0,
        source_fingerprint=fingerprint,
        cache_dir=cache,
        model_name="tiny",
        chunk_seconds=20.0,
        language="en",
        has_audio=True,
    )
    assert len(transcript.units) == 2
    assert transcript.units[0].start == 0.2
    assert transcript.units[1].start > 17.0
    assert counter["transcribe"] == 2
    assert counter["load"] == 1
    chunk_files = list(cache.rglob("chunk_*.json"))
    assert len(chunk_files) == 2

    transcript_again = transcribe_media(
        source,
        duration=60.0,
        source_fingerprint=fingerprint,
        cache_dir=cache,
        model_name="tiny",
        chunk_seconds=20.0,
        language="en",
        has_audio=True,
    )
    assert [unit.start for unit in transcript_again.units] == [
        unit.start for unit in transcript.units
    ]
    assert counter["transcribe"] == 2
    assert counter["load"] == 1


def test_no_audio_rejected(tmp_path):
    with pytest.raises(AsrError):
        transcribe_media(
            tmp_path / "source.mp4",
            duration=10.0,
            source_fingerprint={},
            cache_dir=tmp_path / "asr",
            has_audio=False,
        )


def test_missing_faster_whisper_has_actionable_error(tmp_path, monkeypatch):
    # Simulate a missing dependency by making the import fail inside the loader.
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "faster_whisper":
            raise ImportError("no faster_whisper")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(AsrError) as exc:
        asr._load_model("tiny")
    assert "faster-whisper" in str(exc.value)
