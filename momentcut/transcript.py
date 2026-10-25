"""Transcript normalization: words -> sentence units -> line IDs.

The AI never receives the whole video and never owns the final cut times. It
receives units (`L0001`, start, end, text); the resolver later turns chosen line
IDs back into exact source seconds. This module is the only place where
transcript timing is constructed.
"""

from __future__ import annotations

import html
import json
import math
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .util import (
    MomentCutError,
    clean_text,
    coerce_float,
    format_clock,
    normalize_for_match,
    read_json,
)

CJK_LANGUAGES = {"zh", "ja", "yue", "ko", "th", "lo", "km", "my"}
SENTENCE_END_RE = re.compile(r"""[.!?…。？！]+["'”’)\]】》]*$""")


class TranscriptError(MomentCutError):
    pass


@dataclass(frozen=True)
class Word:
    text: str
    start: float
    end: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "start": round(self.start, 3),
            "end": round(self.end, 3),
        }


@dataclass(frozen=True)
class Unit:
    id: str
    start: float
    end: float
    text: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "text": self.text,
        }


@dataclass
class Transcript:
    language: str
    units: list[Unit]
    backend: str
    source_duration: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "1.0",
            "language": self.language or "unknown",
            "backend": self.backend,
            "source_duration": round(float(self.source_duration), 3),
            "unit_count": len(self.units),
            "metadata": self.metadata,
            "units": [unit.to_dict() for unit in self.units],
        }

    def unit_by_id(self, unit_id: str) -> Unit | None:
        for unit in self.units:
            if unit.id == unit_id:
                return unit
        return None

    def index_by_id(self) -> dict[str, int]:
        return {unit.id: index for index, unit in enumerate(self.units)}

    def text_between(self, start_index: int, end_index: int) -> str:
        return clean_text(" ".join(unit.text for unit in self.units[start_index : end_index + 1]))


def assemble_words(words: Sequence[Word], language: str | None = None) -> str:
    """Rebuild readable text without destroying languages that do not use spaces."""
    if not words:
        return ""
    raw = [word.text for word in words]
    has_leading_space = any(text[:1].isspace() for text in raw[1:])
    language_code = (language or "").split("-")[0].lower()
    cjk = language_code in CJK_LANGUAGES
    if has_leading_space:
        assembled = "".join(raw)
    elif cjk:
        assembled = "".join(text.strip() for text in raw)
    else:
        assembled = " ".join(text.strip() for text in raw)
    assembled = assembled.replace("\u00a0", " ").replace("\u202f", " ")
    assembled = re.sub(r"\s+", " ", assembled).strip()
    assembled = re.sub(r"\s+([,.;:!?%…])", r"\1", assembled)
    assembled = re.sub(r"([(\[¿¡])\s+", r"\1", assembled)
    return assembled


def _normalize_word_text(value: Any) -> str:
    text = clean_text(value, limit=200)
    return text


def _prepare_words(
    raw_words: Iterable[Any],
    source_duration: float,
    *,
    offset: float = 0.0,
) -> list[Word]:
    prepared: list[Word] = []
    for item in raw_words:
        if isinstance(item, Word):
            text, start, end = item.text, item.start, item.end
        elif isinstance(item, dict):
            text = item.get("text", item.get("word", item.get("w", "")))
            start = coerce_float(item.get("start", item.get("s")))
            end = coerce_float(item.get("end", item.get("e")))
        else:
            continue
        text = _normalize_word_text(text)
        if not text or start is None or end is None:
            continue
        start = start + offset
        end = end + offset
        if not math.isfinite(start) or not math.isfinite(end):
            continue
        if start < 0:
            start = 0.0
        if end <= start:
            end = start + 0.05
        if start > source_duration + 0.5:
            continue
        end = min(end, source_duration)
        start = min(start, source_duration)
        if end <= start:
            continue
        # A single "word" lasting minutes is an ASR hallucination, not a word.
        if end - start > 15.0:
            end = start + 15.0
        prepared.append(Word(text=text, start=round(start, 3), end=round(end, 3)))
    prepared.sort(key=lambda word: (word.start, word.end))
    return prepared


def merge_overlapping_words(word_chunks: Sequence[Sequence[Word]]) -> list[Word]:
    """Merge chunks transcribed with overlap, removing boundary duplicates.

    This is deliberately conservative: only near-identical words whose times
    overlap are collapsed. A word that merely appears twice in the speech is
    kept.
    """
    all_words: list[Word] = []
    for chunk in word_chunks:
        all_words.extend(chunk)
    all_words.sort(key=lambda word: (word.start, word.end))

    merged: list[Word] = []
    for word in all_words:
        duplicate = False
        window_start = max(0, len(merged) - 2)
        for index in range(window_start, len(merged)):
            previous = merged[index]
            previous_key = normalize_for_match(previous.text)
            word_key = normalize_for_match(word.text)
            if not previous_key or previous_key != word_key:
                continue
            overlap = min(previous.end, word.end) - max(previous.start, word.start)
            shorter = max(
                1e-6,
                min(previous.end - previous.start, word.end - word.start),
            )
            close_start = abs(previous.start - word.start) <= 0.25
            if overlap > 0 and (overlap / shorter >= 0.5 or close_start):
                duplicate = True
                # Keep the longer rendering, but never move a word far forward.
                if (word.end - word.start) > (previous.end - previous.start):
                    merged[index] = word
                break
        if not duplicate:
            merged.append(word)
    merged.sort(key=lambda word: (word.start, word.end))
    return merged


def build_units(
    raw_words: Iterable[Any],
    source_duration: float,
    *,
    language: str | None = None,
    pause_seconds: float = 1.2,
    soft_max_seconds: float = 30.0,
    hard_max_seconds: float = 60.0,
    soft_max_chars: int = 450,
    hard_max_chars: int = 900,
    offset: float = 0.0,
) -> list[Unit]:
    """Build sentence-like units without gratuitously cutting mid-thought.

    Punctuation and long pauses are the preferred boundaries. A soft length
    limit only closes the unit if the next word follows a small pause; otherwise
    the unit is allowed to grow to a hard limit. This is a compromise: a 2-minute
    unpunctuated monologue still needs a boundary, but a normal 32-second
    sentence is not split at an arbitrary word.
    """
    words = _prepare_words(raw_words, source_duration, offset=offset)
    if not words:
        return []

    units: list[Unit] = []
    current: list[Word] = []

    def close_current() -> None:
        nonlocal current
        if not current:
            return
        text = assemble_words(current, language)
        if not text or not re.search(r"\w", text, flags=re.UNICODE):
            current = []
            return
        start = max(0.0, current[0].start)
        end = min(source_duration, current[-1].end)
        if end > start + 0.05:
            units.append(Unit(id="", start=round(start, 3), end=round(end, 3), text=text))
        current = []

    for index, word in enumerate(words):
        current.append(word)
        next_word = words[index + 1] if index + 1 < len(words) else None
        current_text = assemble_words(current, language)
        gap = (next_word.start - word.end) if next_word else 999.0
        ends_sentence = bool(SENTENCE_END_RE.search(word.text.strip()))
        long_pause = gap >= pause_seconds and len(current) >= 6
        soft_limit = (
            (word.end - current[0].start) >= soft_max_seconds
            or len(current_text) >= soft_max_chars
        )
        hard_limit = (
            (word.end - current[0].start) >= hard_max_seconds
            or len(current_text) >= hard_max_chars
        )
        # At the soft limit, wait for a small pause rather than cutting in the
        # middle of a phrase. The hard limit is the last-resort bound.
        breakable = next_word is None or gap >= 0.6
        if ends_sentence or long_pause or hard_limit or (soft_limit and breakable):
            close_current()

    if current:
        close_current()
    return _assign_ids(units, source_duration)


def _assign_ids(units: Sequence[Unit], source_duration: float) -> list[Unit]:
    normalized: list[Unit] = []
    for unit in units:
        text = clean_text(unit.text)
        start = coerce_float(unit.start)
        end = coerce_float(unit.end)
        if not text or start is None or end is None:
            continue
        start = max(0.0, min(float(start), source_duration))
        end = max(0.0, min(float(end), source_duration))
        if end <= start:
            continue
        if normalized and start < normalized[-1].start:
            continue
        normalized.append(
            Unit(id="", start=round(start, 3), end=round(end, 3), text=text)
        )
    normalized.sort(key=lambda unit: (unit.start, unit.end))
    width = max(4, len(str(len(normalized))))
    return [
        Unit(id=f"L{index:0{width}d}", start=unit.start, end=unit.end, text=unit.text)
        for index, unit in enumerate(normalized, start=1)
    ]


def units_from_cues(
    cues: Iterable[Any],
    source_duration: float,
    *,
    language: str | None = None,
) -> list[Unit]:
    prepared: list[tuple[float, float, str]] = []
    for cue in cues:
        if isinstance(cue, dict):
            start = coerce_float(cue.get("start"))
            end = coerce_float(cue.get("end"))
            text = clean_text(cue.get("text", ""))
        else:
            continue
        if start is None or end is None or not text:
            continue
        start = max(0.0, min(start, source_duration))
        end = max(0.0, min(end, source_duration))
        if end <= start:
            continue
        prepared.append((start, end, text))
    prepared.sort(key=lambda cue: (cue[0], cue[1]))

    units: list[Unit] = []
    buffer: list[tuple[float, float, str]] = []

    def close() -> None:
        nonlocal buffer
        if not buffer:
            return
        text = clean_text(" ".join(item[2] for item in buffer))
        if text:
            units.append(
                Unit(
                    id="",
                    start=round(buffer[0][0], 3),
                    end=round(buffer[-1][1], 3),
                    text=text,
                )
            )
        buffer = []

    for cue in prepared:
        buffer.append(cue)
        duration = buffer[-1][1] - buffer[0][0]
        ends = bool(SENTENCE_END_RE.search(cue[2]))
        if ends or duration >= 20.0:
            close()
    close()
    return _assign_ids(units, source_duration)


def parse_timestamp(value: str) -> float:
    text = value.strip().replace(",", ".")
    parts = text.split(":")
    try:
        if len(parts) == 3:
            hours, minutes, seconds = parts
            return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
        if len(parts) == 2:
            minutes, seconds = parts
            return int(minutes) * 60 + float(seconds)
        if len(parts) == 1:
            return float(parts[0])
    except ValueError as exc:
        raise TranscriptError(f"Invalid subtitle timestamp: {value!r}") from exc
    raise TranscriptError(f"Invalid subtitle timestamp: {value!r}")


_TIMESTAMP = r"\d{1,3}:\d{2}(?::\d{2})?[.,]\d{1,3}"
_CUE_RE = re.compile(
    rf"({_TIMESTAMP})\s*-->\s*({_TIMESTAMP})(?:[^\n]*)\n(.*?)(?=\n\s*\n|\Z)",
    re.DOTALL,
)


def _clean_subtitle_text(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\{\\[^}]+\}", " ", text)
    text = html.unescape(text)
    return clean_text(text)


def parse_srt(text: str, source_duration: float) -> list[dict[str, Any]]:
    normalized = text.replace("\ufeff", "").replace("\r\n", "\n").replace("\r", "\n")
    cues: list[dict[str, Any]] = []
    for match in _CUE_RE.finditer(normalized):
        start = parse_timestamp(match.group(1))
        end = parse_timestamp(match.group(2))
        body = _clean_subtitle_text(match.group(3))
        if body and end > start:
            cues.append({"start": start, "end": end, "text": body})
    if not cues:
        raise TranscriptError(
            "No subtitle cues found. Expected SRT/VTT blocks with 'start --> end' lines."
        )
    return cues


def parse_vtt(text: str, source_duration: float) -> list[dict[str, Any]]:
    normalized = text.replace("\ufeff", "").replace("\r\n", "\n").replace("\r", "\n")
    lines = normalized.splitlines()
    kept: list[str] = []
    skipping = False
    for line in lines:
        stripped = line.strip()
        if stripped == "WEBVTT":
            continue
        if stripped.startswith(("NOTE", "STYLE", "REGION")):
            skipping = True
            continue
        if skipping:
            if not stripped:
                skipping = False
            continue
        if "-->" in line:
            line = re.sub(r"\s+align:[^\s]+.*$", "", line)
        kept.append(line)
    return parse_srt("\n".join(kept), source_duration)


def _parse_word_list(raw_words: Iterable[Any], source_duration: float, offset: float = 0.0) -> list[Word]:
    return _prepare_words(raw_words, source_duration, offset=offset)


def load_transcript_json(payload: Any, source_duration: float) -> Transcript:
    if isinstance(payload, list):
        payload = {"units": payload}
    if not isinstance(payload, dict):
        raise TranscriptError("Transcript JSON must be an object or a list of units.")

    language = str(payload.get("language") or "unknown")
    backend = str(payload.get("backend") or "imported-json")
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}

    raw_units = payload.get("units")
    if isinstance(raw_units, list) and raw_units:
        units: list[Unit] = []
        for item in raw_units:
            if not isinstance(item, dict):
                continue
            start = coerce_float(item.get("start"))
            end = coerce_float(item.get("end"))
            text = clean_text(item.get("text", ""))
            if start is None or end is None or not text:
                continue
            units.append(Unit(id="", start=start, end=end, text=text))
        units = _assign_ids(units, source_duration)
        if not units:
            raise TranscriptError("Transcript JSON contains no usable units.")
        return Transcript(
            language=language,
            units=units,
            backend=backend,
            source_duration=source_duration,
            metadata=metadata,
        )

    raw_words = payload.get("words")
    if isinstance(raw_words, list) and raw_words:
        words = _parse_word_list(raw_words, source_duration)
        units = build_units(words, source_duration, language=language)
        if not units:
            raise TranscriptError("Transcript JSON words produced no usable units.")
        return Transcript(language, units, backend, source_duration, metadata)

    segments = payload.get("segments")
    if isinstance(segments, list) and segments:
        flat_words: list[Word] = []
        for segment in segments:
            if not isinstance(segment, dict):
                continue
            segment_words = segment.get("words")
            if isinstance(segment_words, list):
                start = coerce_float(segment.get("start")) or 0.0
                # Whisper word times are already absolute; only use the segment
                # offset when a source supplied relative word times.
                absolute = all(
                    isinstance(word, dict)
                    and coerce_float(word.get("start", word.get("s"))) is not None
                    and float(coerce_float(word.get("start", word.get("s"))) or 0) >= start - 0.1
                    for word in segment_words
                )
                flat_words.extend(
                    _parse_word_list(segment_words, source_duration, offset=0.0 if absolute else start)
                )
        if flat_words:
            units = build_units(flat_words, source_duration, language=language)
        else:
            units = units_from_cues(
                [
                    {
                        "start": segment.get("start"),
                        "end": segment.get("end"),
                        "text": segment.get("text", ""),
                    }
                    for segment in segments
                    if isinstance(segment, dict)
                ],
                source_duration,
                language=language,
            )
        if not units:
            raise TranscriptError("Transcript segments produced no usable units.")
        return Transcript(language, units, backend, source_duration, metadata)

    raise TranscriptError(
        "Unsupported transcript JSON. Provide MomentCut units, Whisper-style "
        "segments/words, or a top-level words array."
    )


def load_transcript_file(path: str | Path, source_duration: float) -> Transcript:
    source = Path(path)
    if not source.exists():
        raise TranscriptError(f"Transcript file not found: {source}")
    suffix = source.suffix.lower()
    raw = source.read_text(encoding="utf-8-sig", errors="replace")
    if suffix == ".json":
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise TranscriptError(f"Invalid transcript JSON: {exc}") from exc
        transcript = load_transcript_json(payload, source_duration)
        transcript.metadata.setdefault("imported_from", str(source.resolve()))
        return transcript
    if suffix in {".srt", ".vtt"}:
        cues = parse_vtt(raw, source_duration) if suffix == ".vtt" else parse_srt(raw, source_duration)
        units = units_from_cues(cues, source_duration)
        if not units:
            raise TranscriptError("Subtitle file produced no usable transcript units.")
        return Transcript(
            language="unknown",
            units=units,
            backend=f"imported-{suffix.lstrip('.')}",
            source_duration=source_duration,
            metadata={"imported_from": str(source.resolve())},
        )
    # Last chance: a JSON file saved with a different extension.
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        raise TranscriptError(
            f"Unsupported transcript format '{source.suffix}'. Use .json, .srt, or .vtt."
        )
    transcript = load_transcript_json(payload, source_duration)
    transcript.metadata.setdefault("imported_from", str(source.resolve()))
    return transcript


def save_transcript(transcript: Transcript, path: str | Path) -> None:
    from .util import atomic_write_json

    atomic_write_json(path, transcript.to_dict())


def load_saved_transcript(path: str | Path) -> Transcript:
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise TranscriptError("Saved transcript is not a JSON object.")
    language = str(payload.get("language") or "unknown")
    backend = str(payload.get("backend") or "saved")
    duration = coerce_float(payload.get("source_duration")) or 0.0
    units: list[Unit] = []
    for item in payload.get("units") or []:
        if not isinstance(item, dict):
            continue
        start = coerce_float(item.get("start"))
        end = coerce_float(item.get("end"))
        text = clean_text(item.get("text", ""))
        unit_id = str(item.get("id") or "")
        if start is None or end is None or not text or not unit_id:
            continue
        units.append(Unit(unit_id, start, end, text))
    if not units:
        raise TranscriptError("Saved transcript contains no usable units.")
    units.sort(key=lambda unit: (unit.start, unit.end))
    return Transcript(
        language=language,
        units=units,
        backend=backend,
        source_duration=duration,
        metadata=payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {},
    )


def find_unit_index(transcript: Transcript, unit_id: str) -> int | None:
    for index, unit in enumerate(transcript.units):
        if unit.id == unit_id:
            return index
    return None


def transcript_preview(transcript: Transcript, max_units: int = 12) -> str:
    lines = [
        f"{unit.id} [{format_clock(unit.start)}-{format_clock(unit.end)}] {unit.text}"
        for unit in transcript.units[:max_units]
    ]
    if len(transcript.units) > max_units:
        lines.append(f"... {len(transcript.units) - max_units} more units")
    return "\n".join(lines)
