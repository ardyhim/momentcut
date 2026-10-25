"""Resumable faster-whisper transcription.

The local model is the only mandatory model in MomentCut. Audio is split
into overlapping chunks; each completed chunk is a JSON checkpoint, so a lost
Colab runtime does not throw away the whole transcription.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

from .media import extract_audio_chunk
from .transcript import (
    Transcript,
    Word,
    build_units,
    merge_overlapping_words,
    units_from_cues,
)
from .util import (
    MomentCutError,
    atomic_write_json,
    format_clock,
    sha256_text,
    utc_now,
)


class AsrError(MomentCutError):
    pass


def plan_chunks(
    duration: float,
    chunk_seconds: float = 600.0,
    *,
    overlap_seconds: float = 3.0,
    min_tail_seconds: float = 30.0,
) -> list[tuple[float, float]]:
    """Plan overlapping chunks that cover the whole source without a tiny tail."""
    duration = max(0.0, float(duration))
    if duration <= 0:
        raise AsrError("Cannot plan audio chunks for a zero-length source.")
    chunk_seconds = float(chunk_seconds)
    if chunk_seconds <= 0 or duration <= chunk_seconds:
        return [(0.0, round(duration, 3))]

    overlap_seconds = max(0.0, min(float(overlap_seconds), chunk_seconds / 2))
    starts = [0.0]
    while starts[-1] + chunk_seconds < duration:
        next_start = starts[-1] + chunk_seconds - overlap_seconds
        if next_start <= starts[-1]:
            break
        if duration - next_start < min_tail_seconds:
            break
        starts.append(round(next_start, 3))

    ranges: list[tuple[float, float]] = []
    for index, start in enumerate(starts):
        end = duration if index == len(starts) - 1 else min(start + chunk_seconds, duration)
        if end - start >= 0.05:
            ranges.append((round(start, 3), round(end, 3)))
    if not ranges:
        return [(0.0, round(duration, 3))]
    if ranges[-1][1] < duration - 0.05:
        # The loop should make this unreachable, but never silently omit audio.
        ranges.append((ranges[-1][1], round(duration, 3)))
    return ranges


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def _model_config(model_name: str) -> tuple[str, str, str]:
    """Return (device, compute_type, backend_label), preferring CUDA when usable."""
    device = "cpu"
    compute_type = "int8"
    try:
        import ctranslate2  # type: ignore

        if ctranslate2.get_cuda_device_count() > 0:
            device = "cuda"
            compute_type = "float16"
    except Exception:
        pass
    return device, compute_type, f"faster-whisper:{model_name}:{device}"


def _load_model(model_name: str):
    download_root = os.environ.get("MOMENTCUT_MODEL_DIR") or None
    if download_root:
        # Keep the downloaded model with the user's chosen persistent directory.
        # `download_root` below is the primary control; these variables also keep
        # Hugging Face metadata from landing in a read-only home directory.
        os.environ.setdefault("HF_HOME", download_root)
        os.environ.setdefault("HUGGINGFACE_HUB_CACHE", download_root)
    try:
        from faster_whisper import WhisperModel  # type: ignore
    except ImportError as exc:
        raise AsrError(
            "faster-whisper is not installed. Run "
            "`pip install -r requirements.txt` or install `faster-whisper`."
        ) from exc

    device, compute_type, _ = _model_config(model_name)
    print(
        f"[asr] loading faster-whisper '{model_name}' on {device} "
        f"({compute_type}); the model is downloaded on first use"
    )
    try:
        return (
            WhisperModel(
                model_name,
                device=device,
                compute_type=compute_type,
                download_root=download_root,
            ),
            device,
            compute_type,
        )
    except Exception as exc:
        if device == "cuda":
            print(
                f"[warn] CUDA Whisper failed ({type(exc).__name__}: {exc}); "
                "falling back to CPU int8.",
                file=sys.stderr,
            )
            return (
                WhisperModel(
                    model_name,
                    device="cpu",
                    compute_type="int8",
                    download_root=download_root,
                ),
                "cpu",
                "int8",
            )
        raise AsrError(f"Could not load faster-whisper model '{model_name}': {exc}") from exc


def _transcribe_one_chunk(
    model: Any,
    wav_path: Path,
    *,
    offset: float,
    expected_duration: float,
    language: str | None,
) -> dict[str, Any]:
    segments_iter, info = model.transcribe(
        str(wav_path),
        language=language or None,
        beam_size=5,
        vad_filter=True,
        condition_on_previous_text=False,
        word_timestamps=True,
    )
    words: list[dict[str, Any]] = []
    segments: list[dict[str, Any]] = []
    last_report = -1.0
    for segment in segments_iter:
        seg_start = max(0.0, float(segment.start) + offset)
        seg_end = min(offset + expected_duration, float(segment.end) + offset)
        text = str(getattr(segment, "text", "") or "").strip()
        raw_words = getattr(segment, "words", None) or []
        chunk_words: list[dict[str, Any]] = []
        for word in raw_words:
            word_text = str(getattr(word, "word", "") or "")
            start = float(getattr(word, "start", 0.0)) + offset
            end = float(getattr(word, "end", 0.0)) + offset
            if not word_text.strip() or end <= start:
                continue
            entry = {
                "text": word_text,
                "start": round(start, 3),
                "end": round(end, 3),
            }
            probability = getattr(word, "probability", None)
            if probability is not None:
                entry["probability"] = round(float(probability), 4)
            words.append(entry)
            chunk_words.append(entry)
        if text:
            segments.append(
                {
                    "start": round(seg_start, 3),
                    "end": round(seg_end, 3),
                    "text": text,
                    "words": chunk_words,
                }
            )
        if seg_end - last_report >= 60.0:
            last_report = seg_end
            print(
                f"    {format_clock(seg_end, milliseconds=False)} / "
                f"{format_clock(offset + expected_duration, milliseconds=False)}"
            )

    detected_language = str(getattr(info, "language", "") or language or "unknown")
    probability = getattr(info, "language_probability", None)
    return {
        "schema_version": "1.0",
        "language": detected_language,
        "language_probability": round(float(probability), 4) if probability is not None else None,
        "words": words,
        "segments": segments,
    }


def _cache_key(
    *,
    model_name: str,
    source_fingerprint: dict[str, Any],
    chunk_seconds: float,
    language: str | None,
) -> str:
    payload = {
        "model": model_name,
        "quick_hash": source_fingerprint.get("quick_hash"),
        "size": source_fingerprint.get("size"),
        "duration": source_fingerprint.get("duration"),
        "chunk_seconds": chunk_seconds,
        "language": language or "auto",
        "asr_schema": 1,
    }
    return sha256_text(json.dumps(payload, sort_keys=True))[:16]


def transcribe_media(
    source: str | Path,
    *,
    duration: float,
    source_fingerprint: dict[str, Any],
    cache_dir: str | Path,
    model_name: str = "small",
    chunk_seconds: float = 600.0,
    language: str | None = None,
    has_audio: bool = True,
) -> Transcript:
    if not has_audio:
        raise AsrError(
            "The source has no audio stream, so speech transcription is impossible. "
            "Supply a transcript only if you have one from elsewhere; visual-only "
            "moment detection is not part of this baseline."
        )

    cache_root = Path(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)
    key = _cache_key(
        model_name=model_name,
        source_fingerprint=source_fingerprint,
        chunk_seconds=chunk_seconds,
        language=language,
    )
    work_dir = cache_root / f"chunks-{key}"
    work_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = work_dir / "manifest.json"
    manifest = {
        "schema_version": "1.0",
        "created_at": utc_now(),
        "model": model_name,
        "source": source_fingerprint,
        "chunk_seconds": chunk_seconds,
        "language": language,
        "plan": plan_chunks(duration, chunk_seconds),
    }
    atomic_write_json(manifest_path, manifest)
    chunks = plan_chunks(duration, chunk_seconds)
    print(
        f"[asr] {len(chunks)} audio chunk(s) planned; completed chunks are reusable "
        f"from {work_dir}"
    )

    model = None
    device = "cpu"
    compute_type = "int8"
    active_language = language or None
    chunk_payloads: list[dict[str, Any]] = []
    started = time.time()

    for index, (start, end) in enumerate(chunks, start=1):
        chunk_duration = end - start
        chunk_json = work_dir / f"chunk_{index:04d}.json"
        payload: dict[str, Any] | None = None
        if chunk_json.exists():
            try:
                candidate = _read_json(chunk_json)
                if isinstance(candidate, dict) and (
                    "words" in candidate or "segments" in candidate
                ):
                    payload = candidate
                    print(
                        f"[asr] chunk {index}/{len(chunks)} "
                        f"{format_clock(start, milliseconds=False)}-"
                        f"{format_clock(end, milliseconds=False)}: reuse checkpoint"
                    )
            except Exception as exc:
                print(
                    f"[warn] ignoring unreadable ASR checkpoint {chunk_json.name}: {exc}",
                    file=sys.stderr,
                )
                payload = None

        if payload is None:
            if model is None:
                model, device, compute_type = _load_model(model_name)
            print(
                f"[asr] chunk {index}/{len(chunks)} "
                f"{format_clock(start, milliseconds=False)}-"
                f"{format_clock(end, milliseconds=False)} ({chunk_duration:.0f}s)"
            )
            wav_path = work_dir / f"chunk_{index:04d}.wav"
            try:
                extract_audio_chunk(source, start, chunk_duration, wav_path)
                payload = _transcribe_one_chunk(
                    model,
                    wav_path,
                    offset=start,
                    expected_duration=chunk_duration,
                    language=active_language,
                )
                payload["chunk_index"] = index
                payload["chunk_start"] = round(start, 3)
                payload["chunk_end"] = round(end, 3)
                payload["model"] = model_name
                atomic_write_json(chunk_json, payload)
            finally:
                try:
                    wav_path.unlink()
                except OSError:
                    pass

        if payload is None:
            continue
        chunk_payloads.append(payload)
        if active_language is None:
            detected = str(payload.get("language") or "").strip()
            if detected and detected != "unknown":
                active_language = detected

    if not chunk_payloads:
        raise AsrError("Audio extraction/transcription produced no chunks.")

    word_chunks: list[list[Word]] = []
    fallback_segments: list[dict[str, Any]] = []
    for payload in chunk_payloads:
        converted: list[Word] = []
        for item in payload.get("words") or []:
            if not isinstance(item, dict):
                continue
            start = item.get("start")
            end = item.get("end")
            text = str(item.get("text") or "")
            try:
                start_value = float(start)
                end_value = float(end)
            except (TypeError, ValueError):
                continue
            if not text.strip() or end_value <= start_value:
                continue
            converted.append(Word(text=text, start=start_value, end=end_value))
        if converted:
            word_chunks.append(converted)
        for segment in payload.get("segments") or []:
            if isinstance(segment, dict) and segment.get("text"):
                fallback_segments.append(segment)

    merged_words = merge_overlapping_words(word_chunks) if word_chunks else []
    if merged_words:
        units = build_units(merged_words, duration, language=active_language)
    else:
        units = units_from_cues(fallback_segments, duration, language=active_language)
    if not units:
        raise AsrError(
            "Transcription produced no usable speech units. The audio may be music, "
            "silence, or too noisy. Supply a transcript if one exists."
        )

    elapsed = time.time() - started
    backend = f"faster-whisper:{model_name}:{device}"
    print(
        f"[asr] done: {len(units)} units, language={active_language or 'unknown'}, "
        f"{elapsed:.1f}s"
    )
    return Transcript(
        language=active_language or "unknown",
        units=units,
        backend=backend,
        source_duration=duration,
        metadata={
            "model": model_name,
            "device": device,
            "compute_type": compute_type,
            "chunks": len(chunk_payloads),
            "chunk_seconds": chunk_seconds,
            "elapsed_seconds": round(elapsed, 1),
        },
    )


def clear_asr_cache(cache_dir: str | Path) -> None:
    """Remove ASR chunk caches created by this tool. Used only by explicit CLI."""
    target = Path(cache_dir)
    if target.exists():
        shutil.rmtree(target)
