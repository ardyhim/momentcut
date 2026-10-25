"""Render validated analysis ranges with FFmpeg and verify the outputs."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .media import (
    MediaError,
    fingerprint_compatible,
    render_cut,
    source_fingerprint,
    verify_rendered_clip,
)
from .pipeline import (
    PipelineError,
    _load_run_state,
    _save_run_state,
    _verify_source_for_run,
)
from .util import (
    atomic_write_json,
    format_clock,
    read_json,
    slugify,
    utc_now,
)


@dataclass
class RenderResult:
    rendered: int
    skipped: int
    failed: int
    manifest_path: Path
    errors: list[str]


def _load_analysis(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "analysis.json"
    if not path.exists():
        raise PipelineError(
            f"No analysis found at {path}. Import provider responses first."
        )
    try:
        payload = read_json(path)
    except Exception as exc:
        raise PipelineError(f"Could not read {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise PipelineError(f"Invalid analysis manifest at {path}.")
    if str(payload.get("schema_version")) != "1.0":
        raise PipelineError(f"Unsupported analysis schema in {path}.")
    return payload


def _filename_for(clip: dict[str, Any], index: int, aspect: str) -> str:
    clip_id = str(clip.get("id") or f"C{index:03d}")
    title = str(clip.get("title") or "")
    suffix = "_vertical" if aspect == "vertical" else ""
    return f"{clip_id}_{slugify(title)}{suffix}.mp4"


def _valid_existing(
    path: Path,
    *,
    expected_duration: float,
    aspect: str,
    expect_audio: bool,
) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return verify_rendered_clip(
            path,
            expected_duration,
            expect_audio=expect_audio,
            vertical=aspect == "vertical",
        )
    except MediaError:
        return None


def render_run(
    run_dir: str | Path,
    *,
    source_override: str | Path | None = None,
    aspect: str = "original",
    force: bool = False,
) -> RenderResult:
    if aspect not in ("original", "vertical"):
        raise PipelineError("aspect must be 'original' or 'vertical'.")
    run_path = Path(run_dir)
    state = _load_run_state(run_path)
    analysis = _load_analysis(run_path)
    source_path, source_duration, has_audio = _verify_source_for_run(
        state, source_override, allow_missing=False
    )
    if source_path is None:
        raise PipelineError("Source video is required for rendering.")
    # Confirm this is the source the analysis used, not merely the run's source.
    analysis_source = analysis.get("source") or {}
    current = source_fingerprint(source_path)
    if analysis_source and not fingerprint_compatible(analysis_source, current):
        raise PipelineError(
            "The current source file does not match the file used for analysis. "
            "Refusing to render mismatched timestamps."
        )

    clips = analysis.get("clips") or []
    if not isinstance(clips, list) or not clips:
        raise PipelineError(
            "This analysis selected no clips. Review analysis/report.md; if the "
            "provider output is valid, you can re-run import with a lower --min-score."
        )
    clips_dir = run_path / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_path / "render_manifest.json"
    previous: dict[str, Any] = {}
    manifest_created_at = utc_now()
    if manifest_path.exists() and not force:
        try:
            existing = read_json(manifest_path)
            if isinstance(existing, dict):
                manifest_created_at = str(existing.get("created_at") or manifest_created_at)
            if (
                isinstance(existing, dict)
                and fingerprint_compatible(
                    existing.get("source") or {}, current
                )
            ):
                previous = existing.get("clips") or {}
        except Exception:
            previous = {}

    manifest: dict[str, Any] = {
        "schema_version": "1.0",
        "created_at": manifest_created_at,
        "updated_at": utc_now(),
        "source": current,
        "aspect": aspect,
        "clips": dict(previous),
    }
    rendered = 0
    skipped = 0
    failed = 0
    errors: list[str] = []

    for index, clip in enumerate(clips, start=1):
        clip_id = str(clip.get("id") or f"C{index:03d}")
        manifest_key = f"{clip_id}:{aspect}"
        start = float(clip.get("start"))
        end = float(clip.get("end"))
        duration = end - start
        title = str(clip.get("title") or clip_id)
        if start < 0 or end <= start or end > source_duration + 0.01:
            message = (
                f"{clip_id}: invalid range {start:.3f}-{end:.3f} against source "
                f"duration {source_duration:.3f}"
            )
            errors.append(message)
            failed += 1
            manifest["clips"][manifest_key] = {
                "id": clip_id,
                "title": title,
                "status": "failed",
                "aspect": aspect,
                "error": message,
            }
            atomic_write_json(manifest_path, manifest)
            continue

        filename = _filename_for(clip, index, aspect)
        destination = clips_dir / filename
        relative = destination.relative_to(run_path).as_posix()
        previous_entry = manifest["clips"].get(manifest_key) or {}
        if not force and previous_entry.get("status") == "rendered":
            valid = _valid_existing(
                destination,
                expected_duration=duration,
                aspect=aspect,
                expect_audio=has_audio,
            )
            if valid is not None:
                skipped += 1
                print(f"[render] {clip_id}: reuse existing {relative}")
                previous_entry.update(
                    {
                        "path": relative,
                        "verified": valid,
                        "updated_at": utc_now(),
                    }
                )
                manifest["clips"][manifest_key] = previous_entry
                continue

        print(
            f"[render] {clip_id}: {format_clock(start)}-{format_clock(end)} "
            f"({duration:.1f}s) -> {relative}"
        )
        try:
            render_cut(
                source_path,
                destination,
                start,
                duration,
                aspect=aspect,
                has_audio=has_audio,
            )
            verified = verify_rendered_clip(
                destination,
                duration,
                expect_audio=has_audio,
                vertical=aspect == "vertical",
            )
            rendered += 1
            manifest["clips"][manifest_key] = {
                "id": clip_id,
                "title": title,
                "status": "rendered",
                "path": relative,
                "start": round(start, 3),
                "end": round(end, 3),
                "duration": round(duration, 3),
                "aspect": aspect,
                "verified": verified,
                "updated_at": utc_now(),
            }
        except Exception as exc:
            failed += 1
            message = f"{clip_id}: {exc}"
            errors.append(message)
            print(f"[error] {message}", file=sys.stderr)
            manifest["clips"][manifest_key] = {
                "id": clip_id,
                "title": title,
                "status": "failed",
                "path": relative,
                "aspect": aspect,
                "error": str(exc),
                "updated_at": utc_now(),
            }
            try:
                if destination.exists():
                    destination.unlink()
            except OSError:
                pass
        manifest["updated_at"] = utc_now()
        atomic_write_json(manifest_path, manifest)

    manifest["updated_at"] = utc_now()
    current_clip_ids = {
        str(clip.get("id") or f"C{index:03d}")
        for index, clip in enumerate(clips, start=1)
    }
    manifest["clips"] = {
        key: value
        for key, value in manifest["clips"].items()
        if key.split(":", 1)[0] in current_clip_ids
    }
    atomic_write_json(manifest_path, manifest)
    state["last_render_at"] = utc_now()
    _save_run_state(run_path, state)
    print(
        f"[render] rendered {rendered}, skipped {skipped}, failed {failed}; "
        f"manifest: {manifest_path}"
    )
    return RenderResult(
        rendered=rendered,
        skipped=skipped,
        failed=failed,
        manifest_path=manifest_path,
        errors=errors,
    )
