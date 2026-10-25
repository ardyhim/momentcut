"""Run lifecycle: analyze, import provider answers, and persist analysis."""

from __future__ import annotations

import json
import sys
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import PROMPT_VERSION, SCHEMA_VERSION
from .asr import transcribe_media
from .config import ApiSettings, RunSettings
from .contract import ParsedResponse, load_response_file
from .media import (
    MediaInfo,
    fingerprint_compatible,
    probe_media,
    source_fingerprint,
)
from .prompting import (
    Block,
    load_blocks,
    write_prompt_bundle,
)
from .resolve import RejectedClip, ResolvedClip, resolve_responses
from .select import SelectionResult, select_clips
from .transcript import (
    Transcript,
    load_saved_transcript,
    load_transcript_file,
    save_transcript,
)
from .util import (
    MomentCutError,
    atomic_write_json,
    atomic_write_text,
    clean_text,
    format_clock,
    human_duration,
    read_json,
    sha256_text,
    utc_now,
)

RUN_SCHEMA_VERSION = "1.0"


class PipelineError(MomentCutError):
    pass


@dataclass
class AnalyzeResult:
    run_dir: Path
    transcript: Transcript
    blocks: list[Block]
    api_errors: list[dict[str, Any]]
    import_result: ImportResult | None
    api_complete: bool


@dataclass
class ImportResult:
    analysis_path: Path
    report_path: Path
    selected: int
    resolved: int
    rejected: int
    dropped: int
    parsed_candidates: int
    coverage: dict[str, Any]
    response_files: int
    warnings: list[str]


def _run_file(run_dir: str | Path) -> Path:
    return Path(run_dir) / "run.json"


def _new_run_state(
    source: Path,
    info: MediaInfo,
    fingerprint: dict[str, Any],
    settings: RunSettings,
) -> dict[str, Any]:
    return {
        "schema_version": RUN_SCHEMA_VERSION,
        "app": "MomentCut",
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "source": fingerprint,
        "media": info.to_dict(),
        "settings": settings.to_dict(),
        "transcript": None,
        "prompt_bundle_key": None,
        "last_import_at": None,
        "api": None,
    }


def _load_run_state(run_dir: str | Path) -> dict[str, Any]:
    path = _run_file(run_dir)
    if not path.exists():
        raise PipelineError(
            f"No run found at {Path(run_dir)}. Run `analyze` first."
        )
    try:
        payload = read_json(path)
    except (OSError, json.JSONDecodeError) as exc:
        raise PipelineError(f"Could not read {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise PipelineError(f"Invalid run manifest at {path}.")
    if str(payload.get("schema_version")) != RUN_SCHEMA_VERSION:
        raise PipelineError(
            f"Unsupported run schema {payload.get('schema_version')!r} in {path}."
        )
    return payload


def _save_run_state(run_dir: str | Path, state: dict[str, Any]) -> None:
    state["updated_at"] = utc_now()
    atomic_write_json(_run_file(run_dir), state)


def _mkdirs(run_dir: Path) -> None:
    (run_dir / "analysis" / "prompts").mkdir(parents=True, exist_ok=True)
    (run_dir / "analysis" / "responses").mkdir(parents=True, exist_ok=True)
    (run_dir / "clips").mkdir(parents=True, exist_ok=True)


def _prepare_run(
    run_dir: Path,
    source: Path,
    info: MediaInfo,
    fingerprint: dict[str, Any],
    overrides: dict[str, Any],
    *,
    allow_path_update: bool = True,
) -> tuple[dict[str, Any], RunSettings, list[str]]:
    warnings: list[str] = []
    if run_dir.exists() and _run_file(run_dir).exists():
        state = _load_run_state(run_dir)
        stored_fingerprint = state.get("source") or {}
        if not fingerprint_compatible(stored_fingerprint, fingerprint):
            raise PipelineError(
                "This run belongs to a different source video. Use a new --run "
                "directory (or import/analyze under a new run) rather than mixing "
                "timestamps."
            )
        settings = RunSettings.from_dict(state.get("settings"))
        provided = {key: value for key, value in overrides.items() if value is not None}
        if provided:
            settings = replace(settings, **provided)
        settings.validate()
        if allow_path_update:
            state["source"] = fingerprint
            state["media"] = info.to_dict()
        _mkdirs(run_dir)
        return state, settings, warnings

    if run_dir.exists():
        contents = list(run_dir.iterdir())
        if contents:
            raise PipelineError(
                f"Run directory {run_dir} exists but has no run.json. Use an empty "
                "directory or a new --run path."
            )
    run_dir.mkdir(parents=True, exist_ok=True)
    _mkdirs(run_dir)
    settings = RunSettings()
    provided = {key: value for key, value in overrides.items() if value is not None}
    if provided:
        settings = replace(settings, **provided)
    settings.validate()
    return _new_run_state(source, info, fingerprint, settings), settings, warnings


def _verify_source_for_run(
    state: dict[str, Any],
    source_override: str | Path | None,
    *,
    allow_missing: bool,
) -> tuple[Path | None, float, bool]:
    recorded = state.get("source") or {}
    path = Path(source_override) if source_override else Path(str(recorded.get("path") or ""))
    if not path or not path.exists():
        if allow_missing:
            duration = float(state.get("media", {}).get("duration") or 0.0)
            has_audio = bool(state.get("media", {}).get("has_audio", True))
            if duration <= 0:
                raise PipelineError("Run has no usable source duration.")
            return None, duration, has_audio
        raise PipelineError(
            f"Source video not found at {path or '(unknown)'}. Pass --source with "
            "the current location of the same file."
        )
    info = probe_media(path)
    current = source_fingerprint(path, info)
    if not fingerprint_compatible(recorded, current):
        raise PipelineError(
            "The source file at this location is not the file this run analyzed "
            "(size/duration/content fingerprint differs). Refusing to reuse "
            "timestamps against a different video."
        )
    return path, info.duration, info.has_audio


def _validate_transcript(transcript: Transcript, source_duration: float) -> Transcript:
    if not transcript.units:
        raise PipelineError("Transcript has no units.")
    seen: set[str] = set()
    units = []
    for unit in transcript.units:
        if not unit.id or unit.id in seen:
            raise PipelineError(f"Transcript contains a duplicate or empty line ID: {unit.id!r}")
        seen.add(unit.id)
        if not unit.text.strip():
            raise PipelineError(f"Transcript line {unit.id} has no text.")
        if unit.start < 0 or unit.end <= unit.start:
            raise PipelineError(
                f"Transcript line {unit.id} has an invalid range "
                f"{unit.start:.3f}-{unit.end:.3f}."
            )
        if unit.start > source_duration + 0.5:
            raise PipelineError(
                f"Transcript line {unit.id} starts after the source duration."
            )
        end = unit.end
        if end > source_duration + 0.5:
            raise PipelineError(
                f"Transcript line {unit.id} ends at {end:.3f}s, beyond the source "
                f"duration {source_duration:.3f}s."
            )
        if end > source_duration:
            end = source_duration
        units.append(type(unit)(unit.id, unit.start, end, unit.text))
    transcript.units = units
    transcript.source_duration = source_duration
    return transcript


def _transcript_digest(transcript: Transcript) -> str:
    payload = transcript.to_dict()
    return sha256_text(json.dumps(payload, ensure_ascii=False, sort_keys=True))


def _prompt_bundle_key(transcript: Transcript, settings: RunSettings) -> str:
    payload = {
        "prompt_version": PROMPT_VERSION,
        "transcript": _transcript_digest(transcript),
        "min_duration": round(float(settings.min_duration), 3),
        "max_duration": round(float(settings.max_duration), 3),
        "content_type": settings.content_type,
        "focus": settings.focus,
        "block_seconds": round(float(settings.block_seconds), 3),
        "language": transcript.language,
    }
    return sha256_text(json.dumps(payload, ensure_ascii=False, sort_keys=True))


def _prompt_files_present(run_dir: Path, blocks: Sequence[Block]) -> bool:
    return all((run_dir / block.prompt_file).exists() for block in blocks)


def _archive_responses(run_dir: Path, reason: str) -> None:
    responses = run_dir / "analysis" / "responses"
    if not responses.exists():
        responses.mkdir(parents=True, exist_ok=True)
        return
    files = [path for path in responses.rglob("*") if path.is_file()]
    if not files:
        return
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    destination = run_dir / "analysis" / f"responses.stale-{stamp}"
    suffix = 1
    while destination.exists():
        destination = run_dir / "analysis" / f"responses.stale-{stamp}-{suffix}"
        suffix += 1
    responses.rename(destination)
    responses.mkdir(parents=True, exist_ok=True)
    print(
        f"[warn] {reason}; previous responses were preserved at {destination.name} "
        "and are no longer imported automatically.",
        file=sys.stderr,
    )


def _load_or_create_transcript(
    run_dir: Path,
    state: dict[str, Any],
    source: Path,
    info: MediaInfo,
    fingerprint: dict[str, Any],
    settings: RunSettings,
    transcript_path: str | Path | None,
    force_transcribe: bool,
) -> tuple[Transcript, str]:
    saved_path = run_dir / "transcript.json"
    if transcript_path:
        transcript = load_transcript_file(transcript_path, info.duration)
        if settings.language and transcript.language in ("unknown", "", None):
            transcript.language = settings.language
        transcript = _validate_transcript(transcript, info.duration)
        save_transcript(transcript, saved_path)
        print(
            f"[transcript] imported {len(transcript.units)} units from "
            f"{Path(transcript_path).name}"
        )
        return transcript, "imported"

    if saved_path.exists() and not force_transcribe:
        try:
            transcript = load_saved_transcript(saved_path)
            transcript = _validate_transcript(transcript, info.duration)
            if settings.language and transcript.language in ("unknown", "", None):
                transcript.language = settings.language
                save_transcript(transcript, saved_path)
            print(
                f"[transcript] reused {len(transcript.units)} units from "
                f"{saved_path.name} ({transcript.language or 'unknown'})"
            )
            return transcript, "reused"
        except Exception as exc:
            print(
                f"[warn] saved transcript is unusable ({exc}); transcribing again.",
                file=sys.stderr,
            )

    if not info.has_audio:
        raise PipelineError(
            "The source has no audio stream. Supply --transcript if you have an "
            "external transcript; visual-only analysis is not in this baseline."
        )
    transcript = transcribe_media(
        source,
        duration=info.duration,
        source_fingerprint=fingerprint,
        cache_dir=run_dir / "asr",
        model_name=settings.asr_model,
        chunk_seconds=settings.asr_chunk_seconds,
        language=settings.language,
        has_audio=info.has_audio,
    )
    transcript = _validate_transcript(transcript, info.duration)
    save_transcript(transcript, saved_path)
    return transcript, "asr"


def _response_files(run_dir: Path, explicit: Sequence[str | Path] | None) -> list[Path]:
    paths: list[Path] = []
    if explicit:
        for item in explicit:
            path = Path(item)
            if path.is_dir():
                paths.extend(
                    candidate
                    for candidate in path.rglob("*")
                    if candidate.is_file()
                    and candidate.suffix.lower() in {".json", ".txt", ".md"}
                    and not candidate.name.startswith(".")
                    and not candidate.name.lower().startswith("readme")
                )
            elif path.is_file():
                paths.append(path)
            else:
                raise PipelineError(f"Response path not found: {path}")
    else:
        responses_dir = run_dir / "analysis" / "responses"
        if responses_dir.exists():
            paths.extend(
                candidate
                for candidate in responses_dir.rglob("*")
                if candidate.is_file()
                and candidate.suffix.lower() in {".json", ".txt", ".md"}
                and not candidate.name.startswith(".")
                and not candidate.name.lower().startswith("readme")
            )
    unique = sorted({path.resolve() for path in paths})
    return unique


def _write_report(
    path: Path,
    *,
    run_dir: Path,
    source: dict[str, Any],
    settings: RunSettings,
    coverage: dict[str, Any],
    selected: Sequence[ResolvedClip],
    rejected: Sequence[RejectedClip],
    selection: SelectionResult,
    parsed_candidates: int,
    warnings: Sequence[str],
    language: str = "unknown",
) -> None:
    lines: list[str] = []
    lines.append("# MomentCut analysis report")
    lines.append("")
    lines.append(f"- Source: `{source.get('path', '')}`")
    lines.append(f"- Duration: {human_duration(float(source.get('duration') or 0))}")
    lines.append(f"- Language: {language or 'unknown'}")
    lines.append(
        f"- Requested clip band: {settings.min_duration:g}–{settings.max_duration:g}s; "
        f"minimum score {settings.min_score}; max clips {settings.max_clips}"
    )
    lines.append(
        f"- Coverage: {coverage['blocks_responded']}/{coverage['blocks_total']} blocks"
    )
    if coverage.get("missing_blocks"):
        lines.append(
            "- Missing blocks: " + ", ".join(str(item) for item in coverage["missing_blocks"])
        )
    lines.append(
        f"- Candidates parsed: {parsed_candidates}; resolved: "
        f"{len(selected) + selection.low_context + selection.below_score + selection.duplicate_count + selection.over_limit}; "
        f"selected: {len(selected)}; rejected: {len(rejected)}"
    )
    lines.append("")
    if warnings:
        lines.append("## Warnings")
        lines.append("")
        for warning in warnings:
            lines.append(f"- {warning}")
        lines.append("")
    lines.append("## Selected clips")
    lines.append("")
    if not selected:
        lines.append("_No clip passed the quality threshold and validation rules._")
    else:
        lines.append("| Rank | ID | Time | Dur | Score | Title | Warnings |")
        lines.append("|---:|---|---:|---:|---:|---|---|")
        for clip in selected:
            title = clean_text(clip.metadata.get("title", "")).replace("|", "\\|")
            warning_count = len(clip.warnings)
            lines.append(
                f"| {clip.selection_rank} | {clip.clip_id} | "
                f"{format_clock(clip.start)}–{format_clock(clip.end)} | "
                f"{clip.duration:.1f}s | {clip.selection_score} | {title} | {warning_count} |"
            )
        lines.append("")
        lines.append("### Clip details")
        lines.append("")
        for clip in selected:
            lines.append(f"#### {clip.clip_id} — {clip.metadata.get('title', '')}")
            lines.append("")
            lines.append(f"- Time: {format_clock(clip.start)}–{format_clock(clip.end)} ({clip.duration:.2f}s)")
            lines.append(f"- Hook: {clip.metadata.get('hook', '')}")
            lines.append(f"- Caption: {clip.metadata.get('caption', '')}")
            lines.append(f"- Why: {clip.metadata.get('why', '')}")
            lines.append(f"- Lines: {clip.start_line}–{clip.end_line} ({clip.block_id or 'unknown block'})")
            lines.append(f"- Scores: {clip.candidate.scores}")
            if clip.warnings:
                for warning in clip.warnings:
                    lines.append(f"- Warning: {warning}")
            lines.append("")
    lines.append("## Rejected candidates")
    lines.append("")
    if not rejected:
        lines.append("_None._")
    else:
        reason_counts = Counter(item.reason.split(" (")[0] for item in rejected)
        for reason, count in reason_counts.most_common():
            lines.append(f"- {count} × {reason}")
        lines.append("")
        for item in rejected[:100]:
            title = ""
            if item.candidate:
                title = str(item.candidate.get("title") or "")
            label = f" — {title}" if title else ""
            lines.append(f"- `{Path(item.source_file).name}`{label}: {item.reason}")
        if len(rejected) > 100:
            lines.append(f"- … {len(rejected) - 100} more in `analysis.json`")
    lines.append("")
    lines.append("## Dropped after validation")
    lines.append("")
    if not selection.dropped:
        lines.append("_None._")
    else:
        for item in selection.dropped:
            title = clean_text(item.clip.metadata.get("title", ""))
            lines.append(f"- {title or '(untitled)'} ({format_clock(item.clip.start)}): {item.reason}")
    lines.append("")
    lines.append(f"_Run directory: `{run_dir}`_")
    lines.append("")
    atomic_write_text(path, "\n".join(lines))


def import_analysis(
    run_dir: str | Path,
    response_paths: Sequence[str | Path] | None = None,
    *,
    source_override: str | Path | None = None,
    min_score: int | None = None,
    max_clips: int | None = None,
    quiet: bool = False,
) -> ImportResult:
    run_path = Path(run_dir)
    state = _load_run_state(run_path)
    settings = RunSettings.from_dict(state.get("settings"))
    if min_score is not None:
        settings.min_score = min_score
    if max_clips is not None:
        settings.max_clips = max_clips
    settings.validate()

    source_path, source_duration, _has_audio = _verify_source_for_run(
        state, source_override, allow_missing=True
    )
    if source_path is None:
        print(
            "[warn] source video is not available; using the duration recorded in "
            "run.json. Rendering will still require --source.",
            file=sys.stderr,
        )

    transcript_path = run_path / "transcript.json"
    if not transcript_path.exists():
        raise PipelineError(f"Transcript missing at {transcript_path}. Re-run analyze.")
    transcript = load_saved_transcript(transcript_path)
    transcript = _validate_transcript(transcript, source_duration)
    blocks_path = run_path / "analysis" / "blocks.json"
    if not blocks_path.exists():
        raise PipelineError(f"Block manifest missing at {blocks_path}. Re-run analyze.")
    blocks = load_blocks(blocks_path)

    files = _response_files(run_path, response_paths)
    if not files:
        raise PipelineError(
            "No provider response files found. Save each answer as .json/.txt/.md "
            f"under {run_path / 'analysis' / 'responses'} or pass paths explicitly."
        )

    parsed: list[ParsedResponse] = []
    parse_errors: list[dict[str, Any]] = []
    for file in files:
        try:
            result = load_response_file(file)
            parsed.append(result)
        except Exception as exc:
            parse_errors.append(
                {
                    "source_file": str(file),
                    "reason": str(exc),
                }
            )

    resolved, rejected = resolve_responses(
        parsed, transcript, blocks, source_duration, settings
    )
    for item in parse_errors:
        rejected.append(
            RejectedClip(
                source_file=item["source_file"],
                block_id=None,
                reason=item["reason"],
            )
        )

    selection = select_clips(resolved, settings)

    responded: set[str] = set()
    block_ids = {block.id for block in blocks}
    response_metadata: list[dict[str, Any]] = []
    coverage_warnings: list[str] = []
    for response in parsed:
        if response.block_id and response.block_id in block_ids:
            responded.add(response.block_id)
        elif response.block_id:
            coverage_warnings.append(
                f"{Path(response.source_file).name}: block_id {response.block_id!r} "
                "does not exist in this run"
            )
        response_metadata.append(
            {
                "file": str(Path(response.source_file)),
                "block_id": response.block_id,
                "candidates": len(response.candidates),
                "errors": response.errors,
                "notes": response.notes,
            }
        )
        if not response.block_id and not response.candidates:
            coverage_warnings.append(
                f"{Path(response.source_file).name}: empty answer with no block_id; "
                "cannot attribute it to a block"
            )
    for clip in resolved:
        if clip.block_id in block_ids:
            responded.add(clip.block_id)
    for error in parse_errors:
        coverage_warnings.append(
            f"{Path(error['source_file']).name}: {error['reason']}"
        )

    missing_blocks = sorted(block_ids - responded)
    coverage = {
        "blocks_total": len(blocks),
        "blocks_responded": len(responded),
        "missing_blocks": missing_blocks,
        "complete": not missing_blocks,
    }
    warnings = list(coverage_warnings)
    if missing_blocks:
        warnings.append(
            "Coverage is incomplete; no clip can come from these blocks: "
            + ", ".join(missing_blocks)
        )
    if not selection.selected and resolved:
        warnings.append(
            "No clip survived the quality filters (minimum score, cold-viewer "
            "context floor, duplicate suppression). The analysis is valid; review "
            "the dropped reasons before lowering any threshold."
        )

    parsed_candidates = sum(len(response.candidates) for response in parsed)
    analysis_payload = {
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now(),
        "source": state.get("source"),
        "media": state.get("media"),
        "settings": settings.to_dict(),
        "transcript": {
            "language": transcript.language,
            "backend": transcript.backend,
            "units": len(transcript.units),
            "source_duration": transcript.source_duration,
        },
        "coverage": coverage,
        "responses": response_metadata,
        "counts": {
            "response_files": len(files),
            "parsed_candidates": parsed_candidates,
            "resolved": len(resolved),
            "selected": len(selection.selected),
            "rejected": len(rejected),
            "dropped": len(selection.dropped),
            "low_standalone": selection.low_context,
            "below_min_score": selection.below_score,
            "duplicates": selection.duplicate_count,
            "over_max_clips": selection.over_limit,
        },
        "clips": [clip.to_dict() for clip in selection.selected],
        "rejected": [item.to_dict() for item in rejected],
        "dropped": [item.to_dict() for item in selection.dropped],
        "warnings": warnings,
    }
    analysis_path = run_path / "analysis.json"
    atomic_write_json(analysis_path, analysis_payload)
    report_path = run_path / "analysis" / "report.md"
    _write_report(
        report_path,
        run_dir=run_path,
        source=state.get("source") or {},
        settings=settings,
        coverage=coverage,
        selected=selection.selected,
        rejected=rejected,
        selection=selection,
        parsed_candidates=parsed_candidates,
        warnings=warnings,
        language=transcript.language,
    )

    state["settings"] = settings.to_dict()
    state["last_import_at"] = utc_now()
    _save_run_state(run_path, state)

    if not quiet:
        print(
            f"[import] {len(files)} response file(s), {parsed_candidates} parsed "
            f"candidate(s); resolved {len(resolved)}, rejected {len(rejected)}"
        )
        print(
            f"[select] {len(selection.selected)} selected, "
            f"{selection.duplicate_count} duplicate(s), {selection.below_score} below "
            f"minimum score, {selection.low_context} below the cold-viewer context "
            f"floor, {selection.over_limit} over the clip limit"
        )
        print(
            f"[coverage] {coverage['blocks_responded']}/{coverage['blocks_total']} blocks"
            + (
                f" (missing: {', '.join(missing_blocks)})"
                if missing_blocks
                else ""
            )
        )
        print(f"[write] {analysis_path}")
        print(f"[write] {report_path}")

    if parsed_candidates == 0 and not resolved:
        if parse_errors:
            raise PipelineError(
                "No provider response could be parsed. See analysis.json for reasons."
            )
        return ImportResult(
            analysis_path=analysis_path,
            report_path=report_path,
            selected=0,
            resolved=0,
            rejected=len(rejected),
            dropped=len(selection.dropped),
            parsed_candidates=0,
            coverage=coverage,
            response_files=len(files),
            warnings=warnings,
        )
    if parsed_candidates > 0 and not resolved:
        raise PipelineError(
            "Candidates were parsed, but none passed line-anchor and duration "
            "validation. See analysis.json and analysis/report.md for exact reasons."
        )
    return ImportResult(
        analysis_path=analysis_path,
        report_path=report_path,
        selected=len(selection.selected),
        resolved=len(resolved),
        rejected=len(rejected),
        dropped=len(selection.dropped),
        parsed_candidates=parsed_candidates,
        coverage=coverage,
        response_files=len(files),
        warnings=warnings,
    )


def analyze(
    source: str | Path,
    run_dir: str | Path,
    *,
    overrides: dict[str, Any] | None = None,
    transcript_path: str | Path | None = None,
    api: ApiSettings | None = None,
    force_transcribe: bool = False,
    quiet: bool = False,
) -> AnalyzeResult:
    source_path = Path(source)
    if not source_path.exists():
        raise PipelineError(f"Source video not found: {source_path}")
    run_path = Path(run_dir)

    info = probe_media(source_path)
    fingerprint = source_fingerprint(source_path, info)
    state, settings, _unused_warnings = _prepare_run(
        run_path, source_path, info, fingerprint, overrides or {}
    )
    print(
        f"[source] {source_path.name}: {human_duration(info.duration)}, "
        f"{info.width or '?'}x{info.height or '?'}, audio={'yes' if info.has_audio else 'no'}"
    )
    print(
        f"[run] {run_path} | language={settings.language or 'auto'} | "
        f"clips={settings.min_duration:g}-{settings.max_duration:g}s | "
        f"max={settings.max_clips} | min score={settings.min_score}"
    )

    transcript, transcript_source = _load_or_create_transcript(
        run_path,
        state,
        source_path,
        info,
        fingerprint,
        settings,
        transcript_path,
        force_transcribe,
    )
    if transcript_source == "reused" and (
        transcript.metadata.get("model") not in (None, settings.asr_model)
    ):
        print(
            f"[warn] existing transcript was made with model "
            f"{transcript.metadata.get('model')!r}, not {settings.asr_model!r}; "
            "reuse is safe. Pass --force-transcribe to redo it.",
            file=sys.stderr,
        )

    bundle_key = _prompt_bundle_key(transcript, settings)
    blocks_file = run_path / "analysis" / "blocks.json"
    can_reuse_bundle = (
        state.get("prompt_bundle_key") == bundle_key and blocks_file.exists()
    )
    blocks: list[Block] | None = None
    if can_reuse_bundle:
        try:
            blocks = load_blocks(blocks_file)
            if not _prompt_files_present(run_path, blocks):
                can_reuse_bundle = False
        except Exception:
            can_reuse_bundle = False
    state["settings"] = settings.to_dict()
    state["transcript"] = {
        "path": "transcript.json",
        "sha256": _transcript_digest(transcript),
        "language": transcript.language,
        "backend": transcript.backend,
        "units": len(transcript.units),
        "origin": transcript_source,
    }
    state["source"] = fingerprint
    state["media"] = info.to_dict()
    if api is not None:
        state["api"] = api.describe()
    # Save the recoverable stage before writing prompts. If the process dies
    # during prompt export, the next run finds a valid run.json and transcript
    # instead of refusing a non-empty directory without a manifest.
    state["prompt_bundle_key"] = bundle_key if can_reuse_bundle else None
    _save_run_state(run_path, state)

    if can_reuse_bundle and blocks is not None:
        print(f"[prompts] reused {len(blocks)} exported prompt block(s)")
    else:
        if blocks_file.exists():
            _archive_responses(
                run_path,
                "prompt/transcript settings changed (old responses cannot be trusted)",
            )
        blocks = write_prompt_bundle(run_path, transcript, settings)
        state["prompt_bundle_key"] = bundle_key
        _save_run_state(run_path, state)
        print(f"[prompts] wrote {len(blocks)} portable prompt block(s)")

    api_errors: list[dict[str, Any]] = []
    import_result: ImportResult | None = None
    api_complete = True
    if api is not None:
        responses_dir = run_path / "analysis" / "responses"
        for block in blocks:
            prompt_path = run_path / block.prompt_file
            if not prompt_path.exists():
                api_errors.append(
                    {"block_id": block.id, "error": f"prompt missing at {prompt_path}"}
                )
                api_complete = False
                continue
            response_path = responses_dir / f"block_{block.id[1:]}.json"
            if response_path.exists():
                try:
                    load_response_file(response_path)
                    print(f"[api] {block.id}: reuse existing response")
                    continue
                except Exception as exc:
                    print(
                        f"[warn] existing response for {block.id} is unusable ({exc}); "
                        "calling the provider again.",
                        file=sys.stderr,
                    )
            prompt = prompt_path.read_text(encoding="utf-8")
            print(
                f"[api] {block.id}: calling {api.model} "
                f"({block.start_line}-{block.end_line})"
            )
            try:
                from .provider import call_block_to_file

                call_block_to_file(prompt, api, response_path, block_id=block.id)
            except Exception as exc:
                message = str(exc)
                api_errors.append({"block_id": block.id, "error": message})
                api_complete = False
                print(f"[error] {block.id}: {message}", file=sys.stderr)
        atomic_write_json(
            run_path / "analysis" / "api_errors.json",
            {
                "schema_version": SCHEMA_VERSION,
                "created_at": utc_now(),
                "model": api.model,
                "base_url": api.base_url,
                "errors": api_errors,
            },
        )
        if api_errors:
            print(
                f"[warn] {len(api_errors)} block(s) failed. Re-run analyze --api to "
                "retry only the missing/failed blocks.",
                file=sys.stderr,
            )
        if any(responses_dir.glob("*.json")):
            import_result = import_analysis(run_path, quiet=quiet)
        else:
            raise PipelineError(
                "No provider responses were produced; see analysis/api_errors.json."
            )
        if import_result and not import_result.coverage.get("complete"):
            api_complete = False
    else:
        responses_dir = run_path / "analysis" / "responses"
        existing = list(responses_dir.glob("*.json")) + list(responses_dir.glob("*.txt"))
        print()
        print("Next steps (no API key required):")
        print(f"  1. Paste each prompt under {run_path / 'analysis' / 'prompts'}")
        print("     into any AI provider.")
        print(
            f"  2. Save each JSON answer under {responses_dir} "
            "(one file per block)."
        )
        print(f"  3. python -m momentcut import \"{run_path}\"")
        print(f"  4. python -m momentcut render \"{run_path}\" --aspect vertical")
        if existing:
            print(
                f"  ({len(existing)} response file(s) already present; step 3 will "
                "import them.)"
            )
        print()

    return AnalyzeResult(
        run_dir=run_path,
        transcript=transcript,
        blocks=blocks,
        api_errors=api_errors,
        import_result=import_result,
        api_complete=api_complete,
    )
