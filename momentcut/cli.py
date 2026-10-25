"""Command-line interface for MomentCut."""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from . import __version__
from .config import CONTENT_TYPES, ApiSettings
from .pipeline import PipelineError, analyze, import_analysis
from .render import render_run
from .util import MomentCutError, slugify


def _env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    return value.strip() if value and value.strip() else default


def _default_run(source: str) -> Path:
    stem = Path(source).stem or "video"
    return Path("runs") / slugify(stem, max_length=60)


def _settings_overrides(args: argparse.Namespace) -> dict[str, Any]:
    keys = (
        "min_duration",
        "max_duration",
        "max_clips",
        "min_score",
        "content_type",
        "focus",
        "block_seconds",
        "asr_model",
        "asr_chunk_seconds",
        "language",
    )
    return {
        key: getattr(args, key)
        for key in keys
        if getattr(args, key, None) is not None
    }


def _build_api_settings(args: argparse.Namespace) -> ApiSettings | None:
    if not args.api:
        return None
    base_url = args.api_base_url or _env("MOMENTCUT_API_BASE_URL")
    model = args.api_model or _env("MOMENTCUT_API_MODEL")
    if not base_url:
        raise PipelineError(
            "API mode requires --api-base-url or MOMENTCUT_API_BASE_URL. "
            "The URL is the provider's OpenAI-compatible base (for example "
            "https://api.openai.com/v1 or http://localhost:11434/v1)."
        )
    if not model:
        raise PipelineError(
            "API mode requires --api-model or MOMENTCUT_API_MODEL."
        )
    key = _env("MOMENTCUT_API_KEY")
    max_tokens = args.api_max_tokens
    if max_tokens is None:
        try:
            max_tokens = int(_env("MOMENTCUT_API_MAX_TOKENS", "4096") or "4096")
        except ValueError:
            max_tokens = 4096
    timeout = args.api_timeout
    if timeout is None:
        try:
            timeout = float(_env("MOMENTCUT_API_TIMEOUT", "180") or "180")
        except ValueError:
            timeout = 180.0
    return ApiSettings(
        base_url=base_url,
        model=model,
        api_key=key,
        max_tokens=max_tokens,
        timeout=timeout,
    )


def _cmd_analyze(args: argparse.Namespace) -> int:
    run_dir = Path(args.run) if args.run else _default_run(args.source)
    api = _build_api_settings(args)
    result = analyze(
        args.source,
        run_dir,
        overrides=_settings_overrides(args),
        transcript_path=args.transcript,
        api=api,
        force_transcribe=args.force_transcribe,
    )
    if api is not None:
        if result.import_result:
            print(
                f"[done] {result.import_result.selected} clip(s) selected. "
                f"Render with: python -m momentcut render \"{result.run_dir}\" --aspect vertical"
            )
        if not result.api_complete:
            print(
                "[warn] API analysis is incomplete. Re-run the same command to retry "
                "missing blocks; completed responses are reused.",
                file=sys.stderr,
            )
            return 1
    else:
        print(
            f"[done] prompts exported under {result.run_dir / 'analysis' / 'prompts'}"
        )
    return 0


def _cmd_import(args: argparse.Namespace) -> int:
    result = import_analysis(
        args.run,
        args.responses,
        source_override=args.source,
        min_score=args.min_score,
        max_clips=args.max_clips,
    )
    if not result.coverage.get("complete"):
        print(
            "[warn] Imported a partial analysis. Add the missing block answers and "
            "re-run import; no AI analysis is repeated.",
            file=sys.stderr,
        )
    if result.selected == 0:
        print(
            "[warn] The analysis is valid but selected no clips. See "
            f"{result.report_path}."
        )
    return 0


def _cmd_render(args: argparse.Namespace) -> int:
    result = render_run(
        args.run,
        source_override=args.source,
        aspect=args.aspect,
        force=args.force,
    )
    return 1 if result.failed else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="momentcut",
        description=(
            "Find self-contained moments in long-form video and cut them with "
            "validated timestamps."
        ),
    )
    parser.add_argument("--version", action="version", version=f"MomentCut {__version__}")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="show a traceback for unexpected failures",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze_parser = subparsers.add_parser(
        "analyze",
        help="transcribe a video and export analysis prompts (or call an API)",
    )
    analyze_parser.add_argument("source", help="local video file")
    analyze_parser.add_argument(
        "--run",
        help="run directory (default: runs/<video name>); reused for resuming",
    )
    analyze_parser.add_argument(
        "--transcript",
        help="precomputed .json/.srt/.vtt transcript; skips faster-whisper",
    )
    analyze_parser.add_argument("--language", help="ISO language hint for ASR/copy")
    analyze_parser.add_argument(
        "--asr-model",
        help="faster-whisper model (default: small; use base for CPU-only Colab)",
    )
    analyze_parser.add_argument(
        "--asr-chunk-seconds",
        type=float,
        help="resumable ASR chunk length (default: 600; 0 disables chunking)",
    )
    analyze_parser.add_argument(
        "--force-transcribe",
        action="store_true",
        help="ignore an existing transcript and run ASR again",
    )
    analyze_parser.add_argument(
        "--min-duration", type=float, help="minimum clip duration (default: 15)"
    )
    analyze_parser.add_argument(
        "--max-duration", type=float, help="maximum clip duration (default: 90)"
    )
    analyze_parser.add_argument(
        "--max-clips", type=int, help="maximum selected clips (default: 8)"
    )
    analyze_parser.add_argument(
        "--min-score", type=int, help="minimum model overall score (default: 60)"
    )
    analyze_parser.add_argument(
        "--content-type",
        choices=CONTENT_TYPES,
        help="content-type hint (default: auto)",
    )
    analyze_parser.add_argument(
        "--focus", help="optional editorial focus, e.g. 'funny moments'"
    )
    analyze_parser.add_argument(
        "--block-seconds",
        type=float,
        help="target transcript block length (default: 600)",
    )
    analyze_parser.add_argument(
        "--api",
        action="store_true",
        help="call an OpenAI-compatible endpoint instead of waiting for manual import",
    )
    analyze_parser.add_argument(
        "--api-base-url", help="OpenAI-compatible base URL (or MOMENTCUT_API_BASE_URL)"
    )
    analyze_parser.add_argument(
        "--api-model", help="model name (or MOMENTCUT_API_MODEL)"
    )
    analyze_parser.add_argument(
        "--api-max-tokens",
        type=int,
        help="completion token budget (or MOMENTCUT_API_MAX_TOKENS; default: 4096)",
    )
    analyze_parser.add_argument(
        "--api-timeout",
        type=float,
        help="HTTP timeout in seconds (or MOMENTCUT_API_TIMEOUT; default: 180)",
    )
    analyze_parser.set_defaults(func=_cmd_analyze)

    import_parser = subparsers.add_parser(
        "import",
        help="import provider JSON answers, validate anchors, and select clips",
    )
    import_parser.add_argument("run", help="run directory created by analyze")
    import_parser.add_argument(
        "responses",
        nargs="*",
        help="answer files/directories; default: analysis/responses/*",
    )
    import_parser.add_argument(
        "--source", help="current source video path if it moved"
    )
    import_parser.add_argument(
        "--min-score",
        type=int,
        help="override the stored minimum score without repeating AI work",
    )
    import_parser.add_argument(
        "--max-clips",
        type=int,
        help="override the stored clip limit without repeating AI work",
    )
    import_parser.set_defaults(func=_cmd_import)

    render_parser = subparsers.add_parser(
        "render", help="cut and encode selected clips from an imported analysis"
    )
    render_parser.add_argument("run", help="run directory with analysis.json")
    render_parser.add_argument(
        "--source", help="current source video path if it moved"
    )
    render_parser.add_argument(
        "--aspect",
        choices=("original", "vertical"),
        default="original",
        help="original frame, or 9:16 blurred pillarbox (default: original)",
    )
    render_parser.add_argument(
        "--force", action="store_true", help="re-render clips even if they exist"
    )
    render_parser.set_defaults(func=_cmd_render)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        print("\n[interrupted]", file=sys.stderr)
        return 130
    except MomentCutError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        if getattr(args, "debug", False):
            traceback.print_exc()
        return 1
    except Exception as exc:  # unexpected bug: still give the user a clean exit
        print(f"[error] {type(exc).__name__}: {exc}", file=sys.stderr)
        if getattr(args, "debug", False):
            traceback.print_exc()
        return 1
