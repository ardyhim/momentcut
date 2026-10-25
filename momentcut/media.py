"""FFmpeg/ffprobe boundary: probe sources, extract audio, render clips."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .util import MomentCutError, coerce_float, format_clock, quick_file_hash, utc_now


class MediaError(MomentCutError):
    pass


def require_tool(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise MediaError(
            f"{name} was not found on PATH. Install FFmpeg, then restart the "
            f"runtime. On Colab, FFmpeg is normally preinstalled."
        )
    return path


def tool_versions() -> dict[str, str]:
    result: dict[str, str] = {}
    for tool in ("ffmpeg", "ffprobe"):
        path = require_tool(tool)
        completed = subprocess.run(
            [path, "-version"],
            capture_output=True,
            text=True,
            errors="replace",
            check=False,
        )
        first = (completed.stdout or completed.stderr or "").splitlines()
        result[tool] = first[0].strip() if first else path
    return result


def _run(
    command: list[str],
    *,
    timeout: float | None = None,
    description: str = "ffmpeg",
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError as exc:
        raise MediaError(f"{description}: required executable was not found.") from exc
    except subprocess.TimeoutExpired as exc:
        raise MediaError(f"{description} timed out after {timeout:.0f}s.") from exc


@dataclass(frozen=True)
class MediaInfo:
    path: str
    duration: float
    size: int
    format_name: str
    video_codec: str
    width: int | None
    height: int | None
    fps: float | None
    audio_codec: str | None
    audio_sample_rate: int | None
    has_audio: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "duration": round(self.duration, 3),
            "size": self.size,
            "format_name": self.format_name,
            "video_codec": self.video_codec,
            "width": self.width,
            "height": self.height,
            "fps": round(self.fps, 3) if self.fps else None,
            "audio_codec": self.audio_codec,
            "audio_sample_rate": self.audio_sample_rate,
            "has_audio": self.has_audio,
        }


def _fraction(value: Any) -> float | None:
    if not value or not isinstance(value, str):
        return None
    if "/" in value:
        numerator, _, denominator = value.partition("/")
        try:
            denominator_value = float(denominator)
            if denominator_value == 0:
                return None
            return float(numerator) / denominator_value
        except ValueError:
            return None
    try:
        return float(value)
    except ValueError:
        return None


def probe_media(path: str | Path) -> MediaInfo:
    source = Path(path)
    if not source.exists() or not source.is_file():
        raise MediaError(f"Source video not found: {source}")
    ffprobe = require_tool("ffprobe")
    command = [
        ffprobe,
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(source),
    ]
    completed = _run(command, timeout=120, description="ffprobe")
    if completed.returncode != 0:
        detail = (completed.stderr or "").strip()[-800:]
        raise MediaError(f"ffprobe could not read {source.name}: {detail or 'unknown error'}")
    try:
        payload = json.loads(completed.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise MediaError("ffprobe returned invalid JSON.") from exc

    streams = payload.get("streams") or []
    video_streams = [s for s in streams if s.get("codec_type") == "video"]
    audio_streams = [s for s in streams if s.get("codec_type") == "audio"]
    if not video_streams:
        raise MediaError(f"{source.name} has no video stream.")
    video = video_streams[0]

    fmt = payload.get("format") or {}
    duration = coerce_float(fmt.get("duration"))
    if duration is None:
        duration = coerce_float(video.get("duration"))
    if duration is None:
        for stream in streams:
            candidate = coerce_float(stream.get("duration"))
            if candidate is not None and (duration is None or candidate > duration):
                duration = candidate
    if duration is None or duration <= 0:
        raise MediaError(f"Could not determine a positive duration for {source.name}.")

    size = coerce_float(fmt.get("size"))
    if size is None:
        size = source.stat().st_size

    audio = audio_streams[0] if audio_streams else None
    return MediaInfo(
        path=str(source.resolve()),
        duration=round(float(duration), 3),
        size=int(size),
        format_name=str(fmt.get("format_name") or ""),
        video_codec=str(video.get("codec_name") or "unknown"),
        width=int(video["width"]) if video.get("width") else None,
        height=int(video["height"]) if video.get("height") else None,
        fps=_fraction(video.get("avg_frame_rate")) or _fraction(video.get("r_frame_rate")),
        audio_codec=str(audio.get("codec_name")) if audio else None,
        audio_sample_rate=int(audio["sample_rate"]) if audio and audio.get("sample_rate") else None,
        has_audio=audio is not None,
    )


def source_fingerprint(path: str | Path, info: MediaInfo | None = None) -> dict[str, Any]:
    source = Path(path)
    if not source.exists():
        raise MediaError(f"Source video not found: {source}")
    if info is None:
        info = probe_media(source)
    stat = source.stat()
    return {
        "path": str(source.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "duration": round(info.duration, 3),
        "quick_hash": quick_file_hash(source),
    }


def fingerprint_compatible(recorded: dict[str, Any], current: dict[str, Any]) -> bool:
    """True when the current file is the same media the analysis used."""
    if not recorded or not current:
        return False
    if int(recorded.get("size", -1)) != int(current.get("size", -2)):
        return False
    if abs(float(recorded.get("duration", -1)) - float(current.get("duration", -2))) > 0.1:
        return False
    if recorded.get("quick_hash") and current.get("quick_hash"):
        return recorded["quick_hash"] == current["quick_hash"]
    return True


def extract_audio_chunk(
    source: str | Path,
    start: float,
    duration: float,
    destination: str | Path,
) -> Path:
    """Extract a mono 16 kHz PCM chunk for ASR.

    Re-encoding is deliberate: a copied audio stream can only split on packet
    boundaries, which is not good enough for resumable chunking.
    """
    ffmpeg = require_tool("ffmpeg")
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{max(0.0, start):.3f}",
        "-i",
        str(source),
        "-t",
        f"{max(0.05, duration):.3f}",
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(target),
    ]
    completed = _run(command, timeout=600, description="audio extraction")
    if completed.returncode != 0 or not target.exists() or target.stat().st_size < 1000:
        detail = (completed.stderr or "").strip()[-800:]
        raise MediaError(
            f"Could not extract audio from {Path(source).name}: {detail or 'ffmpeg wrote no audio'}"
        )
    return target


def ffmpeg_cut_command(
    source: str | Path,
    destination: str | Path,
    start: float,
    duration: float,
    *,
    aspect: str = "original",
    has_audio: bool = True,
    crf: int = 21,
    preset: str = "veryfast",
) -> list[str]:
    ffmpeg = require_tool("ffmpeg")
    command = [
        ffmpeg,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{start:.3f}",
        "-i",
        str(source),
        "-t",
        f"{duration:.3f}",
    ]
    if aspect == "original":
        command += ["-map", "0:v:0"]
    if has_audio:
        command += ["-map", "0:a:0"]
    if aspect == "vertical":
        # Preserve the entire frame over a blurred fill. This is intentionally
        # not face tracking: it is predictable, provider-independent, and does
        # not silently crop people out of the frame.
        command += [
            "-filter_complex",
            (
                "[0:v]split=2[bg][fg];"
                "[bg]scale=1080:1920:force_original_aspect_ratio=increase,"
                "crop=1080:1920,boxblur=20:2[bgblur];"
                "[fg]scale=1080:1920:force_original_aspect_ratio=decrease[fgfit];"
                "[bgblur][fgfit]overlay=(W-w)/2:(H-h)/2,setsar=1,format=yuv420p[v]"
            ),
            "-map",
            "[v]",
        ]
    elif aspect == "original":
        command += ["-pix_fmt", "yuv420p"]
    else:
        raise MediaError(f"Unsupported aspect '{aspect}'. Use 'original' or 'vertical'.")

    command += [
        "-c:v",
        "libx264",
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-c:a",
        "aac",
        "-b:a",
        "160k",
        "-map_metadata",
        "-1",
        "-movflags",
        "+faststart",
        str(destination),
    ]
    return command


def render_cut(
    source: str | Path,
    destination: str | Path,
    start: float,
    duration: float,
    *,
    aspect: str = "original",
    has_audio: bool = True,
    crf: int = 21,
    preset: str = "veryfast",
) -> Path:
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        target.unlink()
    command = ffmpeg_cut_command(
        source,
        target,
        start,
        duration,
        aspect=aspect,
        has_audio=has_audio,
        crf=crf,
        preset=preset,
    )
    completed = _run(command, description="clip render")
    if completed.returncode != 0:
        detail = (completed.stderr or "").strip()[-1200:]
        raise MediaError(
            f"FFmpeg failed for {format_clock(start)} ({duration:.1f}s): "
            f"{detail or 'unknown error'}"
        )
    if not target.exists() or target.stat().st_size < 1000:
        raise MediaError(
            f"FFmpeg reported success but wrote no usable file for "
            f"{format_clock(start)} ({duration:.1f}s)."
        )
    return target


def verify_rendered_clip(
    path: str | Path,
    expected_duration: float,
    *,
    expect_audio: bool = True,
    vertical: bool = False,
) -> dict[str, Any]:
    info = probe_media(path)
    tolerance = max(0.6, min(1.5, expected_duration * 0.05))
    delta = abs(info.duration - expected_duration)
    if delta > tolerance:
        raise MediaError(
            f"Rendered clip duration {info.duration:.3f}s differs from the planned "
            f"{expected_duration:.3f}s by {delta:.3f}s (tolerance {tolerance:.2f}s)."
        )
    if expect_audio and not info.has_audio:
        raise MediaError("Rendered clip has no audio stream although the source has audio.")
    if vertical and (info.width, info.height) != (1080, 1920):
        raise MediaError(
            f"Vertical render is {info.width}x{info.height}, expected 1080x1920."
        )
    return info.to_dict()


def now_iso() -> str:
    return utc_now()
