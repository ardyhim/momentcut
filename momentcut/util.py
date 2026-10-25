"""Small shared helpers. Standard library only."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
import unicodedata
from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class MomentCutError(RuntimeError):
    """An expected, user-actionable failure."""


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def ensure_dir(path: str | Path) -> Path:
    resolved = Path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def atomic_write_json(path: str | Path, payload: Any) -> None:
    """Write JSON so a crash cannot leave a half-written file."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, target)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def atomic_write_text(path: str | Path, text: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, target)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def quick_file_hash(path: str | Path, sample_bytes: int = 1024 * 1024) -> str:
    """Hash the size plus the first and last MiB of a file.

    This is a change detector, not a cryptographic content identity. It keeps
    re-probing a multi-gigabyte source cheap while still catching a replaced
    file whose size and timestamps might match by accident.
    """
    file_path = Path(path)
    size = file_path.stat().st_size
    digest = hashlib.sha256()
    digest.update(str(size).encode("ascii"))
    with file_path.open("rb") as handle:
        digest.update(handle.read(sample_bytes))
        if size > sample_bytes:
            handle.seek(max(0, size - sample_bytes))
            digest.update(handle.read(sample_bytes))
    return digest.hexdigest()


def clean_text(value: Any, *, limit: int | None = None) -> str:
    """Collapse whitespace and remove control characters, preserving Unicode."""
    if value is None:
        return ""
    text = unicodedata.normalize("NFC", str(value))
    text = "".join(
        char if (char in "\n\t" or ord(char) >= 32) else " " for char in text
    )
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s*\n\s*", " ", text)
    text = re.sub(r"\s{2,}", " ", text).strip()
    if limit is not None and len(text) > limit:
        text = text[:limit].rstrip()
    return text


def normalize_for_match(text: str) -> str:
    value = unicodedata.normalize("NFKC", text or "").casefold()
    value = re.sub(r"[^\w\s]", " ", value, flags=re.UNICODE)
    value = value.replace("_", " ")
    return re.sub(r"\s+", " ", value).strip()


def tokens_for_match(text: str) -> list[str]:
    normalized = normalize_for_match(text)
    return normalized.split() if normalized else []


def truncate_chars(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    if limit <= 1:
        return text[:limit]
    cut = text[: limit - 1].rstrip()
    return cut + "\u2026"


def truncate_words(text: str, max_words: int, max_chars: int | None = None) -> str:
    words = text.split()
    result = " ".join(words[: max(1, max_words)]).strip()
    if max_chars is not None:
        result = truncate_chars(result, max_chars)
    return result


def unique_preserve(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result


def slugify(text: str, max_length: int = 48) -> str:
    value = unicodedata.normalize("NFKD", text or "")
    value = value.encode("ascii", "ignore").decode("ascii").lower()
    value = re.sub(r"[^a-z0-9]+", "-", value).strip("-")
    value = value[:max_length].strip("-")
    return value or "clip"


def format_clock(seconds: float, *, milliseconds: bool = True) -> str:
    seconds = max(0.0, float(seconds))
    whole = int(seconds)
    hours, rem = divmod(whole, 3600)
    minutes, secs = divmod(rem, 60)
    if milliseconds:
        millis = int(round((seconds - whole) * 1000))
        if millis == 1000:
            millis = 0
            secs += 1
            if secs == 60:
                secs = 0
                minutes += 1
            if minutes == 60:
                minutes = 0
                hours += 1
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def human_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    rest = int(round(seconds % 60))
    if minutes < 60:
        return f"{minutes}m {rest:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m"


def coerce_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def coerce_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value) or not value.is_integer():
            return None
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if re.fullmatch(r"-?\d+", text):
            return int(text)
        try:
            number = float(text)
        except ValueError:
            return None
        if math.isfinite(number) and number.is_integer():
            return int(number)
    return None


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def snippet(text: str, limit: int = 180) -> str:
    text = clean_text(text)
    return truncate_chars(text, limit)


def describe_seconds(seconds: Sequence[float]) -> str:
    return "-".join(format_clock(value, milliseconds=False) for value in seconds)
