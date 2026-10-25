"""Run settings and their persisted, validated representation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .util import MomentCutError, clean_text

CONTENT_TYPES = (
    "auto",
    "podcast",
    "interview",
    "tutorial",
    "comedy",
    "vlog",
    "debate",
    "other",
)


class SettingsError(MomentCutError):
    pass


@dataclass
class RunSettings:
    min_duration: float = 15.0
    max_duration: float = 90.0
    max_clips: int = 8
    min_score: int = 60
    content_type: str = "auto"
    focus: str = ""
    block_seconds: float = 600.0
    asr_model: str = "small"
    asr_chunk_seconds: float = 600.0
    language: str | None = None

    def validate(self) -> None:
        if not (5.0 <= self.min_duration <= 175.0):
            raise SettingsError("min-duration must be between 5 and 175 seconds.")
        if not (10.0 <= self.max_duration <= 180.0):
            raise SettingsError("max-duration must be between 10 and 180 seconds.")
        if self.max_duration <= self.min_duration:
            raise SettingsError("max-duration must be greater than min-duration.")
        if not (1 <= int(self.max_clips) <= 50):
            raise SettingsError("max-clips must be between 1 and 50.")
        if not (0 <= int(self.min_score) <= 100):
            raise SettingsError("min-score must be between 0 and 100.")
        if self.content_type not in CONTENT_TYPES:
            raise SettingsError(
                f"content-type must be one of: {', '.join(CONTENT_TYPES)}"
            )
        if not (120.0 <= self.block_seconds <= 1800.0):
            raise SettingsError("block-seconds must be between 120 and 1800.")
        if self.asr_chunk_seconds != 0 and self.asr_chunk_seconds < 60:
            raise SettingsError("asr-chunk-seconds must be 0 or at least 60.")
        if not self.asr_model.strip():
            raise SettingsError("asr-model cannot be empty.")
        self.focus = clean_text(self.focus, limit=500)

    def to_dict(self) -> dict[str, Any]:
        return {
            "min_duration": round(float(self.min_duration), 3),
            "max_duration": round(float(self.max_duration), 3),
            "max_clips": int(self.max_clips),
            "min_score": int(self.min_score),
            "content_type": self.content_type,
            "focus": self.focus,
            "block_seconds": round(float(self.block_seconds), 3),
            "asr_model": self.asr_model,
            "asr_chunk_seconds": round(float(self.asr_chunk_seconds), 3),
            "language": self.language,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> RunSettings:
        if not payload:
            return cls()
        return cls(
            min_duration=float(payload.get("min_duration", 15.0)),
            max_duration=float(payload.get("max_duration", 90.0)),
            max_clips=int(payload.get("max_clips", 8)),
            min_score=int(payload.get("min_score", 60)),
            content_type=str(payload.get("content_type", "auto")),
            focus=str(payload.get("focus", "")),
            block_seconds=float(payload.get("block_seconds", 600.0)),
            asr_model=str(payload.get("asr_model", "small")),
            asr_chunk_seconds=float(payload.get("asr_chunk_seconds", 600.0)),
            language=payload.get("language"),
        )


@dataclass
class ApiSettings:
    base_url: str
    model: str
    api_key: str | None = None
    max_tokens: int = 4096
    temperature: float = 0.2
    timeout: float = 180.0
    max_attempts: int = 3
    retry_base_seconds: float = 2.0

    def describe(self) -> dict[str, Any]:
        """Safe-to-persist provider description; never includes the key."""
        return {
            "base_url": self.base_url,
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
