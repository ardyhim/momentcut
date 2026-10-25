"""One small OpenAI-compatible chat-completions client.

This is intentionally not a provider framework. It covers the common
`POST {base}/chat/completions` shape used by OpenAI, DeepSeek, OpenRouter,
Groq, Together, local Ollama/LM Studio/vLLM, and others. The manual prompt
export path remains the universal fallback.
"""

from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Any

from .config import ApiSettings
from .util import MomentCutError, atomic_write_json, utc_now


class ProviderError(MomentCutError):
    pass


def completions_url(base_url: str) -> str:
    base = (base_url or "").strip().rstrip("/")
    if not base:
        raise ProviderError("API base URL is empty.")
    if not base.startswith(("http://", "https://")):
        raise ProviderError("API base URL must start with http:// or https://.")
    if base.endswith("/chat/completions"):
        return base
    return f"{base}/chat/completions"


def _extract_content(payload: dict[str, Any]) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ProviderError("API response has no choices.")
    first = choices[0]
    if not isinstance(first, dict):
        raise ProviderError("API response choice is not an object.")
    message = first.get("message")
    if not isinstance(message, dict):
        raise ProviderError("API response has no message object.")
    content = message.get("content")
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                value = part.get("text") or part.get("content")
                if isinstance(value, str):
                    parts.append(value)
        text = "".join(parts)
    else:
        text = ""
    if not text.strip():
        finish = first.get("finish_reason")
        suffix = f" (finish_reason={finish})" if finish else ""
        raise ProviderError(
            "API returned an empty message content" + suffix + ". "
            "The model may have refused, run out of tokens, or used a "
            "non-chat response format."
        )
    return text


def _retry_after_seconds(headers: Any) -> float | None:
    if headers is None:
        return None
    raw = None
    try:
        raw = headers.get("Retry-After")
    except Exception:
        return None
    if raw is None:
        return None
    try:
        return max(0.0, min(60.0, float(raw)))
    except (TypeError, ValueError):
        return None


def _make_variants(settings: ApiSettings) -> list[dict[str, Any]]:
    base: dict[str, Any] = {
        "model": settings.model,
        "messages": [],
        "temperature": settings.temperature,
        "max_tokens": settings.max_tokens,
        "response_format": {"type": "json_object"},
        "stream": False,
    }
    variants = [dict(base)]
    without_format = dict(base)
    without_format.pop("response_format", None)
    variants.append(without_format)
    minimal = {
        "model": settings.model,
        "messages": [],
        "max_completion_tokens": settings.max_tokens,
        "stream": False,
    }
    variants.append(minimal)
    return variants


def _error_text(response: Any) -> str:
    try:
        text = response.text or ""
    except Exception:
        text = ""
    return text.strip()[:1200]


def _format_rejection(status: int, body: str) -> bool:
    if status not in (400, 404, 422):
        return False
    lowered = body.lower()
    return any(
        token in lowered
        for token in ("response_format", "json_object", "max_tokens", "temperature",
                      "max_completion_tokens", "unsupported parameter")
    )


def call_chat_completion(
    prompt: str,
    settings: ApiSettings,
    *,
    block_id: str = "",
    sleep_fn=time.sleep,
) -> dict[str, Any]:
    try:
        import requests  # type: ignore
    except ImportError as exc:
        raise ProviderError(
            "The requests package is not installed. Run "
            "`pip install -r requirements.txt`."
        ) from exc

    url = completions_url(settings.base_url)
    headers = {"Content-Type": "application/json"}
    if settings.api_key:
        headers["Authorization"] = f"Bearer {settings.api_key}"
    variants = _make_variants(settings)
    attempts = 0
    variant_index = 0
    started = time.time()
    last_error = ""

    while attempts < max(1, settings.max_attempts):
        attempts += 1
        body = dict(variants[variant_index])
        body["messages"] = [{"role": "user", "content": prompt}]
        try:
            response = requests.post(
                url,
                headers=headers,
                json=body,
                timeout=settings.timeout,
            )
        except requests.exceptions.RequestException as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempts >= settings.max_attempts:
                break
            wait = min(30.0, settings.retry_base_seconds * (2 ** (attempts - 1)))
            print(f"[warn] provider transport error; retrying in {wait:.1f}s: {last_error}")
            sleep_fn(wait + random.uniform(0, 0.25))
            continue

        if response.status_code >= 400:
            body_text = _error_text(response)
            last_error = f"HTTP {response.status_code}: {body_text}"
            # Compatibility fallbacks for providers that reject optional
            # parameters. Only move to a simpler request when the error names
            # one of those parameters; a real authentication/content error is
            # not retried under a different shape.
            if _format_rejection(response.status_code, body_text):
                if variant_index + 1 < len(variants):
                    variant_index += 1
                    attempts -= 1  # a format fallback is not an API retry
                    continue
            if response.status_code in (429, 500, 502, 503, 504):
                if attempts >= settings.max_attempts:
                    break
                wait = _retry_after_seconds(getattr(response, "headers", None))
                if wait is None:
                    wait = min(30.0, settings.retry_base_seconds * (2 ** (attempts - 1)))
                print(f"[warn] provider HTTP {response.status_code}; retrying in {wait:.1f}s")
                sleep_fn(wait + random.uniform(0, 0.25))
                continue
            break

        try:
            payload = response.json()
        except ValueError as exc:
            raise ProviderError(
                f"API returned non-JSON HTTP body: {_error_text(response)[:300]}"
            ) from exc
        if not isinstance(payload, dict):
            raise ProviderError("API returned a non-object JSON body.")
        content = _extract_content(payload)
        usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
        return {
            "schema_version": "1.0",
            "block_id": block_id,
            "provider": "openai-compatible",
            "model": settings.model,
            "base_url": settings.base_url,
            "usage": usage,
            "attempts": attempts,
            "elapsed_seconds": round(time.time() - started, 2),
            "content": content,
            "received_at": utc_now(),
        }

    raise ProviderError(
        f"Provider call failed after {attempts} attempt(s): {last_error or 'unknown error'}"
    )


def call_block_to_file(
    prompt: str,
    settings: ApiSettings,
    destination: str | Path,
    *,
    block_id: str,
) -> dict[str, Any]:
    result = call_chat_completion(prompt, settings, block_id=block_id)
    atomic_write_json(destination, result)
    return result
