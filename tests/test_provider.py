from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from momentcut.config import ApiSettings
from momentcut.contract import load_response_file
from momentcut.provider import (
    ProviderError,
    call_block_to_file,
    call_chat_completion,
    completions_url,
)


class ServerState:
    def __init__(self):
        self.requests = []
        self.actions = []
        self.server = None


class Handler(BaseHTTPRequestHandler):
    state: ServerState

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        try:
            body = json.loads(raw)
        except Exception:
            body = {}
        self.state.requests.append(
            {
                "path": self.path,
                "headers": {key: value for key, value in self.headers.items()},
                "body": body,
            }
        )
        action = self.state.actions.pop(0) if self.state.actions else {
            "status": 500,
            "body": {"error": "no action"},
        }
        status = int(action.get("status", 200))
        payload = action.get("body", {})
        if isinstance(payload, str):
            data = payload.encode("utf-8")
            content_type = "text/plain"
        else:
            data = json.dumps(payload).encode("utf-8")
            content_type = "application/json"
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        for header, value in action.get("headers", {}).items():
            self.send_header(header, value)
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format, *args):
        pass


@pytest.fixture
def provider_server():
    state = ServerState()

    handler = type("BoundHandler", (Handler,), {"state": state})
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    state.server = server
    state.base_url = f"http://127.0.0.1:{server.server_port}/v1"
    try:
        yield state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def content_response(text: str) -> dict:
    return {
        "choices": [
            {"message": {"content": text}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20},
    }


def settings(base_url: str, **kwargs) -> ApiSettings:
    values = {
        "base_url": base_url,
        "model": "test-model",
        "api_key": None,
        "max_tokens": 1024,
        "timeout": 5,
        "max_attempts": 3,
        "retry_base_seconds": 0.001,
    }
    values.update(kwargs)
    return ApiSettings(**values)


class TestUrl:
    def test_join(self):
        assert completions_url("https://api.example.com/v1") == (
            "https://api.example.com/v1/chat/completions"
        )
        assert completions_url("http://localhost:11434/v1/chat/completions") == (
            "http://localhost:11434/v1/chat/completions"
        )

    def test_invalid(self):
        with pytest.raises(ProviderError):
            completions_url("api.example.com")


class TestProvider:
    def test_success(self, provider_server):
        provider_server.actions.append(
            {"status": 200, "body": content_response('{"clips": []}')}
        )
        result = call_chat_completion(
            "prompt text", settings(provider_server.base_url), block_id="B001"
        )
        assert result["content"] == '{"clips": []}'
        assert result["block_id"] == "B001"
        assert result["usage"]["prompt_tokens"] == 10
        request = provider_server.requests[0]
        assert request["path"] == "/v1/chat/completions"
        assert request["body"]["messages"][0]["content"] == "prompt text"
        assert request["body"]["response_format"] == {"type": "json_object"}

    def test_auth_header(self, provider_server):
        provider_server.actions.append(
            {"status": 200, "body": content_response('{"clips": []}')}
        )
        call_chat_completion(
            "prompt", settings(provider_server.base_url, api_key="secret-key")
        )
        headers = provider_server.requests[0]["headers"]
        assert headers["Authorization"] == "Bearer secret-key"

    def test_response_format_fallback(self, provider_server):
        provider_server.actions.extend(
            [
                {
                    "status": 400,
                    "body": {"error": "response_format is not supported"},
                },
                {"status": 200, "body": content_response('{"clips": []}')},
            ]
        )
        result = call_chat_completion(
            "prompt", settings(provider_server.base_url), block_id="B001"
        )
        assert result["content"] == '{"clips": []}'
        assert len(provider_server.requests) == 2
        assert "response_format" not in provider_server.requests[1]["body"]

    def test_429_retry(self, provider_server):
        provider_server.actions.extend(
            [
                {
                    "status": 429,
                    "body": {"error": "rate limited"},
                    "headers": {"Retry-After": "0"},
                },
                {"status": 200, "body": content_response('{"clips": []}')},
            ]
        )
        result = call_chat_completion(
            "prompt",
            settings(provider_server.base_url, max_attempts=2),
            block_id="B001",
            sleep_fn=lambda _: None,
        )
        assert result["attempts"] == 2
        assert len(provider_server.requests) == 2

    def test_5xx_retry(self, provider_server):
        provider_server.actions.extend(
            [
                {"status": 503, "body": {"error": "busy"}},
                {"status": 200, "body": content_response('{"clips": []}')},
            ]
        )
        result = call_chat_completion(
            "prompt",
            settings(provider_server.base_url, max_attempts=2),
            block_id="B001",
            sleep_fn=lambda _: None,
        )
        assert result["attempts"] == 2

    def test_empty_content_raises(self, provider_server):
        provider_server.actions.append(
            {"status": 200, "body": {"choices": [{"message": {"content": ""}}]}}
        )
        with pytest.raises(ProviderError):
            call_chat_completion("prompt", settings(provider_server.base_url))

    def test_auth_error_not_retried(self, provider_server):
        provider_server.actions.append(
            {"status": 401, "body": {"error": "invalid api key"}}
        )
        with pytest.raises(ProviderError):
            call_chat_completion("prompt", settings(provider_server.base_url))
        assert len(provider_server.requests) == 1

    def test_file_wrapper_can_be_imported(self, provider_server, tmp_path):
        provider_server.actions.append(
            {
                "status": 200,
                "body": content_response(
                    json.dumps(
                        {
                            "schema_version": "1.0",
                            "block_id": "B001",
                            "clips": [],
                        }
                    )
                ),
            }
        )
        destination = tmp_path / "block_001.json"
        call_block_to_file(
            "prompt",
            settings(provider_server.base_url),
            destination,
            block_id="B001",
        )
        parsed = load_response_file(destination)
        assert parsed.block_id == "B001"
        assert parsed.candidates == []

    def test_no_base_url(self):
        with pytest.raises(ProviderError):
            call_chat_completion("prompt", settings(""))
