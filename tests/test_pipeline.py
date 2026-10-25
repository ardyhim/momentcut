from __future__ import annotations

import copy
import json
import shutil
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import ClassVar

import pytest

from momentcut.config import ApiSettings
from momentcut.media import probe_media
from momentcut.pipeline import PipelineError, analyze, import_analysis
from momentcut.render import render_run
from momentcut.util import read_json

from .conftest import response_payload, transcript_payload, write_json

SETTINGS = {
    "min_duration": 8.0,
    "max_duration": 30.0,
    "max_clips": 3,
    "min_score": 60,
    "block_seconds": 120.0,
}


def make_manual_run(video, tmp_path, *, settings=None):
    run = tmp_path / "run"
    transcript_file = write_json(tmp_path / "transcript.json", transcript_payload())
    result = analyze(
        video,
        run,
        overrides=settings or SETTINGS,
        transcript_path=transcript_file,
    )
    assert result.blocks
    response_file = run / "analysis" / "responses" / "block_001.json"
    write_json(response_file, response_payload())
    return run


def long_transcript():
    units = []
    for index in range(14):
        units.append(
            {
                "id": f"L{index + 1:04d}",
                "start": index * 10.0 + 0.2,
                "end": index * 10.0 + 9.8,
                "text": f"This is line number {index + 1} and it says one complete thing.",
            }
        )
    return {
        "language": "en",
        "backend": "fixture",
        "source_duration": 140.0,
        "units": units,
    }


class TestManualFlow:
    def test_end_to_end(self, video, tmp_path, capsys):
        run = make_manual_run(video, tmp_path)
        assert (run / "analysis" / "prompts" / "block_001.md").exists()
        assert (run / "transcript.json").exists()

        result = import_analysis(run)
        assert result.parsed_candidates == 7
        assert result.selected == 2
        assert result.rejected == 2
        assert result.coverage["complete"] is True
        analysis = read_json(run / "analysis.json")
        assert len(analysis["clips"]) == 2
        for clip in analysis["clips"]:
            assert 0 <= clip["start"] < clip["end"] <= 50.0
            assert 8.0 <= clip["duration"] <= 30.01
            assert clip["start_line"].startswith("L")
            assert clip["end_line"].startswith("L")
            assert clip["title"]
            assert clip["hook"]
        assert analysis["rejected"]
        assert analysis["dropped"]

        rendered = render_run(run, aspect="original")
        assert rendered.failed == 0
        assert rendered.rendered == 2
        for entry in read_json(run / "render_manifest.json")["clips"].values():
            if entry["status"] == "rendered":
                output = run / entry["path"]
                assert output.exists()
                info = probe_media(output)
                assert abs(info.duration - entry["duration"]) < 1.0
                assert info.has_audio

        resumed = render_run(run, aspect="original")
        assert resumed.rendered == 0
        assert resumed.skipped == 2

        vertical = render_run(run, aspect="vertical")
        assert vertical.failed == 0
        assert vertical.rendered == 2
        for entry in read_json(run / "render_manifest.json")["clips"].values():
            if entry["aspect"] == "vertical" and entry["status"] == "rendered":
                info = probe_media(run / entry["path"])
                assert (info.width, info.height) == (1080, 1920)
                assert info.has_audio

        vertical_resumed = render_run(run, aspect="vertical")
        assert vertical_resumed.rendered == 0
        assert vertical_resumed.skipped == 2

    def test_reimport_overrides_without_new_ai(self, video, tmp_path):
        run = make_manual_run(video, tmp_path)
        result = import_analysis(run, min_score=85, max_clips=1)
        assert result.selected == 1
        analysis = read_json(run / "analysis.json")
        assert len(analysis["clips"]) == 1
        assert analysis["clips"][0]["selection_score"] >= 85
        # The source response file is still there; no provider call is involved.
        assert (run / "analysis" / "responses" / "block_001.json").exists()

    def test_analyze_reuses_work_then_archives_old_responses(
        self, video, tmp_path, capsys
    ):
        run = make_manual_run(video, tmp_path)
        response_file = run / "analysis" / "responses" / "block_001.json"
        assert response_file.exists()

        analyze(
            video,
            run,
            overrides=SETTINGS,
        )
        output = capsys.readouterr().out
        assert "reused" in output
        assert response_file.exists()

        analyze(
            video,
            run,
            overrides={**SETTINGS, "max_clips": 5},
        )
        output = capsys.readouterr().out
        assert "reused" in output
        assert response_file.exists(), "max-clips must not invalidate analysis"

        analyze(
            video,
            run,
            overrides={**SETTINGS, "focus": "funny moments"},
        )
        stale = list((run / "analysis").glob("responses.stale-*"))
        assert len(stale) == 1
        assert any(path.is_file() for path in stale[0].rglob("*"))
        assert not list((run / "analysis" / "responses").glob("*.json"))
        assert "funny moments" in (
            run / "analysis" / "prompts" / "block_001.md"
        ).read_text(encoding="utf-8")

    def test_source_can_be_relocated_for_render(self, video, tmp_path):
        run = make_manual_run(video, tmp_path)
        import_analysis(run)
        moved = tmp_path / "moved.mp4"
        shutil.copy2(video, moved)
        result = render_run(run, source_override=moved, aspect="original", force=True)
        assert result.failed == 0
        assert result.rendered == 2

    def test_different_source_is_refused(self, video, other_video, tmp_path):
        run = make_manual_run(video, tmp_path)
        with pytest.raises(PipelineError):
            import_analysis(run, source_override=other_video)
        with pytest.raises(PipelineError):
            render_run(run, source_override=other_video)


class _ApiHandler(BaseHTTPRequestHandler):
    payload: ClassVar[dict] = {}

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        self.rfile.read(length)
        content = json.dumps(self.payload)
        body = {
            "choices": [
                {
                    "message": {"content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 100, "completion_tokens": 50},
        }
        data = json.dumps(body).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format, *args):
        pass


class TestApiMode:
    def test_analyze_api_calls_provider_and_imports(self, video, tmp_path):
        _ApiHandler.payload = response_payload()
        server = ThreadingHTTPServer(("127.0.0.1", 0), _ApiHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            run = tmp_path / "api-run"
            transcript_file = write_json(
                tmp_path / "api-transcript.json", transcript_payload()
            )
            api = ApiSettings(
                base_url=f"http://127.0.0.1:{server.server_port}/v1",
                model="test-model",
                api_key="secret-key",
                max_attempts=1,
                timeout=5,
                retry_base_seconds=0.001,
            )
            result = analyze(
                video,
                run,
                overrides=SETTINGS,
                transcript_path=transcript_file,
                api=api,
            )
            assert result.api_complete is True
            assert result.import_result is not None
            assert result.import_result.selected == 2
            assert (run / "analysis" / "responses" / "block_001.json").exists()
            run_text = (run / "run.json").read_text(encoding="utf-8")
            assert "secret-key" not in run_text
            analysis_text = (run / "analysis.json").read_text(encoding="utf-8")
            assert "secret-key" not in analysis_text
            wrapper = read_json(run / "analysis" / "responses" / "block_001.json")
            assert wrapper["block_id"] == "B001"
            assert "secret-key" not in json.dumps(wrapper)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)


class TestInvalidAndPartial:
    def test_cli_entry_points(self, video, tmp_path, capsys):
        from momentcut.cli import main

        run = tmp_path / "cli-run"
        transcript_file = write_json(tmp_path / "cli-transcript.json", transcript_payload())
        assert (
            main(
                [
                    "analyze",
                    str(video),
                    "--run",
                    str(run),
                    "--transcript",
                    str(transcript_file),
                    "--min-duration",
                    "8",
                    "--max-duration",
                    "30",
                    "--max-clips",
                    "3",
                    "--block-seconds",
                    "120",
                ]
            )
            == 0
        )
        write_json(run / "analysis" / "responses" / "block_001.json", response_payload())
        assert main(["import", str(run)]) == 0
        assert main(["render", str(run), "--aspect", "vertical"]) == 0
        assert list((run / "clips").glob("*_vertical.mp4"))
        assert main(["analyze", str(tmp_path / "missing.mp4"), "--run", str(tmp_path / "x")]) == 1
        capsys.readouterr()

    def test_unparseable_response_writes_report_then_errors(self, video, tmp_path):
        run = tmp_path / "run"
        analyze(
            video,
            run,
            overrides=SETTINGS,
            transcript_path=write_json(tmp_path / "t.json", transcript_payload()),
        )
        (run / "analysis" / "responses" / "bad.json").write_text(
            "I refuse to return JSON.", encoding="utf-8"
        )
        with pytest.raises(PipelineError):
            import_analysis(run)
        assert (run / "analysis.json").exists()
        analysis = read_json(run / "analysis.json")
        assert analysis["rejected"]
        assert "Could not find a JSON" in analysis["rejected"][0]["reason"]

    def test_partial_coverage_is_reported(self, long_video, tmp_path):
        run = tmp_path / "run"
        transcript_file = write_json(tmp_path / "long-transcript.json", long_transcript())
        analyze(
            long_video,
            run,
            overrides=SETTINGS,
            transcript_path=transcript_file,
        )
        payload = response_payload()
        # Shorten the end anchor to a line that exists in the 14-line transcript.
        candidate = copy.deepcopy(payload["clips"][0])
        candidate.update(
            {
                "start_line": "L0002",
                "end_line": "L0004",
                "start_quote": "This is line number 2 and it says one complete thing.",
                "end_quote": "This is line number 4 and it says one complete thing.",
            }
        )
        write_json(
            run / "analysis" / "responses" / "block_001.json",
            {
                "schema_version": "1.0",
                "block_id": "B001",
                "clips": [candidate],
            },
        )
        result = import_analysis(run)
        assert result.coverage["blocks_total"] > 1
        assert result.coverage["complete"] is False
        assert result.coverage["missing_blocks"]
        assert result.selected == 1
        assert any("Coverage is incomplete" in warning for warning in result.warnings)
