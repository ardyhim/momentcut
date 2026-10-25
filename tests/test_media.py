from __future__ import annotations

import pytest

from momentcut.media import (
    MediaError,
    ffmpeg_cut_command,
    fingerprint_compatible,
    probe_media,
    render_cut,
    source_fingerprint,
    verify_rendered_clip,
)


class TestProbe:
    def test_probe(self, video):
        info = probe_media(video)
        assert info.duration == pytest.approx(50.0, abs=0.1)
        assert info.width == 640
        assert info.height == 360
        assert info.has_audio is True

    def test_missing_file(self, tmp_path):
        with pytest.raises(MediaError):
            probe_media(tmp_path / "nope.mp4")


class TestFingerprint:
    def test_compatible_ignores_mtime(self, video):
        info = probe_media(video)
        first = source_fingerprint(video, info)
        second = dict(first)
        second["mtime_ns"] += 1000
        assert fingerprint_compatible(first, second)

    def test_incompatible_on_size(self, video):
        info = probe_media(video)
        first = source_fingerprint(video, info)
        second = dict(first)
        second["size"] += 1
        assert not fingerprint_compatible(first, second)

    def test_incompatible_on_hash(self, video):
        info = probe_media(video)
        first = source_fingerprint(video, info)
        second = dict(first)
        second["quick_hash"] = "different"
        assert not fingerprint_compatible(first, second)


class TestCommands:
    def test_original_command(self, video, tmp_path):
        command = ffmpeg_cut_command(
            video, tmp_path / "out.mp4", 10.0, 12.0, aspect="original"
        )
        joined = " ".join(command)
        assert "-map 0:v:0" in joined
        assert "-map 0:a:0" in joined
        assert "-filter_complex" not in joined
        assert "-ss 10.000" in joined
        assert "-t 12.000" in joined

    def test_vertical_command(self, video, tmp_path):
        command = ffmpeg_cut_command(
            video, tmp_path / "out.mp4", 10.0, 12.0, aspect="vertical"
        )
        joined = " ".join(command)
        assert "boxblur" in joined
        assert "-map [v]" in joined
        assert "1080:1920" in joined
        assert "-map 0:a:0" in joined

    def test_no_audio_command(self, video, tmp_path):
        command = ffmpeg_cut_command(
            video, tmp_path / "out.mp4", 10.0, 5.0, has_audio=False
        )
        assert "0:a:0" not in " ".join(command)


class TestRenderAndVerify:
    def test_render_and_verify(self, video, tmp_path):
        output = tmp_path / "clip.mp4"
        render_cut(video, output, 5.0, 10.0, aspect="original")
        info = verify_rendered_clip(output, 10.0, expect_audio=True)
        assert info["has_audio"] is True

    def test_verify_wrong_duration(self, video, tmp_path):
        output = tmp_path / "clip.mp4"
        render_cut(video, output, 5.0, 5.0, aspect="original")
        with pytest.raises(MediaError):
            verify_rendered_clip(output, 20.0, expect_audio=True)

    def test_verify_vertical_dimensions(self, video, tmp_path):
        output = tmp_path / "vertical.mp4"
        render_cut(video, output, 5.0, 5.0, aspect="vertical")
        info = verify_rendered_clip(
            output, 5.0, expect_audio=True, vertical=True
        )
        assert info["width"] == 1080
        assert info["height"] == 1920

    def test_unicode_and_space_in_paths(self, video, tmp_path):
        source = tmp_path / "video unicode üji.mp4"
        output = tmp_path / "clip unicode üji.mp4"
        source.write_bytes(video.read_bytes())
        info = probe_media(source)
        assert info.duration == pytest.approx(50.0, abs=0.1)
        render_cut(source, output, 5.0, 4.0, aspect="original")
        verified = verify_rendered_clip(output, 4.0, expect_audio=True)
        assert verified["has_audio"] is True
