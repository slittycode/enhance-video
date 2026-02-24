"""End-to-end smoke tests using a real (tiny) video file.

These tests exercise the pipeline with actual ffmpeg calls rather than mocked
subprocesses. They validate argument parsing, video probing, frame extraction,
and the overall orchestration path — but they do NOT invoke the Real-ESRGAN
binary (which may not be available in CI).
"""

import json
import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import upscale_video

FIXTURE_VIDEO = Path(__file__).resolve().parent / "fixtures" / "tiny_input.mp4"


@unittest.skipUnless(FIXTURE_VIDEO.exists(), "requires tests/fixtures/tiny_input.mp4")
class TestE2ESmoke(unittest.TestCase):
    """End-to-end smoke tests with a real video file."""

    def test_dry_run_completes_with_exit_0(self):
        """Full pipeline dry-run with real video input returns 0."""
        with tempfile.TemporaryDirectory() as work_dir:
            output = Path(work_dir) / "output.mp4"
            rc = upscale_video.main([
                str(FIXTURE_VIDEO),
                "--output", str(output),
                "--dry-run",
                "--work-dir", work_dir,
                "--disable-runtime-guardrail",
            ])
        self.assertEqual(rc, 0)

    def test_plan_only_produces_valid_json(self):
        """--plan-only with real video emits parseable JSON with scene data."""
        with tempfile.TemporaryDirectory() as work_dir:
            with mock.patch("sys.stdout", new_callable=io.StringIO) as captured:
                rc = upscale_video.main([
                    str(FIXTURE_VIDEO),
                    "--plan-only",
                    "--scene-adaptive",
                    "--disable-runtime-guardrail",
                    "--work-dir", work_dir,
                ])

        self.assertEqual(rc, 0)
        payload = json.loads(captured.getvalue())
        self.assertIn("mode", payload)
        self.assertIn("settings", payload)
        self.assertIn("scene", payload)
        self.assertGreater(len(payload["scene"]["entries"]), 0)

    def test_video_info_reads_real_metadata(self):
        """get_video_info extracts correct metadata from the tiny fixture."""
        info = upscale_video.get_video_info("ffprobe", FIXTURE_VIDEO)
        self.assertEqual(info.width, 64)
        self.assertEqual(info.height, 64)
        self.assertAlmostEqual(info.framerate, 10.0, places=1)
        self.assertGreater(info.duration_seconds, 0)

    def test_frame_extraction_produces_jpeg_files(self):
        """extract_frames now produces JPEG (.jpg) output files."""
        with tempfile.TemporaryDirectory() as work_dir:
            frames_dir = Path(work_dir) / "frames"
            frames_dir.mkdir()
            count = upscale_video.extract_frames("ffmpeg", FIXTURE_VIDEO, frames_dir)

            self.assertGreater(count, 0)
            jpg_files = list(frames_dir.glob("frame_*.jpg"))
            png_files = list(frames_dir.glob("frame_*.png"))
            self.assertEqual(len(jpg_files), count)
            self.assertEqual(len(png_files), 0, "Expected JPEG frames, not PNG")


@unittest.skipUnless(FIXTURE_VIDEO.exists(), "requires tests/fixtures/tiny_input.mp4")
class TestNewHelpers(unittest.TestCase):
    """Tests for newly added helper functions."""

    def test_get_codec_flags_h264(self):
        flags = upscale_video.get_codec_flags("h264", "slow", 18)
        self.assertEqual(flags, ["-c:v", "libx264", "-preset", "slow", "-crf", "18"])

    def test_get_codec_flags_h265(self):
        flags = upscale_video.get_codec_flags("h265", "medium", 20)
        self.assertEqual(flags, ["-c:v", "libx265", "-preset", "medium", "-crf", "20"])

    def test_get_codec_flags_h265_hw(self):
        flags = upscale_video.get_codec_flags("h265-hw", "medium", 22)
        self.assertEqual(flags, ["-c:v", "hevc_videotoolbox", "-q:v", "22"])

    def test_get_codec_flags_invalid_raises(self):
        with self.assertRaises(ValueError):
            upscale_video.get_codec_flags("av1", "slow", 18)

    def test_check_disk_space_passes_with_ample_space(self):
        """check_disk_space should not raise when disk has plenty of room."""
        info = upscale_video.VideoInfo(
            framerate=10.0, width=64, height=64,
            audio_codec=None, has_audio=False, duration_seconds=2.0,
        )
        # Should not raise for a tiny video
        with tempfile.TemporaryDirectory() as tmp:
            upscale_video.check_disk_space(Path(tmp), info, scale=2, frame_count=20)

    def test_codec_arg_appears_in_parse_args(self):
        args = upscale_video.parse_args(["input.mp4"])
        self.assertEqual(args.codec, "h264")

    def test_codec_arg_accepts_h265(self):
        args = upscale_video.parse_args(["input.mp4", "--codec", "h265"])
        self.assertEqual(args.codec, "h265")

    def test_traced_decorator_preserves_function_name(self):
        """@_traced should preserve the wrapped function's __name__."""
        @upscale_video._traced
        def example_function():
            pass
        self.assertEqual(example_function.__name__, "example_function")


if __name__ == "__main__":
    unittest.main()
