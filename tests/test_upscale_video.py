import stat
import tempfile
import unittest
import json
import io
import subprocess
from pathlib import Path
from unittest import mock

import upscale_video


class TestBinaryResolution(unittest.TestCase):
    def test_resolve_realesrgan_binary_detects_bundled_binary(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            binary_name = upscale_video.get_realesrgan_binary_name()
            bundled = (
                root
                / "Real-ESRGAN-ncnn-vulkan"
                / "realesrgan-ncnn-vulkan-v0.2.0-macos"
                / binary_name
            )
            bundled.parent.mkdir(parents=True, exist_ok=True)
            bundled.write_text("#!/bin/sh\nexit 0\n")
            bundled.chmod(bundled.stat().st_mode | stat.S_IEXEC)

            def fake_which(command: str):
                if command in ("ffmpeg", "ffprobe"):
                    return f"/usr/bin/{command}"
                return None

            with mock.patch("upscale_video.shutil.which", side_effect=fake_which):
                resolved = upscale_video.resolve_realesrgan_binary(
                    custom_path=None,
                    search_root=root,
                )

            self.assertEqual(resolved, bundled.resolve())

    def test_resolve_realesrgan_binary_detects_bundled_binary_in_enhance_ai_dir(self):
        """Verify detection still works when vendored files live under `enhance-ai/`."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            binary_name = upscale_video.get_realesrgan_binary_name()
            bundled = (
                root
                / "enhance-ai"
                / "realesrgan-ncnn-vulkan-v0.2.0-macos"
                / binary_name
            )
            bundled.parent.mkdir(parents=True, exist_ok=True)
            bundled.write_text("#!/bin/sh\nexit 0\n")
            bundled.chmod(bundled.stat().st_mode | stat.S_IEXEC)

            def fake_which(command: str):
                if command in ("ffmpeg", "ffprobe"):
                    return f"/usr/bin/{command}"
                return None

            with mock.patch("upscale_video.shutil.which", side_effect=fake_which):
                resolved = upscale_video.resolve_realesrgan_binary(
                    custom_path=None,
                    search_root=root,
                )

            self.assertEqual(resolved, bundled.resolve())


class TestModeSelection(unittest.TestCase):
    def test_choose_upscale_mode_prefers_frame_when_resuming(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            (output_dir / "frame_00000001.png").touch()
            mode = upscale_video.choose_upscale_mode(
                requested_mode="auto",
                output_dir=output_dir,
                force=False,
            )
        self.assertEqual(mode, "frame")

    def test_choose_upscale_mode_prefers_batch_when_clean_output(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            mode = upscale_video.choose_upscale_mode(
                requested_mode="auto",
                output_dir=output_dir,
                force=False,
            )
        self.assertEqual(mode, "batch")

    def test_choose_upscale_mode_force_prefers_batch(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            (output_dir / "frame_00000001.png").touch()
            mode = upscale_video.choose_upscale_mode(
                requested_mode="auto",
                output_dir=output_dir,
                force=True,
            )
        self.assertEqual(mode, "batch")


class TestFrameExtractionReuse(unittest.TestCase):
    def test_ensure_input_frames_reuses_existing_when_not_forced(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            input_dir = Path(temp_dir) / "input_frames"
            input_dir.mkdir()
            for idx in range(1, 4):
                (input_dir / f"frame_{idx:08d}.jpg").touch()

            with mock.patch("upscale_video.extract_frames") as extract_mock:
                frame_count = upscale_video.ensure_input_frames(
                    ffmpeg_bin="/usr/bin/ffmpeg",
                    input_video=Path("/tmp/input.mp4"),
                    input_frames_dir=input_dir,
                    force=False,
                )

        self.assertEqual(frame_count, 3)
        extract_mock.assert_not_called()

    def test_ensure_input_frames_calls_extract_when_forced(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            input_dir = Path(temp_dir) / "input_frames"
            input_dir.mkdir()
            (input_dir / "frame_00000001.jpg").touch()

            with mock.patch("upscale_video.extract_frames", return_value=5) as extract_mock:
                frame_count = upscale_video.ensure_input_frames(
                    ffmpeg_bin="/usr/bin/ffmpeg",
                    input_video=Path("/tmp/input.mp4"),
                    input_frames_dir=input_dir,
                    force=True,
                )

        self.assertEqual(frame_count, 5)
        extract_mock.assert_called_once()


class TestTemporalFilter(unittest.TestCase):
    def test_temporal_filter_expression_none(self):
        self.assertIsNone(upscale_video.get_temporal_filter_expression("none"))

    def test_temporal_filter_expression_medium(self):
        expression = upscale_video.get_temporal_filter_expression("medium")
        self.assertIn("atadenoise", expression)
        self.assertIn("unsharp", expression)


class TestWorkspaceBehavior(unittest.TestCase):
    def test_prepare_workspace_uses_custom_dir_without_cleanup(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            custom = Path(temp_dir) / "workspace"
            workspace, should_cleanup = upscale_video.prepare_workspace(
                work_dir_arg=str(custom),
                keep_temp=False,
                cleanup_work_dir=False,
            )

        self.assertEqual(workspace, custom.resolve())
        self.assertFalse(should_cleanup)

    def test_prepare_workspace_temp_dir_cleans_by_default(self):
        workspace, should_cleanup = upscale_video.prepare_workspace(
            work_dir_arg=None,
            keep_temp=False,
            cleanup_work_dir=False,
        )
        try:
            self.assertTrue(workspace.exists())
            self.assertTrue(should_cleanup)
        finally:
            if workspace.exists():
                workspace.rmdir()


class TestQualityProfile(unittest.TestCase):
    def test_apply_quality_profile_sets_defaults(self):
        args = upscale_video.parse_args(["input.mp4", "--profile", "max_quality"])
        upscale_video.apply_quality_profile(args, cli_overrides=set())

        self.assertTrue(args.tta)
        self.assertEqual(args.temporal_filter, "strong")
        self.assertEqual(args.preset, "veryslow")
        self.assertEqual(args.crf, 14)

    def test_apply_quality_profile_respects_explicit_override(self):
        args = upscale_video.parse_args(
            ["input.mp4", "--profile", "max_quality", "--preset", "medium"]
        )
        cli_overrides = {"preset"}
        upscale_video.apply_quality_profile(args, cli_overrides=cli_overrides)

        self.assertEqual(args.preset, "medium")
        self.assertTrue(args.tta)


class TestWorkspaceFingerprint(unittest.TestCase):
    def test_workspace_cache_validation_detects_mismatch(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest = Path(temp_dir) / "workspace_manifest.json"
            manifest.write_text(json.dumps({"fingerprint": {"scale": 2}}))
            current = {"scale": 4}

            valid, reason = upscale_video.validate_workspace_cache(manifest, current)

        self.assertFalse(valid)
        self.assertIn("fingerprint mismatch", reason)

    def test_workspace_cache_validation_accepts_match(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest = Path(temp_dir) / "workspace_manifest.json"
            payload = {"fingerprint": {"scale": 2, "model": "realesrgan-x4plus"}}
            manifest.write_text(json.dumps(payload))

            valid, reason = upscale_video.validate_workspace_cache(
                manifest,
                {"scale": 2, "model": "realesrgan-x4plus"},
            )

        self.assertTrue(valid)
        self.assertEqual(reason, "cache valid")


class TestGuardrailCandidates(unittest.TestCase):
    def test_build_guardrail_candidates_degrades_quality(self):
        args = upscale_video.parse_args(["input.mp4", "--profile", "max_quality"])
        upscale_video.apply_quality_profile(args, cli_overrides=set())

        candidates = upscale_video.build_guardrail_candidates(args)

        self.assertGreaterEqual(len(candidates), 3)
        self.assertTrue(candidates[0].tta)
        self.assertIn(candidates[-1].temporal_filter, ("light", "none"))


class TestCalibration(unittest.TestCase):
    def test_load_calibration_returns_default_for_missing_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            calibration_path = Path(temp_dir) / "calibration.json"
            data = upscale_video.load_calibration(calibration_path)
        self.assertEqual(data, {"entries": {}})

    def test_update_calibration_entry_records_fps(self):
        calibration = {"entries": {}}
        key = "machine|realesrgan-x4plus|2x|320x240|tta1|strong"

        upscale_video.update_calibration_entry(
            calibration,
            key=key,
            fps=1.25,
            source="actual",
        )
        entry = calibration["entries"][key]
        self.assertGreater(entry["ema_fps"], 0.0)
        self.assertEqual(entry["samples"], 1)

    def test_estimate_runtime_prefers_calibration_entry(self):
        calibration = {
            "entries": {
                "k": {
                    "ema_fps": 2.0,
                    "samples": 3,
                    "updated_at": 1700000000,
                }
            }
        }
        estimate = upscale_video.estimate_runtime_from_calibration(
            calibration,
            key="k",
            total_frames=120,
            candidate_name="full_quality",
        )
        self.assertIsNotNone(estimate)
        self.assertEqual(estimate.source, "calibration")
        self.assertEqual(estimate.projected_seconds, 60.0)


class TestSceneAdaptiveHelpers(unittest.TestCase):
    def test_build_scene_ranges_creates_expected_chunks(self):
        ranges = upscale_video.build_scene_ranges(
            total_frames=12,
            boundaries=[4, 8],
            min_scene_frames=1,
        )
        self.assertEqual(ranges, [(1, 4), (5, 8), (9, 12)])

    def test_build_scene_ranges_merges_short_scenes(self):
        ranges = upscale_video.build_scene_ranges(
            total_frames=10,
            boundaries=[2, 3, 8],
            min_scene_frames=4,
        )
        self.assertEqual(ranges, [(1, 3), (4, 10)])

    def test_select_scene_candidate_chooses_first_under_budget(self):
        candidates = [
            upscale_video.GuardrailCandidate("hq", True, "strong", "veryslow", 14),
            upscale_video.GuardrailCandidate("mq", False, "medium", "slow", 16),
            upscale_video.GuardrailCandidate("lq", False, "none", "medium", 17),
        ]
        projected = {"hq": 120.0, "mq": 90.0, "lq": 60.0}

        selected = upscale_video.select_scene_candidate_by_projection(
            candidates,
            projected_seconds_by_name=projected,
            budget_seconds=95.0,
        )
        self.assertEqual(selected.name, "mq")

    def test_select_scene_candidate_texture_priority_biases_budget(self):
        candidates = [
            upscale_video.GuardrailCandidate("hq", True, "strong", "veryslow", 14),
            upscale_video.GuardrailCandidate("mq", False, "medium", "slow", 16),
            upscale_video.GuardrailCandidate("lq", False, "none", "medium", 18),
        ]
        projected = {"hq": 105.0, "mq": 95.0, "lq": 80.0}

        baseline = upscale_video.select_scene_candidate_by_projection(
            candidates,
            projected_seconds_by_name=projected,
            budget_seconds=100.0,
        )
        texture_biased = upscale_video.select_scene_candidate_by_projection(
            candidates,
            projected_seconds_by_name=projected,
            budget_seconds=100.0,
            texture_score=1.0,
            texture_priority=1.0,
        )

        self.assertEqual(baseline.name, "mq")
        self.assertEqual(texture_biased.name, "hq")

    def test_build_scene_texture_scores_normalizes_by_scene(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            frames: list[Path] = []
            sizes = [100, 120, 1000, 1020]
            for idx, size in enumerate(sizes, start=1):
                frame = root / f"frame_{idx:08d}.png"
                frame.write_bytes(b"x" * size)
                frames.append(frame)

            scores = upscale_video.build_scene_texture_scores(
                frames=frames,
                scene_ranges=[(1, 2), (3, 4)],
                sample_size=2,
            )

        self.assertEqual(len(scores), 2)
        self.assertLess(scores[0], scores[1])
        self.assertGreaterEqual(scores[0], 0.0)
        self.assertLessEqual(scores[1], 1.0)

    def test_scene_adaptive_candidates_use_quality_ladder_for_max_quality(self):
        args = upscale_video.parse_args(["input.mp4", "--profile", "max_quality"])
        upscale_video.apply_quality_profile(args, cli_overrides=set())

        candidates = upscale_video.build_scene_adaptive_candidates(args)

        names = [candidate.name for candidate in candidates]
        self.assertIn("full_quality", names)
        self.assertIn("last_resort", names)


class TestMainBehavior(unittest.TestCase):
    def test_main_returns_1_when_output_equals_input(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "input.mp4"
            input_path.touch()

            with mock.patch("sys.stderr"):
                rc = upscale_video.main([str(input_path), "--output", str(input_path)])

        self.assertEqual(rc, 1)

    def test_main_dry_run_returns_0(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "input.mp4"
            output_path = Path(temp_dir) / "output.mp4"
            input_path.touch()

            with mock.patch("upscale_video.resolve_toolchain") as resolve_toolchain_mock:
                resolve_toolchain_mock.return_value = mock.Mock()

                with mock.patch(
                    "upscale_video.run_pipeline",
                    return_value=0,
                ) as run_pipeline_mock:
                    rc = upscale_video.main(
                        [
                            str(input_path),
                            "--output",
                            str(output_path),
                            "--dry-run",
                        ]
                    )

        self.assertEqual(rc, 0)
        run_pipeline_mock.assert_called_once()

    def test_main_plan_only_outputs_clean_json(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "input.mp4"
            input_path.touch()

            candidate = upscale_video.GuardrailCandidate(
                "full_quality",
                True,
                "strong",
                "veryslow",
                14,
            )
            scene_plan = [
                upscale_video.ScenePlanEntry(
                    scene_number=1,
                    start_frame=1,
                    end_frame=60,
                    frame_count=60,
                    texture_score=0.8,
                    budget_seconds=None,
                    selected_candidate=candidate,
                    projected_seconds_by_name={},
                    source_by_name={},
                ),
                upscale_video.ScenePlanEntry(
                    scene_number=2,
                    start_frame=61,
                    end_frame=120,
                    frame_count=60,
                    texture_score=0.2,
                    budget_seconds=None,
                    selected_candidate=candidate,
                    projected_seconds_by_name={},
                    source_by_name={},
                ),
            ]

            toolchain = upscale_video.Toolchain(
                ffmpeg="ffmpeg",
                ffprobe="ffprobe",
                realesrgan_binary=Path("/tmp/realesrgan"),
                model_path=None,
            )
            info = upscale_video.VideoInfo(
                framerate=30.0,
                width=1920,
                height=1080,
                audio_codec="aac",
                has_audio=True,
                duration_seconds=4.0,
            )

            with mock.patch("upscale_video.resolve_toolchain", return_value=toolchain):
                with mock.patch("upscale_video.get_video_info", return_value=info):
                    with mock.patch("upscale_video.ensure_input_frames", return_value=120):
                        with mock.patch("upscale_video.detect_scene_boundaries", return_value=[60]):
                            with mock.patch(
                                "upscale_video.build_scene_ranges",
                                return_value=[(1, 60), (61, 120)],
                            ):
                                with mock.patch(
                                    "upscale_video.plan_scene_adaptive_strategy",
                                    return_value=scene_plan,
                                ):
                                    with mock.patch("sys.stdout", new_callable=io.StringIO) as stdout_mock:
                                        rc = upscale_video.main(
                                            [
                                                str(input_path),
                                                "--plan-only",
                                                "--scene-adaptive",
                                                "--disable-runtime-guardrail",
                                            ]
                                        )

            self.assertEqual(rc, 0)
            payload = json.loads(stdout_mock.getvalue())
            self.assertEqual(payload["mode"], "scene")
            self.assertTrue(payload["settings"]["scene_adaptive"])
            self.assertEqual(len(payload["scene"]["entries"]), 2)
            self.assertTrue(payload["settings"]["auto_clean_scene_chunks"])


class TestSceneChunkHelpers(unittest.TestCase):
    def test_concat_scene_chunks_reencodes_when_copy_fails(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            chunk_a = root / "scene_0001.mp4"
            chunk_b = root / "scene_0002.mp4"
            chunk_a.write_bytes(b"a")
            chunk_b.write_bytes(b"b")
            output = root / "scene_concat_raw.mp4"

            copy_fail = subprocess.CompletedProcess(
                args=["ffmpeg"],
                returncode=1,
                stdout="",
                stderr="copy failed",
            )
            reencode_ok = subprocess.CompletedProcess(
                args=["ffmpeg"],
                returncode=0,
                stdout="",
                stderr="",
            )

            with mock.patch(
                "upscale_video.run_subprocess",
                side_effect=[copy_fail, reencode_ok],
            ) as run_subprocess_mock:
                concat_manifest = upscale_video.concat_scene_chunks(
                    "ffmpeg",
                    [chunk_a, chunk_b],
                    output,
                    preset="slow",
                    crf=16,
                )

                self.assertEqual(run_subprocess_mock.call_count, 2)
                self.assertTrue(concat_manifest.exists())

    def test_mux_audio_to_video_moves_when_audio_missing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            video_path = root / "video_only.mp4"
            output_path = root / "final.mp4"
            video_path.write_bytes(b"video")

            upscale_video.mux_audio_to_video(
                "ffmpeg",
                video_path,
                output_path,
                audio_path=None,
                audio_bitrate="192k",
            )

            self.assertFalse(video_path.exists())
            self.assertTrue(output_path.exists())

    def test_mux_audio_to_video_uses_ffmpeg_when_audio_present(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            video_path = root / "video_only.mp4"
            audio_path = root / "audio.m4a"
            output_path = root / "final.mp4"
            video_path.write_bytes(b"video")
            audio_path.write_bytes(b"audio")

            # Stream-copy attempt "fails", then AAC fallback succeeds.
            copy_fail = subprocess.CompletedProcess(
                args=["ffmpeg"], returncode=1, stdout="", stderr="copy failed",
            )
            aac_ok = subprocess.CompletedProcess(
                args=["ffmpeg"], returncode=0, stdout="", stderr="",
            )

            with mock.patch(
                "upscale_video.run_subprocess",
                side_effect=[copy_fail, aac_ok],
            ) as run_subprocess_mock:
                upscale_video.mux_audio_to_video(
                    "ffmpeg",
                    video_path,
                    output_path,
                    audio_path=audio_path,
                    audio_bitrate="256k",
                )

        self.assertEqual(run_subprocess_mock.call_count, 2)
        # First call: stream-copy attempt
        first_cmd = run_subprocess_mock.call_args_list[0].args[0]
        self.assertIn("-c:a", first_cmd)
        copy_idx = first_cmd.index("-c:a")
        self.assertEqual(first_cmd[copy_idx + 1], "copy")
        # Second call: AAC fallback
        second_cmd = run_subprocess_mock.call_args_list[1].args[0]
        aac_idx = second_cmd.index("-c:a")
        self.assertEqual(second_cmd[aac_idx + 1], "aac")


if __name__ == "__main__":
    unittest.main()
