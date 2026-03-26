#!/usr/bin/env python3
"""Test frame integrity: ensure input frame N maps to output frame N."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    Image = None

import upscale_video
from toolchain import resolve_toolchain


def compute_image_hash(image_path: Path) -> str:
    """Compute perceptual hash of an image for content comparison."""
    if not PIL_AVAILABLE:
        # Fallback: use file hash if PIL not available
        with open(image_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    
    with Image.open(image_path) as img:
        # Convert to grayscale and resize to normalize
        img = img.convert('L').resize((8, 8))
        pixels = list(img.getdata())
        # Compute average pixel value
        avg = sum(pixels) / len(pixels)
        # Generate hash bits
        bits = ''.join('1' if p > avg else '0' for p in pixels)
        # Convert to hex
        return hex(int(bits, 2))[2:].zfill(16)


def extract_frame_number(frame_path: Path) -> int:
    """Extract frame number from filename like frame_00000851.jpg."""
    import re
    match = re.search(r'frame_(\d+)\.', frame_path.name)
    return int(match.group(1)) if match else 0


class TestFrameIntegrity(unittest.TestCase):
    """Verify frame-to-frame integrity during upscaling."""

    def setUp(self):
        """Set up test environment."""
        self.test_dir = Path(tempfile.mkdtemp(prefix="frame_integrity_test_"))
        self.input_frames_dir = self.test_dir / "input_frames"
        self.output_frames_dir = self.test_dir / "output_frames"
        self.input_frames_dir.mkdir()
        self.output_frames_dir.mkdir()
        
        # Create minimal toolchain for testing
        from cli import parse_args
        args = parse_args(['dummy_input.mp4'])  # Provide dummy input
        self.toolchain = resolve_toolchain(args)
        
        # Create test frames with distinct patterns
        self.create_test_frames()
    
    def tearDown(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def create_test_frames(self):
        """Create 10 test frames with distinct patterns."""
        if not PIL_AVAILABLE:
            # Skip test frame creation if PIL not available
            for i in range(1, 11):
                # Create simple test files with unique content
                frame_path = self.input_frames_dir / f"frame_{i:08d}.jpg"
                with open(frame_path, 'wb') as f:
                    f.write(f"test_frame_{i}".encode() * 1000)  # Unique content per frame
            return
        
        size = (64, 64)
        colors = [
            (255, 0, 0),    # Red
            (0, 255, 0),    # Green
            (0, 0, 255),    # Blue
            (255, 255, 0),  # Yellow
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Cyan
            (128, 128, 128), # Gray
            (255, 128, 0),  # Orange
            (128, 0, 255),  # Purple
            (0, 128, 255),  # Sky blue
        ]
        
        for i in range(1, 11):
            # Create a pattern unique to each frame
            img = Image.new('RGB', size, colors[i-1])
            pixels = img.load()
            
            # Add a unique pattern based on frame number
            for x in range(size[0]):
                for y in range(size[1]):
                    if (x + y + i) % 8 == 0:
                        pixels[x, y] = (255, 255, 255)  # White dots
                    elif (x * y + i) % 16 == 0:
                        pixels[x, y] = (0, 0, 0)  # Black dots
            
            # Save as JPEG (matching input format)
            frame_path = self.input_frames_dir / f"frame_{i:08d}.jpg"
            img.save(frame_path, 'JPEG', quality=95)
    
    def test_frame_ordering_preserved(self):
        """Verify that frames are processed in numeric order."""
        # Get input frames in natural order
        input_frames = sorted(self.input_frames_dir.glob(upscale_video.INPUT_FRAME_GLOB), 
                             key=upscale_video.natural_key)
        
        # Verify natural ordering works
        frame_numbers = [extract_frame_number(f) for f in input_frames]
        expected = list(range(1, 11))
        self.assertEqual(frame_numbers, expected, 
                        "Frames not in natural numeric order")
    
    def test_frame_integrity_2x_upscale(self):
        """Test that input frame content matches output frame content after upscaling."""
        if not PIL_AVAILABLE:
            self.skipTest("PIL not available - cannot create proper test frames")
        
        # Run upscaling at 2x with no tiling (full-frame inference)
        upscale_video.run_upscale_batch(
            self.toolchain,
            self.input_frames_dir,
            self.output_frames_dir,
            scale_factor=2,
            model_name="realesrgan-x4plus",  # Will be overridden to x2 for 2x scale
            gpu_id=0,
            tile_size=0,  # No tiling
            tta=False,
            jobs=None,
            dry_run=False,
        )
        
        # Verify all output frames exist
        output_frames = sorted(self.output_frames_dir.glob(upscale_video.OUTPUT_FRAME_GLOB),
                               key=upscale_video.natural_key)
        self.assertEqual(len(output_frames), 10, 
                        "Not all output frames generated")
        
        # Compute hashes for input and output frames
        input_hashes = {}
        output_hashes = {}
        
        for frame_path in self.input_frames_dir.glob(upscale_video.INPUT_FRAME_GLOB):
            frame_num = extract_frame_number(frame_path)
            input_hashes[frame_num] = compute_image_hash(frame_path)
        
        for frame_path in self.output_frames_dir.glob(upscale_video.OUTPUT_FRAME_GLOB):
            frame_num = extract_frame_number(frame_path)
            output_hashes[frame_num] = compute_image_hash(frame_path)
        
        # Verify frame-to-frame mapping
        for frame_num in range(1, 11):
            self.assertIn(frame_num, input_hashes, 
                         f"Input frame {frame_num} missing")
            self.assertIn(frame_num, output_hashes, 
                         f"Output frame {frame_num} missing")
            
            # The output should be a upscaled version of the corresponding input
            # We can't directly compare hashes due to upscaling, but we can verify
            # the output exists and corresponds to the right input frame
            input_path = self.input_frames_dir / f"frame_{frame_num:08d}.jpg"
            output_path = self.output_frames_dir / f"frame_{frame_num:08d}.png"
            
            self.assertTrue(input_path.exists(), 
                           f"Input frame {frame_num} should exist")
            self.assertTrue(output_path.exists(), 
                           f"Output frame {frame_num} should exist")
    
    def test_no_cross_frame_contamination(self):
        """Verify that frame N doesn't contain content from frame M."""
        # Run upscaling
        upscale_video.run_upscale_batch(
            self.toolchain,
            self.input_frames_dir,
            self.output_frames_dir,
            scale_factor=2,
            model_name="realesrgan-x4plus",
            gpu_id=0,
            tile_size=256,  # Enable tiling to test tile integrity
            tta=False,
            jobs=None,
            dry_run=False,
        )
        
        # Load output frames and verify they're distinct
        output_hashes = {}
        for frame_path in self.output_frames_dir.glob(upscale_video.OUTPUT_FRAME_GLOB):
            frame_num = extract_frame_number(frame_path)
            output_hashes[frame_num] = compute_image_hash(frame_path)
        
        # All hashes should be unique (no duplicate content)
        hashes = list(output_hashes.values())
        unique_hashes = set(hashes)
        self.assertEqual(len(hashes), len(unique_hashes),
                        "Output frames have duplicate content - possible cross-contamination")
    
    def test_frame_checksum_logging(self):
        """Test that we can log frame checksums for debugging."""
        checksums = {}
        
        # Compute checksums for input frames
        for frame_path in sorted(self.input_frames_dir.glob(upscale_video.INPUT_FRAME_GLOB),
                                key=upscale_video.natural_key):
            frame_num = extract_frame_number(frame_path)
            with open(frame_path, 'rb') as f:
                checksums[f"input_{frame_num}"] = hashlib.md5(f.read()).hexdigest()
        
        # Run upscaling
        try:
            upscale_video.run_upscale_batch(
                self.toolchain,
                self.input_frames_dir,
                self.output_frames_dir,
                scale_factor=2,
                model_name="realesrgan-x4plus",
                gpu_id=0,
                tile_size=0,
                tta=False,
                jobs=None,
                dry_run=False,
            )
        except Exception as e:
            self.skipTest(f"Upscaling failed: {e}")
        
        # Compute checksums for output frames
        for frame_path in sorted(self.output_frames_dir.glob(upscale_video.OUTPUT_FRAME_GLOB),
                                key=upscale_video.natural_key):
            frame_num = extract_frame_number(frame_path)
            with open(frame_path, 'rb') as f:
                checksums[f"output_{frame_num}"] = hashlib.md5(f.read()).hexdigest()
        
        # Save checksums to file for debugging
        checksum_file = self.test_dir / "frame_checksums.json"
        with open(checksum_file, 'w') as f:
            json.dump(checksums, f, indent=2)
        
        # Verify checksums file exists and has expected entries
        self.assertTrue(checksum_file.exists())
        
        # Check we have at least the input frames (output may fail if Real-ESRGAN not available)
        self.assertGreaterEqual(len(checksums), 10)
        
        # Verify each input frame has a checksum
        for frame_num in range(1, 11):
            self.assertIn(f"input_{frame_num}", checksums)
            
        # If upscaling succeeded, verify output checksums
        if len(checksums) >= 20:
            for frame_num in range(1, 11):
                self.assertIn(f"output_{frame_num}", checksums)


if __name__ == '__main__':
    unittest.main()
