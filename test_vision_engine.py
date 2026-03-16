"""Unit tests for vision_engine fretboard detection and fallback logic.

These tests exercise ``detect_fretboard_lines`` angle filtering and
the ``VisionEngine`` fretboard fallback mechanism without requiring a
camera, GPU, or model file.
"""

import math
import unittest
from unittest.mock import patch, MagicMock

import cv2
import numpy as np

import vision_engine


# ---------------------------------------------------------------------------
# Tests – detect_fretboard_lines angle filtering
# ---------------------------------------------------------------------------

class TestDetectFretboardLinesAngle(unittest.TestCase):
    """Verify that the 45° angle threshold is applied correctly."""

    def _frame_with_line(self, angle_deg: float, width: int = 640, height: int = 480):
        """Return a grayscale frame containing a single white line at *angle_deg*."""
        frame = np.zeros((height, width), dtype=np.uint8)
        cx, cy = width // 2, height // 2
        length = width // 3
        rad = math.radians(angle_deg)
        dx = int(length * math.cos(rad))
        dy = int(length * math.sin(rad))
        cv2.line(frame, (cx - dx, cy - dy), (cx + dx, cy + dy), 255, 2)
        return frame

    def test_horizontal_line_detected(self):
        """A perfectly horizontal line (0°) should be detected."""
        gray = self._frame_with_line(0)
        lines = vision_engine.detect_fretboard_lines(gray)
        self.assertGreater(len(lines), 0)

    def test_line_at_30_degrees_detected(self):
        """A 30° line should now be detected (was rejected at 20° threshold)."""
        gray = self._frame_with_line(30)
        lines = vision_engine.detect_fretboard_lines(gray)
        self.assertGreater(len(lines), 0)

    def test_line_at_44_degrees_detected(self):
        """A 44° line should be detected under the 45° threshold."""
        gray = self._frame_with_line(44)
        lines = vision_engine.detect_fretboard_lines(gray)
        self.assertGreater(len(lines), 0)

    def test_vertical_line_rejected(self):
        """A vertical line (90°) should be rejected."""
        gray = self._frame_with_line(90)
        lines = vision_engine.detect_fretboard_lines(gray)
        self.assertEqual(len(lines), 0)

    def test_line_at_70_degrees_rejected(self):
        """A 70° line is beyond the 45° threshold and should be rejected."""
        gray = self._frame_with_line(70)
        lines = vision_engine.detect_fretboard_lines(gray)
        self.assertEqual(len(lines), 0)

    def test_blank_frame_returns_empty(self):
        """A blank frame has no edges and should return an empty list."""
        gray = np.zeros((480, 640), dtype=np.uint8)
        lines = vision_engine.detect_fretboard_lines(gray)
        self.assertEqual(lines, [])


# ---------------------------------------------------------------------------
# Tests – VisionEngine fretboard fallback
# ---------------------------------------------------------------------------

class TestFretboardFallback(unittest.TestCase):
    """Verify the fallback mechanism that reuses previous fretboard lines."""

    @patch("vision_engine._ensure_model", return_value="/fake/model.task")
    @patch("vision_engine.HandLandmarker")
    def _make_engine(self, mock_landmarker_cls, mock_ensure):
        """Create a VisionEngine with mocked MediaPipe."""
        mock_landmarker = MagicMock()
        mock_landmarker.detect.return_value = MagicMock(hand_landmarks=[])
        mock_landmarker_cls.create_from_options.return_value = mock_landmarker
        engine = vision_engine.VisionEngine(model_path="/fake/model.task")
        return engine

    def test_last_fret_lines_initialized_empty(self):
        """_last_fret_lines should start as an empty list."""
        engine = self._make_engine()
        self.assertEqual(engine._last_fret_lines, [])

    @patch("vision_engine.detect_fretboard_lines")
    def test_successful_detection_updates_cache(self, mock_detect):
        """When lines are detected, _last_fret_lines should be updated."""
        engine = self._make_engine()
        expected_lines = [(0.0, 100.0, 640.0, 100.0)]
        mock_detect.return_value = expected_lines

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        engine.process_frame(frame, frame_index=0)

        self.assertEqual(engine._last_fret_lines, expected_lines)

    @patch("vision_engine.detect_fretboard_lines")
    def test_fallback_uses_previous_lines(self, mock_detect):
        """When current detection returns empty, previous lines should be used."""
        engine = self._make_engine()
        cached_lines = [(0.0, 200.0, 640.0, 200.0)]
        engine._last_fret_lines = cached_lines

        # Current frame detects nothing
        mock_detect.return_value = []

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = engine.process_frame(frame, frame_index=1)

        # The cached lines should still be in _last_fret_lines (not cleared)
        self.assertEqual(engine._last_fret_lines, cached_lines)

    @patch("vision_engine.detect_fretboard_lines")
    def test_no_fallback_when_cache_empty(self, mock_detect):
        """When both detection and cache are empty, no lines should be used."""
        engine = self._make_engine()
        mock_detect.return_value = []

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = engine.process_frame(frame, frame_index=0)

        self.assertEqual(engine._last_fret_lines, [])
        # No fretboard detected means fretboard_y should be None
        self.assertIsNone(result.fretboard_y)

    @patch("vision_engine.detect_fretboard_lines")
    def test_new_detection_replaces_cache(self, mock_detect):
        """A new successful detection should replace previously cached lines."""
        engine = self._make_engine()
        old_lines = [(0.0, 100.0, 640.0, 100.0)]
        new_lines = [(0.0, 300.0, 640.0, 300.0)]
        engine._last_fret_lines = old_lines

        mock_detect.return_value = new_lines

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        engine.process_frame(frame, frame_index=2)

        self.assertEqual(engine._last_fret_lines, new_lines)


if __name__ == "__main__":
    unittest.main()
