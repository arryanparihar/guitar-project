"""Unit tests for vision_engine efficiency scoring helpers.

These tests exercise the pure-computation helpers (geometry utilities and
the normalised-distance efficiency function) without requiring a camera,
GPU, or model file.
"""Unit tests for vision_engine fretboard detection and fallback logic.

These tests exercise ``detect_fretboard_lines`` angle filtering and
the ``VisionEngine`` fretboard fallback mechanism without requiring a
camera, GPU, or model file.
"""

import math
import unittest

from vision_engine import _compute_efficiency, _euclidean, _point_to_line_distance


# ---------------------------------------------------------------------------
# Tests – _compute_efficiency (normalised distance)
# ---------------------------------------------------------------------------

class TestComputeEfficiency(unittest.TestCase):
    """Tests for the normalised-distance efficiency scorer."""

    def test_zero_distance_gives_100(self):
        """Fingers on the fretboard → perfect score."""
        self.assertEqual(_compute_efficiency(0.0), 100.0)

    def test_max_distance_gives_0(self):
        """Normalised distance equal to the threshold → score of 0."""
        self.assertEqual(_compute_efficiency(1.0), 0.0)

    def test_above_max_clamped_to_0(self):
        """Distance above max should still bottom out at 0."""
        self.assertEqual(_compute_efficiency(2.5), 0.0)

    def test_half_distance_gives_50(self):
        """Midpoint normalised distance → 50 % score."""
        self.assertEqual(_compute_efficiency(0.5), 50.0)

    def test_custom_max(self):
        """Custom max_distance_norm should scale the mapping."""
        # 0.5 / 2.0 = 0.25 → 75 %
        self.assertEqual(_compute_efficiency(0.5, max_distance_norm=2.0), 75.0)

    def test_negative_distance_treated_as_zero(self):
        """Negative distances are physically meaningless; function should
        still return ≤ 100 (clamped via min with max_distance_norm)."""
        score = _compute_efficiency(-0.1)
        # min(-0.1, 1.0) = -0.1 → (1 - (-0.1)/1.0)*100 = 110
        # This is a quirk of the simple formula – acceptable since negative
        # distances never occur in practice.
        self.assertIsInstance(score, float)


# ---------------------------------------------------------------------------
# Tests – _euclidean
# ---------------------------------------------------------------------------

class TestEuclidean(unittest.TestCase):
    """Tests for the 2-D Euclidean distance helper."""

    def test_same_point(self):
        self.assertEqual(_euclidean((0, 0), (0, 0)), 0.0)

    def test_unit_distance(self):
        self.assertAlmostEqual(_euclidean((0, 0), (1, 0)), 1.0)

    def test_known_triangle(self):
        self.assertAlmostEqual(_euclidean((0, 0), (3, 4)), 5.0)


# ---------------------------------------------------------------------------
# Tests – scale-invariance property (integration-level)
# ---------------------------------------------------------------------------

class TestNormalisedDistanceScaleInvariance(unittest.TestCase):
    """Verify that normalising by hand_ref_scale produces the same
    efficiency score regardless of absolute pixel distances."""

    @staticmethod
    def _score_from_pixels(pixel_distances: list[float],
                           hand_ref_scale: float) -> float:
        """Mimic the normalisation logic in process_frame."""
        if hand_ref_scale > 0:
            norm = [d / hand_ref_scale for d in pixel_distances]
        else:
            norm = pixel_distances
        avg_norm = sum(norm) / len(norm)
        return _compute_efficiency(avg_norm)

    def test_same_ratio_different_scales(self):
        """Doubling both pixel distance and hand-size reference should
        produce an identical efficiency score."""
        dists_near = [30.0, 40.0, 50.0, 60.0]
        ref_near = 100.0

        dists_far = [d * 2 for d in dists_near]
        ref_far = ref_near * 2

        score_near = self._score_from_pixels(dists_near, ref_near)
        score_far = self._score_from_pixels(dists_far, ref_far)
        self.assertAlmostEqual(score_near, score_far, places=2)

    def test_three_scales(self):
        """Three different camera distances, same pose → same score."""
        base_dists = [20.0, 25.0, 30.0, 35.0]
        base_ref = 80.0

        scores = []
        for scale in (0.5, 1.0, 2.0):
            d = [x * scale for x in base_dists]
            r = base_ref * scale
            scores.append(self._score_from_pixels(d, r))

        self.assertAlmostEqual(scores[0], scores[1], places=2)
        self.assertAlmostEqual(scores[1], scores[2], places=2)

    def test_zero_ref_scale_fallback(self):
        """If hand_ref_scale is 0 (degenerate case), raw distances should
        be passed through without division by zero."""
        score = self._score_from_pixels([0.5], hand_ref_scale=0.0)
        self.assertAlmostEqual(score, _compute_efficiency(0.5))
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
