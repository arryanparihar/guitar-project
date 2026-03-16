"""Unit tests for vision_engine pure-computation helpers.

These tests exercise the helper functions without requiring a camera,
GPU, or model file.
"""

import math
import unittest

import cv2
import numpy as np

import vision_engine


# ---------------------------------------------------------------------------
# Tests – detect_fretboard_lines (CLAHE + Hough pipeline)
# ---------------------------------------------------------------------------

class TestDetectFretboardLines(unittest.TestCase):
    """Tests for detect_fretboard_lines."""

    def test_blank_frame_returns_empty(self):
        """A blank (all-black) frame has no edges → no lines returned."""
        blank = np.zeros((480, 640), dtype=np.uint8)
        result = vision_engine.detect_fretboard_lines(blank)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 0)

    def test_detects_bright_horizontal_line(self):
        """A bright horizontal line on a dark background should be detected."""
        frame = np.zeros((480, 640), dtype=np.uint8)
        y = 240
        cv2.line(frame, (0, y), (639, y), 255, 2)
        result = vision_engine.detect_fretboard_lines(frame)
        self.assertGreater(len(result), 0, "Expected at least one line to be detected")
        # All returned lines should be near-horizontal
        for x1, y1, x2, y2 in result:
            angle_deg = abs(math.degrees(math.atan2(y2 - y1, x2 - x1)))
            self.assertTrue(
                angle_deg <= 20 or angle_deg >= 160,
                f"Line angle {angle_deg:.1f}° is not near-horizontal",
            )

    def test_vertical_line_is_filtered(self):
        """A vertical line should not appear in the results (angle filter)."""
        frame = np.zeros((480, 640), dtype=np.uint8)
        cv2.line(frame, (320, 0), (320, 479), 255, 2)
        result = vision_engine.detect_fretboard_lines(frame)
        # If any lines are returned they must all be near-horizontal
        for x1, y1, x2, y2 in result:
            angle_deg = abs(math.degrees(math.atan2(y2 - y1, x2 - x1)))
            self.assertTrue(
                angle_deg <= 20 or angle_deg >= 160,
                f"Vertical artefact leaked through: angle {angle_deg:.1f}°",
            )

    def test_clahe_boosts_low_contrast_line(self):
        """CLAHE should make a low-contrast line detectable in poor lighting.

        In real poor-bedroom-lighting conditions the fretboard is dark and
        noisy.  A dim line (pixel value ~60) on a noisy dark background
        (values 15-30) would not reliably survive Canny thresholding without
        contrast enhancement.  With CLAHE the local contrast is boosted
        enough for the Hough Transform to find the line.
        """
        rng = np.random.default_rng(seed=42)
        # Simulate a dark, noisy frame – realistic "poor lighting" baseline
        frame = rng.integers(15, 30, (480, 640), dtype=np.uint8)
        y = 240
        # Draw a low-contrast line slightly brighter than the background
        cv2.line(frame, (0, y), (639, y), 60, 2)
        result = vision_engine.detect_fretboard_lines(frame)
        self.assertGreater(
            len(result), 0,
            "CLAHE should have boosted contrast enough to detect the dim line",
        )

    def test_returns_list_of_four_tuples(self):
        """Each returned line must be a 4-element tuple of floats."""
        frame = np.zeros((480, 640), dtype=np.uint8)
        cv2.line(frame, (0, 200), (639, 200), 255, 2)
        result = vision_engine.detect_fretboard_lines(frame)
        for item in result:
            self.assertEqual(len(item), 4)
            for coord in item:
                self.assertIsInstance(coord, float)


# ---------------------------------------------------------------------------
# Tests – _compute_efficiency
# ---------------------------------------------------------------------------

class TestComputeEfficiency(unittest.TestCase):

    def test_zero_distance_gives_100(self):
        self.assertEqual(vision_engine._compute_efficiency(0.0), 100.0)

    def test_max_distance_gives_zero(self):
        self.assertEqual(vision_engine._compute_efficiency(120.0), 0.0)

    def test_half_distance_gives_50(self):
        self.assertAlmostEqual(vision_engine._compute_efficiency(60.0), 50.0)

    def test_beyond_max_clamped_to_zero(self):
        self.assertEqual(vision_engine._compute_efficiency(200.0), 0.0)


# ---------------------------------------------------------------------------
# Tests – _point_to_line_distance
# ---------------------------------------------------------------------------

class TestPointToLineDistance(unittest.TestCase):

    def test_point_on_line_gives_zero(self):
        dist = vision_engine._point_to_line_distance(5.0, 0.0, 0.0, 0.0, 10.0, 0.0)
        self.assertAlmostEqual(dist, 0.0)

    def test_perpendicular_distance(self):
        # Point (5, 3) to horizontal line y=0 → distance should be 3
        dist = vision_engine._point_to_line_distance(5.0, 3.0, 0.0, 0.0, 10.0, 0.0)
        self.assertAlmostEqual(dist, 3.0)

    def test_degenerate_line_gives_point_distance(self):
        # Both endpoints same → falls back to point-to-point distance
        dist = vision_engine._point_to_line_distance(3.0, 4.0, 0.0, 0.0, 0.0, 0.0)
        self.assertAlmostEqual(dist, 5.0)


if __name__ == "__main__":
    unittest.main()
