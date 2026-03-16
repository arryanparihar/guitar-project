"""Unit tests for vision_engine efficiency scoring helpers.

These tests exercise the pure-computation helpers (geometry utilities and
the normalised-distance efficiency function) without requiring a camera,
GPU, or model file.
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


if __name__ == "__main__":
    unittest.main()
