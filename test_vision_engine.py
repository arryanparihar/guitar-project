"""Unit tests for vision_engine geometry helpers.

These tests exercise pure-computation functions without requiring a camera,
GPU, or MediaPipe model file.
"""

import math
import unittest

from vision_engine import (
    FingertipData,
    _compute_efficiency,
    _point_to_line_distance,
)


# ---------------------------------------------------------------------------
# Tests – _point_to_line_distance (3-D)
# ---------------------------------------------------------------------------

class TestPointToLineDistance(unittest.TestCase):
    """Tests for _point_to_line_distance with the z-depth parameter."""

    def test_zero_z_matches_2d(self):
        """pz=0 must give the same result as the pure-2D perpendicular distance."""
        dist = _point_to_line_distance(0, 100, 0, 200, 640, 200, pz=0.0)
        self.assertAlmostEqual(dist, 100.0, places=6)

    def test_nonzero_z_increases_distance(self):
        """Adding depth must increase the distance beyond the 2D value."""
        dist_2d = _point_to_line_distance(0, 100, 0, 200, 640, 200, pz=0.0)
        dist_3d = _point_to_line_distance(0, 100, 0, 200, 640, 200, pz=50.0)
        self.assertGreater(dist_3d, dist_2d)

    def test_3d_pythagorean(self):
        """When 2D perp distance = 3 and pz = 4, the 3D distance should be 5."""
        # Horizontal line at y=0; point at (0, 3) with pz=4 → 2D dist=3
        dist = _point_to_line_distance(0.0, 3.0, 0.0, 0.0, 100.0, 0.0, pz=4.0)
        self.assertAlmostEqual(dist, 5.0, places=6)

    def test_point_on_line_z_only(self):
        """When the point projects exactly onto the line, distance equals abs(pz)."""
        dist = _point_to_line_distance(50.0, 0.0, 0.0, 0.0, 100.0, 0.0, pz=30.0)
        self.assertAlmostEqual(dist, 30.0, places=6)

    def test_degenerate_line_with_z(self):
        """A zero-length line (point) should fall back to 3D distance to that point."""
        dist = _point_to_line_distance(3.0, 4.0, 0.0, 0.0, 0.0, 0.0, pz=0.0)
        self.assertAlmostEqual(dist, 5.0, places=6)

        dist_3d = _point_to_line_distance(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, pz=7.0)
        self.assertAlmostEqual(dist_3d, 7.0, places=6)

    def test_negative_z_same_as_positive(self):
        """Depth behind the fretboard plane should give the same distance as in front."""
        dist_pos = _point_to_line_distance(0, 3, 0, 0, 100, 0, pz=4.0)
        dist_neg = _point_to_line_distance(0, 3, 0, 0, 100, 0, pz=-4.0)
        self.assertAlmostEqual(dist_pos, dist_neg, places=6)

    def test_default_pz_is_zero(self):
        """Calling without pz should behave as pz=0 (backward compatible)."""
        dist_explicit = _point_to_line_distance(0, 3, 0, 0, 100, 0, pz=0.0)
        dist_default = _point_to_line_distance(0, 3, 0, 0, 100, 0)
        self.assertAlmostEqual(dist_explicit, dist_default, places=10)


# ---------------------------------------------------------------------------
# Tests – FingertipData (z field)
# ---------------------------------------------------------------------------

class TestFingertipData(unittest.TestCase):
    """Tests for the FingertipData dataclass z field."""

    def test_z_defaults_to_zero(self):
        """FingertipData created without z should default to 0.0."""
        ft = FingertipData(name="index", x=100.0, y=200.0)
        self.assertEqual(ft.z, 0.0)

    def test_z_stores_value(self):
        """FingertipData should store an explicit z value."""
        ft = FingertipData(name="middle", x=50.0, y=75.0, z=-12.5)
        self.assertAlmostEqual(ft.z, -12.5)


# ---------------------------------------------------------------------------
# Tests – _compute_efficiency (unchanged, regression guard)
# ---------------------------------------------------------------------------

class TestComputeEfficiency(unittest.TestCase):
    """Regression tests for _compute_efficiency to confirm it still works
    correctly with 3-D distances supplied by the updated pipeline."""

    def test_zero_distance_gives_100(self):
        self.assertAlmostEqual(_compute_efficiency(0.0), 100.0)

    def test_max_distance_gives_zero(self):
        self.assertAlmostEqual(_compute_efficiency(120.0), 0.0)

    def test_half_max_gives_50(self):
        self.assertAlmostEqual(_compute_efficiency(60.0), 50.0)

    def test_beyond_max_clamped_to_zero(self):
        self.assertAlmostEqual(_compute_efficiency(200.0), 0.0)

    def test_3d_dist_larger_than_2d_lowers_score(self):
        """A 3D distance (larger) should produce a lower score than its 2D counterpart."""
        score_2d = _compute_efficiency(30.0)
        score_3d = _compute_efficiency(math.sqrt(30.0 ** 2 + 20.0 ** 2))
        self.assertGreater(score_2d, score_3d)


if __name__ == "__main__":
    unittest.main()
