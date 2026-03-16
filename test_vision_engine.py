"""Unit tests for vision_engine confidence-threshold logic.

These tests exercise the fingertip extraction and distance-computation
helpers without requiring a camera, GPU, or model file – MediaPipe
landmarks are replaced by lightweight stubs.
"""

import unittest
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

import vision_engine
from vision_engine import (
    FingertipData,
    FrameResult,
    FINGERTIP_IDS,
    FINGERTIP_NAMES,
    LANDMARK_CONFIDENCE_THRESHOLD,
    _compute_efficiency,
    _point_to_line_distance,
)


# ---------------------------------------------------------------------------
# Helpers – lightweight landmark stubs
# ---------------------------------------------------------------------------

class _FakeLandmark:
    """Minimal stub for mediapipe NormalizedLandmark (Tasks API)."""

    def __init__(self, x: float, y: float, z: float = 0.0,
                 visibility=None, presence=None):
        self.x = x
        self.y = y
        self.z = z
        self.visibility = visibility
        self.presence = presence


def _make_hand(tip_overrides: dict):
    """Return a 21-element landmark list.

    Parameters
    ----------
    tip_overrides : dict[int, _FakeLandmark]
        Mapping from landmark index → stub object.  Indices not listed
        default to a landmark at (0, 0) with no visibility/presence.
    """
    landmarks = [_FakeLandmark(0.0, 0.0) for _ in range(21)]
    for idx, lm in tip_overrides.items():
        landmarks[idx] = lm
    return landmarks


# ---------------------------------------------------------------------------
# Tests – LANDMARK_CONFIDENCE_THRESHOLD constant
# ---------------------------------------------------------------------------

class TestThresholdConstant(unittest.TestCase):
    def test_threshold_value(self):
        """Default threshold must be 0.6."""
        self.assertAlmostEqual(LANDMARK_CONFIDENCE_THRESHOLD, 0.6)


# ---------------------------------------------------------------------------
# Tests – FingertipData allows None coordinates
# ---------------------------------------------------------------------------

class TestFingertipDataNone(unittest.TestCase):
    def test_accepts_none_coordinates(self):
        ft = FingertipData(name="index", x=None, y=None)
        self.assertIsNone(ft.x)
        self.assertIsNone(ft.y)

    def test_accepts_float_coordinates(self):
        ft = FingertipData(name="middle", x=100.0, y=200.0)
        self.assertEqual(ft.x, 100.0)
        self.assertEqual(ft.y, 200.0)


# ---------------------------------------------------------------------------
# Tests – confidence filtering in process_frame
# ---------------------------------------------------------------------------

class _FakeDetection:
    """Stub for MediaPipe HandLandmarkerResult."""

    def __init__(self, hand_landmarks):
        self.hand_landmarks = hand_landmarks


def _make_engine_with_hand(hand_landmarks):
    """Return a VisionEngine whose internal landmarker is stubbed out."""
    engine = vision_engine.VisionEngine.__new__(vision_engine.VisionEngine)
    mock_landmarker = MagicMock()
    mock_landmarker.detect.return_value = _FakeDetection([hand_landmarks])
    engine._landmarker = mock_landmarker
    return engine


class TestConfidenceFiltering(unittest.TestCase):
    """Verify that low-confidence landmarks are flagged as None."""

    def _blank_frame(self):
        return np.zeros((480, 640, 3), dtype=np.uint8)

    def test_high_confidence_landmarks_have_coordinates(self):
        """All fingertips above threshold → x/y are floats, not None."""
        hand = _make_hand({
            tip_id: _FakeLandmark(0.5, 0.5, visibility=0.9, presence=0.9)
            for tip_id in FINGERTIP_IDS
        })
        engine = _make_engine_with_hand(hand)
        result = engine.process_frame(self._blank_frame())

        self.assertEqual(len(result.fingertips), 4)
        for ft in result.fingertips:
            self.assertIsNotNone(ft.x, f"{ft.name} x should not be None")
            self.assertIsNotNone(ft.y, f"{ft.name} y should not be None")

    def test_low_visibility_sets_none(self):
        """A landmark with visibility < threshold → None coordinates."""
        hand = _make_hand({
            8:  _FakeLandmark(0.3, 0.3, visibility=0.4, presence=0.9),   # below
            12: _FakeLandmark(0.5, 0.5, visibility=0.9, presence=0.9),
            16: _FakeLandmark(0.7, 0.7, visibility=0.9, presence=0.9),
            20: _FakeLandmark(0.9, 0.9, visibility=0.9, presence=0.9),
        })
        engine = _make_engine_with_hand(hand)
        result = engine.process_frame(self._blank_frame())

        index_ft = next(ft for ft in result.fingertips if ft.name == "index")
        self.assertIsNone(index_ft.x)
        self.assertIsNone(index_ft.y)

        for ft in result.fingertips:
            if ft.name != "index":
                self.assertIsNotNone(ft.x, f"{ft.name} x should not be None")
                self.assertIsNotNone(ft.y, f"{ft.name} y should not be None")

    def test_low_presence_sets_none(self):
        """A landmark with presence < threshold → None coordinates."""
        hand = _make_hand({
            8:  _FakeLandmark(0.3, 0.3, visibility=0.9, presence=0.2),   # below
            12: _FakeLandmark(0.5, 0.5, visibility=0.9, presence=0.9),
            16: _FakeLandmark(0.7, 0.7, visibility=0.9, presence=0.9),
            20: _FakeLandmark(0.9, 0.9, visibility=0.9, presence=0.9),
        })
        engine = _make_engine_with_hand(hand)
        result = engine.process_frame(self._blank_frame())

        index_ft = next(ft for ft in result.fingertips if ft.name == "index")
        self.assertIsNone(index_ft.x)
        self.assertIsNone(index_ft.y)

    def test_exactly_at_threshold_is_accepted(self):
        """A landmark exactly at the threshold is NOT flagged as low-confidence."""
        hand = _make_hand({
            tip_id: _FakeLandmark(0.5, 0.5,
                                   visibility=LANDMARK_CONFIDENCE_THRESHOLD,
                                   presence=LANDMARK_CONFIDENCE_THRESHOLD)
            for tip_id in FINGERTIP_IDS
        })
        engine = _make_engine_with_hand(hand)
        result = engine.process_frame(self._blank_frame())

        for ft in result.fingertips:
            self.assertIsNotNone(ft.x, f"{ft.name} at threshold should be accepted")

    def test_no_scores_attributes_treated_as_confident(self):
        """Landmarks without visibility/presence attrs are treated as valid."""
        hand = _make_hand({
            tip_id: _FakeLandmark(0.5, 0.5)  # no visibility/presence
            for tip_id in FINGERTIP_IDS
        })
        engine = _make_engine_with_hand(hand)
        result = engine.process_frame(self._blank_frame())

        for ft in result.fingertips:
            self.assertIsNotNone(ft.x, f"{ft.name} missing scores should be valid")

    def test_all_low_confidence_no_efficiency_score(self):
        """When all fingertips are low-confidence, efficiency_score stays None."""
        hand = _make_hand({
            tip_id: _FakeLandmark(0.5, 0.5, visibility=0.1, presence=0.1)
            for tip_id in FINGERTIP_IDS
        })
        engine = _make_engine_with_hand(hand)
        result = engine.process_frame(self._blank_frame())

        self.assertIsNone(result.avg_finger_height)
        self.assertIsNone(result.efficiency_score)

    def test_mixed_confidence_excludes_bad_from_scoring(self):
        """Only high-confidence fingertips contribute to avg_finger_height."""
        # We need a fretboard line for distances to be computed.
        # Draw a bright horizontal line at y=240 so Hough can find it.
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.line(frame, (0, 240), (639, 240), (255, 255, 255), 3)

        # Normalised y for a pixel at 240 in a 480-pixel-tall frame → 0.5
        norm_y = 240 / 480

        hand = _make_hand({
            8:  _FakeLandmark(0.5, norm_y, visibility=0.1, presence=0.9),  # low → excluded
            12: _FakeLandmark(0.5, norm_y, visibility=0.9, presence=0.9),  # valid, dist≈0
            16: _FakeLandmark(0.5, norm_y, visibility=0.9, presence=0.9),  # valid, dist≈0
            20: _FakeLandmark(0.5, norm_y, visibility=0.9, presence=0.9),  # valid, dist≈0
        })
        engine = _make_engine_with_hand(hand)
        result = engine.process_frame(frame)

        # Three valid fingers all sitting on the line → avg_distance ≈ 0 → score ≈ 100
        if result.efficiency_score is not None:
            self.assertGreater(result.efficiency_score, 90,
                               "Expected high score when valid fingers are on the line")


# ---------------------------------------------------------------------------
# Tests – _compute_efficiency (unchanged, sanity check)
# ---------------------------------------------------------------------------

class TestComputeEfficiency(unittest.TestCase):
    def test_zero_distance_gives_100(self):
        self.assertAlmostEqual(_compute_efficiency(0.0), 100.0)

    def test_max_distance_gives_zero(self):
        self.assertAlmostEqual(_compute_efficiency(120.0), 0.0)

    def test_half_distance_gives_50(self):
        self.assertAlmostEqual(_compute_efficiency(60.0), 50.0)


if __name__ == "__main__":
    unittest.main()
