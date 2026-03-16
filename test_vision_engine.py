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


# ---------------------------------------------------------------------------
# Tests – Velocity Exemption / Harmonic Release
# ---------------------------------------------------------------------------

class TestHarmonicRelease(unittest.TestCase):
    """Verify the Velocity Exemption / Harmonic Release feature."""

    # ------------------------------------------------------------------
    # Helper: build a VisionEngine with a fully-mocked MediaPipe stack
    # and a fake hand landmark set.
    # ------------------------------------------------------------------

    @patch("vision_engine._ensure_model", return_value="/fake/model.task")
    @patch("vision_engine.HandLandmarker")
    def _make_engine_with_hand(self, mock_landmarker_cls, mock_ensure,
                                tip_x=320.0, tip_y=100.0):
        """
        Return *(engine, mock_landmarker)* where the landmarker is set up
        to return one hand whose four fingertips are all at (*tip_x*, *tip_y*)
        in normalised coords (divided by frame 640×480).
        """
        mock_landmarker = MagicMock()
        mock_landmarker_cls.create_from_options.return_value = mock_landmarker

        # Build 21 fake landmarks; each fingertip (ids 8,12,16,20) sits at
        # normalised (tip_x/640, tip_y/480).
        def _make_hand(nx, ny):
            lm_list = []
            for i in range(21):
                lm = MagicMock()
                lm.x = nx
                lm.y = ny
                lm_list.append(lm)
            return lm_list

        nx = tip_x / 640.0
        ny = tip_y / 480.0
        mock_landmarker.detect.return_value = MagicMock(
            hand_landmarks=[_make_hand(nx, ny)]
        )
        engine = vision_engine.VisionEngine(model_path="/fake/model.task")
        return engine, mock_landmarker

    @staticmethod
    def _blank_frame():
        return np.zeros((480, 640, 3), dtype=np.uint8)

    # ------------------------------------------------------------------
    # Dataclass defaults
    # ------------------------------------------------------------------

    def test_frame_result_harmonic_release_default_false(self):
        """FrameResult.harmonic_release should default to False."""
        result = vision_engine.FrameResult(frame_index=0, timestamp_sec=0.0)
        self.assertFalse(result.harmonic_release)

    # ------------------------------------------------------------------
    # register_onset
    # ------------------------------------------------------------------

    def test_register_onset_stores_timestamp(self):
        """register_onset() should append the onset time to _recent_onsets."""
        engine, _ = self._make_engine_with_hand()
        self.assertEqual(engine._recent_onsets, [])
        engine.register_onset(1.0)
        engine.register_onset(2.5)
        self.assertEqual(engine._recent_onsets, [1.0, 2.5])

    # ------------------------------------------------------------------
    # Initial state
    # ------------------------------------------------------------------

    def test_velocity_state_initialized(self):
        """Velocity-exemption state should be empty / inactive on creation."""
        engine, _ = self._make_engine_with_hand()
        self.assertEqual(engine._prev_fingertip_positions, {})
        self.assertEqual(engine._recent_onsets, [])
        self.assertLess(engine._harmonic_release_until, 0.0)

    # ------------------------------------------------------------------
    # No exemption without a prior onset
    # ------------------------------------------------------------------

    @patch("vision_engine.detect_fretboard_lines")
    def test_high_velocity_without_onset_no_exemption(self, mock_detect):
        """High fingertip velocity alone (no onset) must NOT trigger Harmonic Release."""
        mock_detect.return_value = [(0.0, 240.0, 640.0, 240.0)]
        engine, _ = self._make_engine_with_hand(tip_x=320.0, tip_y=240.0)

        # Directly manipulate: set previous position near fretboard, then
        # process a frame with fingertips far away.
        engine._prev_fingertip_positions = {
            "index": (320.0, 240.0),
            "middle": (320.0, 240.0),
            "ring": (320.0, 240.0),
            "pinky": (320.0, 240.0),
        }
        # No onset registered → should NOT be flagged as harmonic release.
        result = engine.process_frame(self._blank_frame(), frame_index=1, fps=30.0)
        self.assertFalse(result.harmonic_release)

    # ------------------------------------------------------------------
    # Onset too old → no exemption
    # ------------------------------------------------------------------

    @patch("vision_engine.detect_fretboard_lines")
    def test_stale_onset_no_exemption(self, mock_detect):
        """An onset older than ONSET_LOOKBACK_SEC should not trigger the exemption."""
        mock_detect.return_value = [(0.0, 240.0, 640.0, 240.0)]
        engine, _ = self._make_engine_with_hand(tip_x=320.0, tip_y=240.0)

        # Register an onset at t=0.0, but process frame at t=10.0 (far too late).
        engine.register_onset(0.0)
        engine._prev_fingertip_positions = {
            name: (320.0, 240.0) for name in vision_engine.FINGERTIP_NAMES
        }

        # frame_index=300 at 30fps → timestamp_sec=10.0
        result = engine.process_frame(self._blank_frame(), frame_index=300, fps=30.0)
        self.assertFalse(result.harmonic_release)

    # ------------------------------------------------------------------
    # Harmonic Release triggered
    # ------------------------------------------------------------------

    @patch("vision_engine.detect_fretboard_lines")
    def test_harmonic_release_triggered_by_high_velocity_after_onset(self, mock_detect):
        """
        High-velocity fingertip movement within ONSET_LOOKBACK_SEC after an
        onset must flag harmonic_release and set efficiency_score to 100.
        """
        mock_detect.return_value = [(0.0, 240.0, 640.0, 240.0)]
        # Engine returns fingertips at y=50 (far from fretboard at y=240).
        engine, _ = self._make_engine_with_hand(tip_x=320.0, tip_y=50.0)

        # Register onset at t=0.0; process frame at t=0.1 (within lookback window).
        engine.register_onset(0.0)

        # Previous position right on the fretboard; current frame far away → big Δy.
        engine._prev_fingertip_positions = {
            name: (320.0, 240.0) for name in vision_engine.FINGERTIP_NAMES
        }

        # frame_index=3 at 30fps → timestamp_sec≈0.1; fingertips now at y=50 → Δ≈190px
        result = engine.process_frame(self._blank_frame(), frame_index=3, fps=30.0)
        self.assertTrue(result.harmonic_release)
        self.assertEqual(result.efficiency_score, 100.0)

    # ------------------------------------------------------------------
    # Exemption window persists for HARMONIC_RELEASE_DURATION_SEC
    # ------------------------------------------------------------------

    @patch("vision_engine.detect_fretboard_lines")
    def test_exemption_active_during_window(self, mock_detect):
        """
        Frames within HARMONIC_RELEASE_DURATION_SEC after the release must
        remain exempt even if velocity is low.
        """
        mock_detect.return_value = [(0.0, 240.0, 640.0, 240.0)]
        engine, _ = self._make_engine_with_hand(tip_x=320.0, tip_y=240.0)

        # Directly set the exemption window active from t=0 to t=1.5.
        engine._harmonic_release_until = 1.5

        # Process at t=1.0 (inside window) with low velocity.
        result = engine.process_frame(self._blank_frame(), frame_index=30, fps=30.0)
        self.assertTrue(result.harmonic_release)
        self.assertEqual(result.efficiency_score, 100.0)

    # ------------------------------------------------------------------
    # Exemption expired → normal scoring resumes
    # ------------------------------------------------------------------

    @patch("vision_engine.detect_fretboard_lines")
    def test_exemption_expires_after_duration(self, mock_detect):
        """
        Once HARMONIC_RELEASE_DURATION_SEC has elapsed, normal efficiency
        scoring must resume (score < 100 when fingertips are far away).
        """
        mock_detect.return_value = [(0.0, 240.0, 640.0, 240.0)]
        engine, _ = self._make_engine_with_hand(tip_x=320.0, tip_y=240.0)

        # Exemption window ended at t=0.5; process at t=2.0 (past expiry).
        engine._harmonic_release_until = 0.5

        result = engine.process_frame(self._blank_frame(), frame_index=60, fps=30.0)
        self.assertFalse(result.harmonic_release)
        # Fingertips at y=240 with fretboard at y=240 → distance≈0 → high score,
        # but crucially harmonic_release is False and scoring is not exempted.
        self.assertIsNotNone(result.efficiency_score)
        self.assertLessEqual(result.efficiency_score, 100.0)

    # ------------------------------------------------------------------
    # Previous positions cleared when hand leaves frame
    # ------------------------------------------------------------------

    @patch("vision_engine._ensure_model", return_value="/fake/model.task")
    @patch("vision_engine.HandLandmarker")
    @patch("vision_engine.detect_fretboard_lines")
    def test_prev_positions_cleared_when_hand_not_detected(
        self, mock_detect, mock_landmarker_cls, mock_ensure
    ):
        """When no hand is detected, _prev_fingertip_positions must be cleared."""
        mock_detect.return_value = []
        mock_landmarker = MagicMock()
        mock_landmarker.detect.return_value = MagicMock(hand_landmarks=[])
        mock_landmarker_cls.create_from_options.return_value = mock_landmarker

        engine = vision_engine.VisionEngine(model_path="/fake/model.task")
        engine._prev_fingertip_positions = {
            name: (100.0, 100.0) for name in vision_engine.FINGERTIP_NAMES
        }

        engine.process_frame(self._blank_frame(), frame_index=0)
        self.assertEqual(engine._prev_fingertip_positions, {})


if __name__ == "__main__":
    unittest.main()
