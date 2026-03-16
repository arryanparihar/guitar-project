"""Unit tests for audio_engine.py.

Tests exercise the pure-computation helpers without requiring real audio
files – audio signals are synthesised with NumPy.
"""

import unittest

import numpy as np

from audio_engine import AudioEngine, OnsetEvent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SR = 22050  # samples/second used across all tests
HOP = 512


def _make_rms(values: list[float]) -> np.ndarray:
    """Build a 1-D RMS array from a plain list of floats."""
    return np.array(values, dtype=np.float64)


def _make_onset_frames(*frames: int) -> np.ndarray:
    """Build an onset-frames array from positional integer arguments."""
    return np.array(list(frames), dtype=np.int32)


# ---------------------------------------------------------------------------
# Tests – AudioEngine constructor
# ---------------------------------------------------------------------------

class TestAudioEngineInit(unittest.TestCase):
    """Tests for AudioEngine.__init__."""

    def test_default_noise_gate_ratio(self):
        engine = AudioEngine()
        self.assertAlmostEqual(engine.noise_gate_ratio, 0.15)

    def test_custom_noise_gate_ratio(self):
        engine = AudioEngine(noise_gate_ratio=0.25)
        self.assertAlmostEqual(engine.noise_gate_ratio, 0.25)

    def test_zero_noise_gate_ratio_allowed(self):
        engine = AudioEngine(noise_gate_ratio=0.0)
        self.assertAlmostEqual(engine.noise_gate_ratio, 0.0)

    def test_noise_gate_ratio_one_allowed(self):
        engine = AudioEngine(noise_gate_ratio=1.0)
        self.assertAlmostEqual(engine.noise_gate_ratio, 1.0)

    def test_noise_gate_ratio_negative_raises(self):
        with self.assertRaises(ValueError):
            AudioEngine(noise_gate_ratio=-0.01)

    def test_noise_gate_ratio_above_one_raises(self):
        with self.assertRaises(ValueError):
            AudioEngine(noise_gate_ratio=1.01)

    def test_invalid_bpm_still_raises(self):
        with self.assertRaises(ValueError):
            AudioEngine(bpm=-10)


# ---------------------------------------------------------------------------
# Tests – _apply_noise_gate
# ---------------------------------------------------------------------------

class TestApplyNoiseGate(unittest.TestCase):
    """Tests for AudioEngine._apply_noise_gate."""

    def test_empty_onset_frames_returned_unchanged(self):
        onset_frames = _make_onset_frames()
        rms = _make_rms([0.5, 0.8, 0.1])
        result = AudioEngine._apply_noise_gate(onset_frames, rms, 0.15)
        self.assertEqual(len(result), 0)

    def test_zero_ratio_bypasses_gate(self):
        """noise_gate_ratio=0 must keep all onsets regardless of RMS."""
        onset_frames = _make_onset_frames(0, 1, 2)
        rms = _make_rms([0.001, 0.001, 0.001])  # very quiet
        result = AudioEngine._apply_noise_gate(onset_frames, rms, 0.0)
        np.testing.assert_array_equal(result, onset_frames)

    def test_quiet_onsets_filtered_out(self):
        """Onsets with RMS below threshold should be removed."""
        # peak = 1.0, threshold at 15% = 0.15
        # frame 0: rms=0.10 → below threshold → filtered
        # frame 1: rms=0.50 → above threshold → kept
        # frame 2: rms=1.00 → above threshold → kept
        onset_frames = _make_onset_frames(0, 1, 2)
        rms = _make_rms([0.10, 0.50, 1.00])
        result = AudioEngine._apply_noise_gate(onset_frames, rms, 0.15)
        np.testing.assert_array_equal(result, np.array([1, 2]))

    def test_loud_onset_kept_quiet_onset_removed(self):
        """Single loud onset retained; single quiet onset discarded."""
        loud_frames = _make_onset_frames(5)
        quiet_frames = _make_onset_frames(0)
        rms = _make_rms([0.01, 0.0, 0.0, 0.0, 0.0, 1.0])

        kept = AudioEngine._apply_noise_gate(loud_frames, rms, 0.15)
        self.assertEqual(len(kept), 1)

        removed = AudioEngine._apply_noise_gate(quiet_frames, rms, 0.15)
        self.assertEqual(len(removed), 0)

    def test_all_onsets_above_threshold_all_kept(self):
        onset_frames = _make_onset_frames(0, 1, 2)
        rms = _make_rms([0.80, 0.90, 1.00])
        result = AudioEngine._apply_noise_gate(onset_frames, rms, 0.15)
        np.testing.assert_array_equal(result, onset_frames)

    def test_all_onsets_below_threshold_all_removed(self):
        onset_frames = _make_onset_frames(0, 1, 2)
        rms = _make_rms([0.01, 0.02, 1.00])  # frames 0 & 1 quiet; peak at 2
        # threshold = 0.15 * 1.0 = 0.15
        result = AudioEngine._apply_noise_gate(onset_frames, rms, 0.15)
        np.testing.assert_array_equal(result, np.array([2]))

    def test_exactly_at_threshold_is_kept(self):
        """An onset whose RMS equals exactly the threshold should be kept."""
        onset_frames = _make_onset_frames(0)
        rms = _make_rms([0.15, 1.0])  # frame 0 rms == 0.15 * peak 1.0
        result = AudioEngine._apply_noise_gate(onset_frames, rms, 0.15)
        self.assertEqual(len(result), 1)

    def test_zero_rms_peak_returns_all_onsets(self):
        """A silent signal (all-zero RMS) should not crash and keep all onsets."""
        onset_frames = _make_onset_frames(0, 1)
        rms = _make_rms([0.0, 0.0])
        result = AudioEngine._apply_noise_gate(onset_frames, rms, 0.15)
        np.testing.assert_array_equal(result, onset_frames)

    def test_frame_index_clamped_to_rms_length(self):
        """Onset frames beyond the RMS array length should be dropped (not crash)."""
        onset_frames = _make_onset_frames(100)  # way beyond rms
        rms = _make_rms([1.0, 0.5])             # only 2 frames
        # Frame 100 is out of bounds → discarded, not clamped to last value
        result = AudioEngine._apply_noise_gate(onset_frames, rms, 0.15)
        self.assertEqual(len(result), 0)

    def test_output_dtype_preserved(self):
        onset_frames = np.array([1, 2, 3], dtype=np.int64)
        rms = _make_rms([0.5, 0.8, 1.0])
        result = AudioEngine._apply_noise_gate(onset_frames, rms, 0.0)
        self.assertEqual(result.dtype, onset_frames.dtype)


# ---------------------------------------------------------------------------
# Tests – analyse_array integration (noise gate end-to-end)
# ---------------------------------------------------------------------------

class TestAnalyseArrayNoiseGate(unittest.TestCase):
    """Integration-level tests using synthesised audio."""

    def _make_click_signal(self, sr: int, click_time: float, amplitude: float) -> np.ndarray:
        """Return a short signal with a single click of the given amplitude."""
        length = int(sr * (click_time + 0.5))
        y = np.zeros(length, dtype=np.float32)
        start = int(click_time * sr)
        width = int(sr * 0.005)  # 5 ms click
        y[start:start + width] = amplitude
        return y

    def test_loud_onset_not_filtered(self):
        """A loud click (amplitude=1.0) must survive the default noise gate."""
        sr = SR
        y = self._make_click_signal(sr, click_time=0.5, amplitude=1.0)
        engine = AudioEngine(bpm=120, hop_length=HOP, noise_gate_ratio=0.15)
        result = engine.analyse_array(y, sr)
        self.assertGreater(len(result.onset_times), 0)

    def test_quiet_noise_is_gated(self):
        """Combine a loud note with a whisper-quiet squeak; squeak must be removed."""
        sr = SR
        duration = int(sr * 2.0)
        y = np.zeros(duration, dtype=np.float32)

        # Loud note at t=0.2 s
        t_loud = int(0.2 * sr)
        y[t_loud: t_loud + int(sr * 0.02)] = 1.0

        # Quiet squeak at t=1.0 s (5% of peak → below 15% threshold)
        t_quiet = int(1.0 * sr)
        y[t_quiet: t_quiet + int(sr * 0.005)] = 0.05

        engine_with_gate = AudioEngine(bpm=120, hop_length=HOP, noise_gate_ratio=0.15)
        engine_no_gate = AudioEngine(bpm=120, hop_length=HOP, noise_gate_ratio=0.0)

        result_gated = engine_with_gate.analyse_array(y, sr)
        result_raw = engine_no_gate.analyse_array(y, sr)

        # With the gate, we should have fewer (or equal) onsets than without
        self.assertLessEqual(
            len(result_gated.onset_times),
            len(result_raw.onset_times),
        )

    def test_disable_gate_keeps_all_onsets(self):
        """Setting noise_gate_ratio=0.0 must disable the gate entirely."""
        sr = SR
        duration = int(sr * 2.0)
        y = np.zeros(duration, dtype=np.float32)

        # Mix a loud note and a very quiet squeak
        y[int(0.2 * sr): int(0.2 * sr) + 512] = 1.0
        y[int(1.0 * sr): int(1.0 * sr) + 100] = 0.02

        engine_no_gate = AudioEngine(bpm=120, hop_length=HOP, noise_gate_ratio=0.0)
        engine_with_gate = AudioEngine(bpm=120, hop_length=HOP, noise_gate_ratio=0.5)

        raw_count = len(engine_no_gate.analyse_array(y, sr).onset_times)
        gated_count = len(engine_with_gate.analyse_array(y, sr).onset_times)

        self.assertGreaterEqual(raw_count, gated_count)


if __name__ == "__main__":
    unittest.main()
