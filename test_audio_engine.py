"""Unit tests for audio_engine.py.

These tests exercise the harmonic onset detection pipeline and other
pure-computation helpers without requiring real audio files on disk.
"""

import unittest

import numpy as np

from audio_engine import AudioEngine, AudioAnalysisResult, OnsetEvent


# ---------------------------------------------------------------------------
# Helpers – synthetic audio signals
# ---------------------------------------------------------------------------

SR = 22050  # sample rate used throughout tests
HOP = 512   # hop length


def _sine(freq_hz: float, duration_sec: float, sr: int = SR, amp: float = 0.5) -> np.ndarray:
    """Return a pure sine-wave as a float32 array."""
    t = np.linspace(0.0, duration_sec, int(sr * duration_sec), endpoint=False)
    return (amp * np.sin(2.0 * np.pi * freq_hz * t)).astype(np.float32)


def _click_train(
    bpm: float,
    duration_sec: float,
    sr: int = SR,
    freq_hz: float = 3000.0,
    click_dur_sec: float = 0.02,
) -> np.ndarray:
    """
    Synthesize a sequence of short sine bursts at *bpm* that mimic the
    high-frequency 'bloom' of guitar harmonics.

    Each click is a brief burst of *freq_hz* separated by silence.
    """
    n_samples = int(sr * duration_sec)
    y = np.zeros(n_samples, dtype=np.float32)
    period_samples = int(sr * 60.0 / bpm)
    click_len = int(sr * click_dur_sec)
    t_click = np.linspace(0.0, click_dur_sec, click_len, endpoint=False)
    burst = (0.8 * np.sin(2.0 * np.pi * freq_hz * t_click)).astype(np.float32)

    pos = 0
    while pos + click_len <= n_samples:
        y[pos: pos + click_len] = burst
        pos += period_samples

    return y


# ---------------------------------------------------------------------------
# Tests – _harmonic_onset_strength
# ---------------------------------------------------------------------------

class TestHarmonicOnsetStrength(unittest.TestCase):
    """Verify properties of the combined onset strength envelope."""

    def setUp(self):
        self.engine = AudioEngine(bpm=120.0, hop_length=HOP)

    def test_returns_1d_array(self):
        """Output must be a 1-D numpy array."""
        y = _sine(440.0, 1.0)
        env = self.engine._harmonic_onset_strength(y, SR)
        self.assertEqual(env.ndim, 1)
        self.assertGreater(len(env), 0)

    def test_values_in_zero_one_range(self):
        """All values should lie in [0, 1] because both components are normalised."""
        y = _sine(3000.0, 2.0)
        env = self.engine._harmonic_onset_strength(y, SR)
        self.assertGreaterEqual(float(env.min()), 0.0)
        self.assertLessEqual(float(env.max()), 1.0 + 1e-6)  # allow tiny float error

    def test_length_consistent_with_rms(self):
        """Envelope length should match the number of RMS frames."""
        import librosa
        y = _sine(440.0, 3.0)
        env = self.engine._harmonic_onset_strength(y, SR)
        rms = librosa.feature.rms(y=y, hop_length=HOP)[0]
        # Lengths may differ by at most 1 due to internal truncation.
        self.assertAlmostEqual(len(env), len(rms), delta=1)

    def test_silent_signal_does_not_raise(self):
        """An all-zero (silent) signal should not raise an exception."""
        y = np.zeros(SR, dtype=np.float32)
        env = self.engine._harmonic_onset_strength(y, SR)
        self.assertIsInstance(env, np.ndarray)

    def test_high_frequency_burst_produces_elevated_envelope(self):
        """
        A burst at 3 kHz (well above the 2 kHz boundary) should raise the
        combined envelope above the floor established by a silent signal.
        """
        y_silent = np.zeros(SR * 2, dtype=np.float32)
        y_burst = _sine(3000.0, 2.0, amp=0.9)

        env_silent = self.engine._harmonic_onset_strength(y_silent, SR)
        env_burst = self.engine._harmonic_onset_strength(y_burst, SR)

        self.assertGreater(float(env_burst.mean()), float(env_silent.mean()))


# ---------------------------------------------------------------------------
# Tests – analyse_array (integration)
# ---------------------------------------------------------------------------

class TestAnalyseArray(unittest.TestCase):
    """End-to-end tests via analyse_array using synthetic audio."""

    def test_returns_analysis_result(self):
        """analyse_array should return an AudioAnalysisResult."""
        engine = AudioEngine(bpm=120.0, hop_length=HOP)
        y = _sine(440.0, 2.0)
        result = engine.analyse_array(y, SR)
        self.assertIsInstance(result, AudioAnalysisResult)

    def test_result_fields_populated(self):
        """All scalar fields of the result should be non-negative."""
        engine = AudioEngine(bpm=120.0, hop_length=HOP)
        y = _sine(440.0, 2.0)
        result = engine.analyse_array(y, SR)

        self.assertEqual(result.sample_rate, SR)
        self.assertAlmostEqual(result.duration_sec, 2.0, delta=0.05)
        self.assertEqual(result.bpm_target, 120.0)
        self.assertGreaterEqual(result.avg_deviation_ms, 0.0)
        self.assertGreaterEqual(result.timing_score, 0.0)
        self.assertLessEqual(result.timing_score, 100.0)

    def test_rms_envelope_normalised(self):
        """rms_envelope values should all lie in [0, 1]."""
        engine = AudioEngine(bpm=120.0, hop_length=HOP)
        y = _sine(440.0, 2.0)
        result = engine.analyse_array(y, SR)

        for v in result.rms_envelope:
            self.assertGreaterEqual(v, 0.0)
            self.assertLessEqual(v, 1.0 + 1e-6)

    def test_harmonic_click_train_detects_onsets(self):
        """
        A click train at 3 kHz (harmonic region) at 120 BPM should produce
        detected onsets – demonstrating that the HF-aware pipeline picks up
        the harmonic 'blooms' the standard onset detector would miss.
        """
        bpm = 120.0
        engine = AudioEngine(bpm=bpm, hop_length=HOP)
        y = _click_train(bpm=bpm, duration_sec=4.0, freq_hz=3000.0)
        result = engine.analyse_array(y, SR)

        # At 120 BPM over 4 s we expect ~8 clicks; at least 2 should be detected.
        self.assertGreaterEqual(len(result.onset_times), 2,
                                "Expected at least 2 harmonic onsets to be detected")

    def test_onset_events_aligned_to_bpm_grid(self):
        """Each OnsetEvent should reference the correct beat index."""
        engine = AudioEngine(bpm=120.0, hop_length=HOP)
        y = _click_train(bpm=120.0, duration_sec=4.0, freq_hz=3000.0)
        result = engine.analyse_array(y, SR)

        for i, ev in enumerate(result.onset_events):
            self.assertEqual(ev.index, i)
            self.assertIsInstance(ev.deviation_ms, float)

    def test_silent_audio_returns_no_onsets(self):
        """Silent audio should produce zero detected onsets."""
        engine = AudioEngine(bpm=120.0, hop_length=HOP)
        y = np.zeros(SR * 2, dtype=np.float32)
        result = engine.analyse_array(y, SR)
        self.assertEqual(len(result.onset_times), 0)
        self.assertEqual(result.avg_deviation_ms, 0.0)
        self.assertEqual(result.timing_score, 100.0)


# ---------------------------------------------------------------------------
# Tests – _timing_score
# ---------------------------------------------------------------------------

class TestTimingScore(unittest.TestCase):
    """Verify the timing score mapping."""

    def test_zero_deviation_is_perfect(self):
        self.assertEqual(AudioEngine._timing_score(0.0), 100.0)

    def test_max_deviation_is_zero(self):
        self.assertEqual(AudioEngine._timing_score(200.0), 0.0)

    def test_half_deviation_is_fifty(self):
        self.assertEqual(AudioEngine._timing_score(100.0), 50.0)

    def test_beyond_max_clamps_to_zero(self):
        self.assertEqual(AudioEngine._timing_score(999.0), 0.0)


# ---------------------------------------------------------------------------
# Tests – _align_onsets
# ---------------------------------------------------------------------------

class TestAlignOnsets(unittest.TestCase):
    """Verify onset-to-beat alignment logic."""

    def test_empty_list_returns_empty(self):
        self.assertEqual(AudioEngine._align_onsets([], 0.5), [])

    def test_single_onset_has_zero_deviation(self):
        events = AudioEngine._align_onsets([1.0], 0.5)
        self.assertEqual(len(events), 1)
        self.assertAlmostEqual(events[0].deviation_ms, 0.0)

    def test_perfect_grid_has_zero_deviations(self):
        """Onsets exactly on the beat grid should all have ~0 ms deviation."""
        beat_period = 0.5  # 120 BPM
        onsets = [i * beat_period for i in range(8)]
        events = AudioEngine._align_onsets(onsets, beat_period)
        for ev in events:
            self.assertAlmostEqual(ev.deviation_ms, 0.0, places=6)

    def test_late_onset_has_positive_deviation(self):
        """An onset that arrives after the beat should have + deviation."""
        beat_period = 0.5
        onsets = [0.0, 0.55]  # second onset is 50 ms late
        events = AudioEngine._align_onsets(onsets, beat_period)
        self.assertGreater(events[1].deviation_ms, 0.0)


if __name__ == "__main__":
    unittest.main()
