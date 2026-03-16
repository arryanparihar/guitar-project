"""Unit tests for audio_engine.py.

These tests exercise the harmonic onset detection pipeline and other
pure-computation helpers without requiring real audio files on disk.
"""Unit tests for AudioEngine._align_onsets (improved anchor selection).

These tests validate the updated onset-alignment logic without requiring
a real audio file – audio arrays are synthesised in-memory.
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

from audio_engine import AudioEngine, OnsetEvent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SR = 22050     # sample rate used across all tests
HOP = 512      # hop length
BEAT = 0.5     # beat period (120 BPM)


def _silent(duration_sec: float = 4.0) -> np.ndarray:
    """Return a silent (all-zeros) audio buffer."""
    return np.zeros(int(SR * duration_sec), dtype=np.float32)


def _audio_with_spikes(
    spikes: list[tuple[float, float]],
    duration_sec: float = 4.0,
) -> np.ndarray:
    """
    Return audio with rectangular impulses at specified times.

    Parameters
    ----------
    spikes : list of (time_sec, amplitude) pairs
    """
    y = np.zeros(int(SR * duration_sec), dtype=np.float32)
    for t, amp in spikes:
        center = int(t * SR)
        start = max(0, center - HOP)
        end = min(len(y), center + HOP)
        y[start:end] = amp
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
# Tests – _align_onsets
# ---------------------------------------------------------------------------

class TestAlignOnsetsEmpty(unittest.TestCase):
    """Edge case: empty onset list."""

    def test_returns_empty_list(self):
        y = _silent()
        result = AudioEngine._align_onsets([], BEAT, y, SR, HOP)
        self.assertEqual(result, [])


class TestAlignOnsetsAnchorSelection(unittest.TestCase):
    """Tests verifying that the loudest onset (not the first) is the anchor."""

    def test_anchor_is_loudest_not_first(self):
        """
        First onset (0.3 s, quiet) must NOT be anchor.
        Second onset (0.8 s, loud) MUST be anchor → its deviation is 0 ms.
        """
        y = _audio_with_spikes([(0.3, 0.05), (0.8, 1.0)])
        onset_times = [0.3, 0.8, 1.3, 1.8]

        events = AudioEngine._align_onsets(
            onset_times, BEAT, y, SR, HOP,
            lead_in_ignore_sec=0.1, anchor_window_sec=2.0,
        )

        anchor_ev = next(e for e in events if abs(e.time_sec - 0.8) < 1e-6)
        self.assertAlmostEqual(anchor_ev.deviation_ms, 0.0, places=3)

        # The quiet first onset is NOT the anchor, so it WILL have a non-zero
        # deviation (0.3 − 0.8 = −0.5 s, which is −1 beat → expected = 0.3).
        first_ev = events[0]
        self.assertAlmostEqual(first_ev.expected_sec, 0.3, places=3)

    def test_anchor_deviation_is_always_zero(self):
        """The chosen anchor onset always has exactly 0 deviation."""
        y = _audio_with_spikes([(0.2, 0.1), (0.7, 0.9), (1.2, 0.5)])
        onset_times = [0.2, 0.7, 1.2]

        events = AudioEngine._align_onsets(
            onset_times, BEAT, y, SR, HOP,
            lead_in_ignore_sec=0.1,
        )

        # Loudest is 0.7 s → it becomes the anchor
        anchor_ev = next(e for e in events if abs(e.time_sec - 0.7) < 1e-6)
        self.assertAlmostEqual(anchor_ev.deviation_ms, 0.0, places=3)

    def test_single_onset_is_its_own_anchor(self):
        """A single onset must always have zero deviation."""
        y = _audio_with_spikes([(1.0, 1.0)])
        events = AudioEngine._align_onsets([1.0], BEAT, y, SR, HOP)
        self.assertEqual(len(events), 1)
        self.assertAlmostEqual(events[0].deviation_ms, 0.0, places=3)


class TestAlignOnsetsLeadIn(unittest.TestCase):
    """Tests verifying that lead_in_ignore_sec filters pre-playing noise."""

    def test_onset_before_lead_in_not_selected_as_anchor(self):
        """
        A very loud onset at 0.05 s (before lead_in_ignore_sec=0.1) must
        not be chosen as anchor; the quieter onset at 0.5 s should be.
        """
        y = _audio_with_spikes([(0.05, 2.0), (0.5, 1.0)])
        onset_times = [0.05, 0.5, 1.0, 1.5]

        events = AudioEngine._align_onsets(
            onset_times, BEAT, y, SR, HOP,
            lead_in_ignore_sec=0.1, anchor_window_sec=2.0,
        )

        # Anchor should be 0.5 s (loudest after lead_in) → deviation == 0
        anchor_ev = next(e for e in events if abs(e.time_sec - 0.5) < 1e-6)
        self.assertAlmostEqual(anchor_ev.deviation_ms, 0.0, places=3)

        # The pre-playing onset at 0.05 s must NOT have zero deviation
        # (it is not the anchor, so expected != 0.05)
        pre_ev = events[0]
        self.assertNotAlmostEqual(pre_ev.expected_sec, 0.05, places=3)

    def test_lead_in_zero_allows_all_onsets_as_candidates(self):
        """lead_in_ignore_sec=0 means even the first onset can be anchor."""
        y = _audio_with_spikes([(0.01, 1.0), (0.5, 0.1)])
        onset_times = [0.01, 0.5]

        events = AudioEngine._align_onsets(
            onset_times, BEAT, y, SR, HOP,
            lead_in_ignore_sec=0.0, anchor_window_sec=2.0,
        )

        # 0.01 s onset is loudest and eligible → anchor → deviation == 0
        anchor_ev = events[0]
        self.assertAlmostEqual(anchor_ev.deviation_ms, 0.0, places=3)


class TestAlignOnsetsFallback(unittest.TestCase):
    """Tests verifying the fallback when no onset lands in the valid window."""

    def test_fallback_to_first_onset_when_window_empty(self):
        """
        If all onsets are beyond anchor_window_sec, fall back to the
        very first onset as the anchor.
        """
        y = _audio_with_spikes([(3.0, 1.0), (3.5, 0.8)])
        onset_times = [3.0, 3.5]

        events = AudioEngine._align_onsets(
            onset_times, BEAT, y, SR, HOP,
            lead_in_ignore_sec=0.1, anchor_window_sec=2.0,
        )

        # Fallback anchor = 3.0 s (the first onset)
        self.assertAlmostEqual(events[0].deviation_ms, 0.0, places=3)

    def test_fallback_when_all_onsets_before_lead_in(self):
        """
        If every onset is before lead_in_ignore_sec, fall back to the
        first onset rather than raising.
        """
        y = _audio_with_spikes([(0.02, 1.0), (0.05, 0.9)])
        onset_times = [0.02, 0.05]

        events = AudioEngine._align_onsets(
            onset_times, BEAT, y, SR, HOP,
            lead_in_ignore_sec=0.1, anchor_window_sec=2.0,
        )

        # Only fallback candidates available → first onset is anchor
        self.assertAlmostEqual(events[0].deviation_ms, 0.0, places=3)


class TestAlignOnsetsDeviationValues(unittest.TestCase):
    """Tests verifying that deviation values are computed correctly."""

    def test_on_beat_onset_has_zero_deviation(self):
        """Onsets that land exactly on beat positions should have 0 ms deviation."""
        # Anchor at 0.5 s, beat period 0.5 s → beats at 0.5, 1.0, 1.5, 2.0
        y = _audio_with_spikes([(0.5, 1.0)])
        onset_times = [0.5, 1.0, 1.5, 2.0]

        events = AudioEngine._align_onsets(
            onset_times, BEAT, y, SR, HOP,
            lead_in_ignore_sec=0.1,
        )

        for ev in events:
            self.assertAlmostEqual(ev.deviation_ms, 0.0, places=6,
                                   msg=f"onset at {ev.time_sec} s should be on beat")

    def test_late_onset_has_positive_deviation(self):
        """An onset slightly after the expected beat should have positive deviation."""
        y = _audio_with_spikes([(0.5, 1.0)])
        # Second onset is 50 ms late (expected at 1.0, actual at 1.05)
        onset_times = [0.5, 1.05]

        events = AudioEngine._align_onsets(
            onset_times, BEAT, y, SR, HOP,
            lead_in_ignore_sec=0.1,
        )

        late_ev = events[1]
        self.assertAlmostEqual(late_ev.deviation_ms, 50.0, places=3)

    def test_early_onset_has_negative_deviation(self):
        """An onset slightly before the expected beat should have negative deviation."""
        y = _audio_with_spikes([(0.5, 1.0)])
        # Second onset is 30 ms early (expected at 1.0, actual at 0.97)
        onset_times = [0.5, 0.97]

        events = AudioEngine._align_onsets(
            onset_times, BEAT, y, SR, HOP,
            lead_in_ignore_sec=0.1,
        )

        early_ev = events[1]
        self.assertAlmostEqual(early_ev.deviation_ms, -30.0, places=3)


class TestAudioEngineInit(unittest.TestCase):
    """Tests for the updated AudioEngine constructor."""

    def test_default_lead_in_ignore_sec(self):
        engine = AudioEngine()
        self.assertAlmostEqual(engine.lead_in_ignore_sec, 0.1)

    def test_custom_lead_in_ignore_sec(self):
        engine = AudioEngine(lead_in_ignore_sec=0.5)
        self.assertAlmostEqual(engine.lead_in_ignore_sec, 0.5)

    def test_zero_lead_in_ignore_sec_allowed(self):
        engine = AudioEngine(lead_in_ignore_sec=0.0)
        self.assertAlmostEqual(engine.lead_in_ignore_sec, 0.0)

    def test_negative_lead_in_ignore_sec_raises(self):
        with self.assertRaises(ValueError):
            AudioEngine(lead_in_ignore_sec=-0.1)

    def test_negative_bpm_still_raises(self):
        with self.assertRaises(ValueError):
            AudioEngine(bpm=-10)


if __name__ == "__main__":
    unittest.main()
