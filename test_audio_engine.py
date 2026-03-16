"""Unit tests for AudioEngine._align_onsets (improved anchor selection).

These tests validate the updated onset-alignment logic without requiring
a real audio file – audio arrays are synthesised in-memory.
"""

import unittest

import numpy as np

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
