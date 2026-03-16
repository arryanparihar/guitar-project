"""Unit tests for main.py helper functions.

Streamlit is mocked at the sys.modules level before importing main so that
``st.set_page_config`` and other Streamlit calls do not raise at import
time.  All tests exercise pure-computation logic and require no browser,
camera, or model file.
"""

from __future__ import annotations

import csv
import io
import sys
import types
import unittest
import unittest.mock


# ---------------------------------------------------------------------------
# Streamlit stub – must be registered BEFORE importing main
# ---------------------------------------------------------------------------

def _make_streamlit_stub() -> types.ModuleType:
    """Return a minimal module that satisfies every top-level Streamlit call
    in main.py without actually starting a server."""
    st = types.ModuleType("streamlit")
    _noop = lambda *a, **kw: None  # noqa: E731
    for attr in (
        "set_page_config", "title", "markdown", "divider",
        "columns", "sidebar", "button", "info", "warning",
        "error", "spinner", "expander", "tabs", "progress",
        "download_button", "plotly_chart", "subheader", "metric",
        "video", "audio", "file_uploader", "json", "empty",
        "image", "add_vline", "add_hline",
    ):
        setattr(st, attr, _noop)

    # sidebar needs attribute access (st.sidebar.title, etc.)
    sidebar = types.SimpleNamespace(
        **{a: _noop for a in (
            "image", "title", "markdown", "divider", "subheader",
            "radio", "slider", "number_input", "checkbox",
        )}
    )
    st.sidebar = sidebar
    return st


sys.modules.setdefault("streamlit", _make_streamlit_stub())

# Now it is safe to import from main.
from main import _build_finger_height_csv  # noqa: E402
from audio_engine import AudioAnalysisResult, OnsetEvent  # noqa: E402
from vision_engine import FrameResult, FingertipData  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_frame(
    frame_index: int,
    timestamp_sec: float,
    avg_height: float | None = 10.0,
    efficiency: float | None = 90.0,
    fingertips: list[FingertipData] | None = None,
) -> FrameResult:
    """Build a minimal FrameResult for testing."""
    if fingertips is None:
        fingertips = [
            FingertipData("index", 100.0, 200.0),
            FingertipData("middle", 150.0, 210.0),
            FingertipData("ring", 200.0, 215.0),
            FingertipData("pinky", 250.0, 220.0),
        ]
    return FrameResult(
        frame_index=frame_index,
        timestamp_sec=timestamp_sec,
        fingertips=fingertips,
        avg_finger_height=avg_height,
        efficiency_score=efficiency,
    )


def _parse_csv(data: bytes) -> tuple[list[str], list[list[str]]]:
    """Return (header_row, data_rows) from CSV bytes."""
    text = data.decode("utf-8")
    reader = csv.reader(io.StringIO(text))
    rows = list(reader)
    return rows[0], rows[1:]


def _make_audio_result(
    onset_times_and_deviations: list[tuple[float, float]],
) -> AudioAnalysisResult:
    """Build a minimal AudioAnalysisResult with the given onset data.

    Parameters
    ----------
    onset_times_and_deviations:
        List of (time_sec, deviation_ms) pairs.
    """
    events = [
        OnsetEvent(
            index=i,
            time_sec=t,
            expected_sec=t,  # simplified – not relevant for matching
            deviation_ms=dev,
        )
        for i, (t, dev) in enumerate(onset_times_and_deviations)
    ]
    onset_times = [t for t, _ in onset_times_and_deviations]
    return AudioAnalysisResult(
        sample_rate=44100,
        duration_sec=10.0,
        bpm_target=120.0,
        bpm_detected=120.0,
        onset_times=onset_times,
        onset_events=events,
        rms_envelope=[0.5],
        rms_times=[0.0],
        avg_deviation_ms=10.0,
        timing_score=95.0,
    )


# ---------------------------------------------------------------------------
# Tests – no audio
# ---------------------------------------------------------------------------

class TestBuildFingerHeightCsvNoAudio(unittest.TestCase):
    """CSV output when no audio result is provided."""

    def setUp(self):
        self.frames = [_make_frame(i, i * 0.033) for i in range(3)]
        self.header, self.rows = _parse_csv(_build_finger_height_csv(self.frames))

    def test_header_has_no_audio_column(self):
        self.assertNotIn("Audio_Deviation_ms", self.header)

    def test_row_count_matches_frames(self):
        self.assertEqual(len(self.rows), len(self.frames))

    def test_standard_columns_present(self):
        for col in ("frame_index", "timestamp_sec",
                    "avg_finger_height_px", "efficiency_score"):
            self.assertIn(col, self.header)

    def test_fingertip_columns_present(self):
        for name in ("index", "middle", "ring", "pinky"):
            self.assertIn(f"{name}_x", self.header)
            self.assertIn(f"{name}_y", self.header)

    def test_explicit_none_audio_result(self):
        header, _ = _parse_csv(
            _build_finger_height_csv(self.frames, audio_result=None)
        )
        self.assertNotIn("Audio_Deviation_ms", header)


# ---------------------------------------------------------------------------
# Tests – audio present but no onset events
# ---------------------------------------------------------------------------

class TestBuildFingerHeightCsvEmptyOnsets(unittest.TestCase):
    """CSV output when audio was run but produced no onsets."""

    def test_no_audio_column_when_no_onset_events(self):
        audio = _make_audio_result([])  # empty onsets
        frames = [_make_frame(0, 0.0)]
        header, _ = _parse_csv(_build_finger_height_csv(frames, audio))
        self.assertNotIn("Audio_Deviation_ms", header)


# ---------------------------------------------------------------------------
# Tests – audio with onset events
# ---------------------------------------------------------------------------

class TestBuildFingerHeightCsvWithAudio(unittest.TestCase):
    """CSV output when audio analysis is provided and has onset events."""

    def setUp(self):
        # Three frames at t=0.0, 0.1, 0.2 s
        self.frames = [_make_frame(i, i * 0.1) for i in range(3)]
        # Two onsets at t=0.05 s (dev=+10 ms) and t=0.15 s (dev=−5 ms)
        self.audio = _make_audio_result([(0.05, 10.0), (0.15, -5.0)])
        self.header, self.rows = _parse_csv(
            _build_finger_height_csv(self.frames, self.audio)
        )

    def test_audio_column_present(self):
        self.assertIn("Audio_Deviation_ms", self.header)

    def test_audio_column_is_last(self):
        self.assertEqual(self.header[-1], "Audio_Deviation_ms")

    def test_row_count_matches_frames(self):
        self.assertEqual(len(self.rows), len(self.frames))

    def test_each_row_has_audio_value(self):
        audio_col = self.header.index("Audio_Deviation_ms")
        for row in self.rows:
            self.assertNotEqual(row[audio_col], "",
                                "Audio_Deviation_ms must not be empty")

    def test_closest_onset_frame_0(self):
        """Frame at t=0.0 is closest to onset at t=0.05 (dev=+10 ms)."""
        audio_col = self.header.index("Audio_Deviation_ms")
        self.assertAlmostEqual(float(self.rows[0][audio_col]), 10.0, places=3)

    def test_closest_onset_frame_1(self):
        """Frame at t=0.1 is equidistant from both onsets; it receives one of the two valid deviations."""
        audio_col = self.header.index("Audio_Deviation_ms")
        value = float(self.rows[1][audio_col])
        self.assertIn(value, (10.0, -5.0),
                      "Equidistant frame should receive one of the two onset deviations")

    def test_closest_onset_frame_2(self):
        """Frame at t=0.2 is closest to onset at t=0.15 (dev=−5 ms)."""
        audio_col = self.header.index("Audio_Deviation_ms")
        self.assertAlmostEqual(float(self.rows[2][audio_col]), -5.0, places=3)

    def test_standard_columns_still_present(self):
        for col in ("frame_index", "timestamp_sec",
                    "avg_finger_height_px", "efficiency_score"):
            self.assertIn(col, self.header)


# ---------------------------------------------------------------------------
# Tests – single onset snapping
# ---------------------------------------------------------------------------

class TestSingleOnsetSnapping(unittest.TestCase):
    """All frames should map to the only onset when there is just one."""

    def test_single_onset_all_frames_get_same_deviation(self):
        frames = [_make_frame(i, i * 0.5) for i in range(5)]
        audio = _make_audio_result([(2.0, 33.0)])  # single onset at t=2.0
        header, rows = _parse_csv(_build_finger_height_csv(frames, audio))
        audio_col = header.index("Audio_Deviation_ms")
        for row in rows:
            self.assertAlmostEqual(float(row[audio_col]), 33.0, places=3)


# ---------------------------------------------------------------------------
# Tests – missing fingertips handled correctly alongside audio column
# ---------------------------------------------------------------------------

class TestMissingFingertipsWithAudio(unittest.TestCase):
    """Empty fingertip cells should not shift the Audio_Deviation_ms column."""

    def test_empty_fingertips_with_audio(self):
        frame = _make_frame(0, 0.0, fingertips=[])  # no fingertips detected
        audio = _make_audio_result([(0.0, 7.5)])
        header, rows = _parse_csv(_build_finger_height_csv([frame], audio))
        audio_col = header.index("Audio_Deviation_ms")
        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(float(rows[0][audio_col]), 7.5, places=3)
        # Fingertip cells should be empty strings
        for name in ("index", "middle", "ring", "pinky"):
            x_col = header.index(f"{name}_x")
            self.assertEqual(rows[0][x_col], "")


if __name__ == "__main__":
    unittest.main()
