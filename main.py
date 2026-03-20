"""
main.py
=======
SyncopateAI – Streamlit dashboard.

Run with:
    streamlit run main.py

Features
--------
* Paste a raw ASCII guitar tab and click **Analyse Tab** to parse it into
  structured JSON data using TabParser.
"""

from __future__ import annotations

import csv
import io
from typing import Optional

import streamlit as st

from audio_engine import AudioAnalysisResult
from tab_engine import TabParser
from vision_engine import FrameResult

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="SyncopateAI",
    page_icon="🎸",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _build_finger_height_csv(
    results: list[FrameResult],
    audio_result: Optional[AudioAnalysisResult] = None,
) -> bytes:
    """
    Convert vision analysis results to a UTF-8 encoded CSV suitable for
    download via ``st.download_button``.

    Columns
    -------
    frame_index, timestamp_sec, avg_finger_height_px, efficiency_score,
    index_x, index_y, middle_x, middle_y, ring_x, ring_y, pinky_x, pinky_y
    [, Audio_Deviation_ms]  – only present when *audio_result* is supplied
                              and contains at least one onset event.

    For each video frame the ``Audio_Deviation_ms`` value is taken from the
    audio onset whose timestamp is closest to the frame's timestamp.  This
    creates a unified dataset that correlates hand technique with timing.
    """
    fingertip_names = ("index", "middle", "ring", "pinky")

    include_audio = (
        audio_result is not None and bool(audio_result.onset_events)
    )

    header = [
        "frame_index",
        "timestamp_sec",
        "avg_finger_height_px",
        "efficiency_score",
    ] + [f"{n}_{axis}" for n in fingertip_names for axis in ("x", "y")]

    if include_audio:
        header.append("Audio_Deviation_ms")

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(header)

    # Pre-extract onset times for fast nearest-neighbour lookup.
    onset_times: list[float] = (
        [e.time_sec for e in audio_result.onset_events] if include_audio else []
    )

    for r in results:
        # Build a lookup so missing fingertips are represented as empty cells
        tip_coords: dict[str, tuple] = {ft.name: (ft.x, ft.y) for ft in r.fingertips}
        row = [
            r.frame_index,
            round(r.timestamp_sec, 6),
            round(r.avg_finger_height, 4) if r.avg_finger_height is not None else "",
            round(r.efficiency_score, 4) if r.efficiency_score is not None else "",
        ]
        for name in fingertip_names:
            if name in tip_coords:
                row += [round(tip_coords[name][0], 2), round(tip_coords[name][1], 2)]
            else:
                row += ["", ""]

        if include_audio:
            # Find the index of the onset whose time is closest to this frame.
            closest_idx = min(
                range(len(onset_times)),
                key=lambda i: abs(onset_times[i] - r.timestamp_sec),
            )
            row.append(round(audio_result.onset_events[closest_idx].deviation_ms, 4))

        writer.writerow(row)

    return buf.getvalue().encode("utf-8")


# ---------------------------------------------------------------------------
# Main dashboard
# ---------------------------------------------------------------------------

def main() -> None:
    st.title("🎸 SyncopateAI – Guitar Tab Analyser")
    st.markdown(
        "Paste a raw ASCII guitar tab below and click **Analyse Tab** "
        "to see the structured note data."
    )
    st.divider()

    raw_tab = st.text_area(
        "Guitar Tab",
        height=300,
        placeholder=(
            "e|--0-------<12>-------[12]-|\n"
            "B|--1--------12---------1---|\n"
            "G|--0---------0---------0---|\n"
            "D|--2---------2---------2---|\n"
            "A|--0---------0---------0---|\n"
            "E|--x---------x---------x---|"
        ),
    )

    if st.button("Analyse Tab", type="primary"):
        if not raw_tab.strip():
            st.warning("⚠️ Please paste a guitar tab before analysing.")
        else:
            parser = TabParser()
            result = parser.parse(raw_tab)
            st.subheader("Parsed Tab Data")
            st.json(result)


if __name__ == "__main__":
    main()
