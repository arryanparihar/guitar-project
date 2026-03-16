"""
main.py
=======
SyncopateAI – Streamlit dashboard.

Run with:
    streamlit run main.py

Features
--------
* Upload a video of a guitarist's left hand **or** use a live webcam feed.
* Run the VisionEngine to track fingertip positions and detect the fretboard.
* (Optional) upload an audio file to run the AudioEngine for onset / BPM analysis.
* Visualise:
    - Annotated video frames with landmark overlays
    - "Finger Height vs. Time" chart
    - "Tempo Deviation" chart
    - Overall efficiency score
"""

from __future__ import annotations

import csv
import io
import os
import tempfile
from typing import Optional

import cv2
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from audio_engine import AudioEngine, AudioAnalysisResult
from vision_engine import VisionEngine, FrameResult

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

def _save_upload(uploaded_file) -> str:
    """Save a Streamlit UploadedFile to a temp file and return its path."""
    suffix = os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        return tmp.name


def _frames_to_gif_bytes(frames: list[np.ndarray], fps: float = 10.0) -> bytes:
    """
    Encode a list of BGR numpy frames as an animated GIF (bytes).
    Falls back to returning the first frame as JPEG if there is only one frame.
    """
    if not frames:
        return b""

    # Convert to RGB for display
    rgb_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]

    if len(rgb_frames) == 1:
        success, buf = cv2.imencode(".jpg", frames[0])
        return bytes(buf) if success else b""

    # Encode each frame as PNG into memory
    import io
    from PIL import Image

    pil_frames = [Image.fromarray(f) for f in rgb_frames]
    buf = io.BytesIO()
    pil_frames[0].save(
        buf,
        format="GIF",
        save_all=True,
        append_images=pil_frames[1:],
        loop=0,
        duration=int(1000 / fps),
    )
    return buf.getvalue()


def _finger_height_chart(results: list[FrameResult]) -> go.Figure:
    """Create a Plotly line chart of average finger height (pixels) vs. time."""
    times = [r.timestamp_sec for r in results if r.avg_finger_height is not None]
    heights = [r.avg_finger_height for r in results if r.avg_finger_height is not None]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=times,
            y=heights,
            mode="lines+markers",
            name="Avg Finger Height",
            line=dict(color="#EF553B", width=2),
            marker=dict(size=4),
        )
    )
    fig.update_layout(
        title="Finger Height vs. Time",
        xaxis_title="Time (s)",
        yaxis_title="Distance from Fretboard (px)",
        template="plotly_dark",
        height=350,
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


def _efficiency_score_chart(results: list[FrameResult]) -> go.Figure:
    """Create a Plotly line chart of per-frame efficiency score vs. time."""
    times = [r.timestamp_sec for r in results if r.efficiency_score is not None]
    scores = [r.efficiency_score for r in results if r.efficiency_score is not None]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=times,
            y=scores,
            mode="lines",
            name="Efficiency Score",
            fill="tozeroy",
            line=dict(color="#00CC96", width=2),
        )
    )
    fig.update_layout(
        title="Efficiency Score vs. Time",
        xaxis_title="Time (s)",
        yaxis_title="Score (0–100)",
        yaxis=dict(range=[0, 105]),
        template="plotly_dark",
        height=350,
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


def _tempo_deviation_chart(audio_result: AudioAnalysisResult) -> go.Figure:
    """Create a Plotly bar chart of tempo deviation (ms) per onset."""
    events = audio_result.onset_events
    indices = [e.index for e in events]
    deviations = [e.deviation_ms for e in events]
    colors = ["#EF553B" if d > 0 else "#636EFA" for d in deviations]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=indices,
            y=deviations,
            marker_color=colors,
            name="Deviation (ms)",
        )
    )
    fig.add_hline(y=0, line_dash="dash", line_color="white", line_width=1)
    fig.update_layout(
        title="Tempo Deviation per Note Onset",
        xaxis_title="Onset #",
        yaxis_title="Deviation from Grid (ms)  [+ late, − early]",
        template="plotly_dark",
        height=350,
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


def _rms_envelope_chart(audio_result: AudioAnalysisResult) -> go.Figure:
    """Create a Plotly area chart of the audio RMS envelope."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=audio_result.rms_times,
            y=audio_result.rms_envelope,
            mode="lines",
            fill="tozeroy",
            line=dict(color="#AB63FA", width=1),
            name="RMS Energy",
        )
    )
    # Overlay onset markers
    for t in audio_result.onset_times:
        fig.add_vline(x=t, line_width=1, line_dash="dot", line_color="yellow")

    fig.update_layout(
        title="Audio Energy Envelope (yellow lines = detected onsets)",
        xaxis_title="Time (s)",
        yaxis_title="Normalised RMS",
        yaxis=dict(range=[0, 1.05]),
        template="plotly_dark",
        height=300,
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


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
# Video processing helpers
# ---------------------------------------------------------------------------

def _process_uploaded_video(
    video_path: str,
    max_frames: int,
    skip_frames: int,
    progress_bar,
) -> list[FrameResult]:
    """Run VisionEngine on *video_path* with a Streamlit progress bar."""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()

    results: list[FrameResult] = []
    processed = 0

    with VisionEngine() as engine:
        cap = cv2.VideoCapture(video_path)
        frame_idx = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if skip_frames > 0 and frame_idx % (skip_frames + 1) != 0:
                    frame_idx += 1
                    continue

                results.append(engine.process_frame(frame, frame_idx, fps))
                processed += 1
                frame_idx += 1

                progress_bar.progress(
                    min(frame_idx / max(total_frames, 1), 1.0),
                    text=f"Analysing frame {frame_idx}/{total_frames}…",
                )
                if processed >= max_frames:
                    break
        finally:
            cap.release()

    progress_bar.progress(1.0, text="Vision analysis complete ✔")
    return results


def _run_webcam_session(
    num_frames: int,
    skip_frames: int,
    status_placeholder,
) -> list[FrameResult]:
    """
    Capture *num_frames* frames from the default webcam (camera index 0).

    Returns a list of FrameResult objects.  A Streamlit placeholder is
    updated during capture so the user can see progress.
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("⚠️ Could not open webcam. Make sure a camera is connected.")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    results: list[FrameResult] = []
    frame_idx = 0
    captured = 0

    frame_placeholder = status_placeholder.empty()

    with VisionEngine() as engine:
        try:
            while captured < num_frames:
                ret, frame = cap.read()
                if not ret:
                    break

                if skip_frames > 0 and frame_idx % (skip_frames + 1) != 0:
                    frame_idx += 1
                    continue

                r = engine.process_frame(frame, frame_idx, fps)
                results.append(r)
                captured += 1
                frame_idx += 1

                # Show the annotated frame live in the placeholder
                if r.annotated_frame is not None:
                    rgb = cv2.cvtColor(r.annotated_frame, cv2.COLOR_BGR2RGB)
                    frame_placeholder.image(
                        rgb,
                        caption=f"Frame {captured}/{num_frames}",
                        use_container_width=True,
                    )
        finally:
            cap.release()

    frame_placeholder.empty()
    return results


# ---------------------------------------------------------------------------
# Sidebar configuration
# ---------------------------------------------------------------------------

def _render_sidebar() -> dict:
    """Render sidebar controls and return the configuration dict."""
    st.sidebar.image(
        "https://img.icons8.com/emoji/96/000000/guitar-emoji.png",
        width=80,
    )
    st.sidebar.title("SyncopateAI")
    st.sidebar.markdown("*Guitar technique analyser*")
    st.sidebar.divider()

    cfg: dict = {}

    # Input source
    cfg["input_source"] = st.sidebar.radio(
        "Input source",
        options=["Upload video", "Webcam"],
        index=0,
    )

    st.sidebar.divider()

    # Vision settings
    st.sidebar.subheader("Vision settings")
    cfg["max_frames"] = st.sidebar.slider(
        "Max frames to analyse", min_value=10, max_value=600, value=150, step=10
    )
    cfg["skip_frames"] = st.sidebar.slider(
        "Skip N frames between analyses", min_value=0, max_value=10, value=1
    )

    st.sidebar.divider()

    # Audio settings
    st.sidebar.subheader("Audio settings")
    cfg["bpm"] = st.sidebar.number_input(
        "Target BPM", min_value=20.0, max_value=300.0, value=120.0, step=1.0
    )
    cfg["analyse_audio"] = st.sidebar.checkbox("Analyse audio file", value=True)

    return cfg


# ---------------------------------------------------------------------------
# Main dashboard
# ---------------------------------------------------------------------------

def main() -> None:
    cfg = _render_sidebar()

    st.title("🎸 SyncopateAI – Guitar Technique Analyser")
    st.markdown(
        """
        Upload a video of your **left (fretting) hand** and—optionally—an **audio file**
        to get instant feedback on your technique efficiency and timing accuracy.
        """
    )
    st.divider()

    # ------------------------------------------------------------------ #
    #  1. Collect inputs                                                   #
    # ------------------------------------------------------------------ #
    video_path: Optional[str] = None
    audio_path: Optional[str] = None

    col_vid, col_aud = st.columns([3, 2])

    with col_vid:
        if cfg["input_source"] == "Upload video":
            uploaded_video = st.file_uploader(
                "Upload video (mp4, avi, mov)", type=["mp4", "avi", "mov"]
            )
            if uploaded_video:
                video_path = _save_upload(uploaded_video)
                st.video(uploaded_video)
        else:
            st.info("📷 Webcam mode: click **Run Analysis** to start capturing.")
            video_path = "__webcam__"

    with col_aud:
        if cfg["analyse_audio"]:
            uploaded_audio = st.file_uploader(
                "Upload audio (wav, mp3, flac)", type=["wav", "mp3", "flac", "ogg"]
            )
            if uploaded_audio:
                audio_path = _save_upload(uploaded_audio)
                st.audio(uploaded_audio)

    st.divider()

    # ------------------------------------------------------------------ #
    #  2. Run Analysis                                                     #
    # ------------------------------------------------------------------ #
    run_btn = st.button("▶ Run Analysis", type="primary", use_container_width=True)

    if not run_btn:
        st.info("Configure the settings in the sidebar and click **Run Analysis**.")
        return

    if video_path is None:
        st.warning("⚠️ Please upload a video file (or switch to Webcam mode).")
        return

    # ---- Vision analysis -----------------------------------------------
    vision_results: list[FrameResult] = []

    with st.spinner("Running vision analysis…"):
        if video_path == "__webcam__":
            webcam_placeholder = st.empty()
            vision_results = _run_webcam_session(
                num_frames=cfg["max_frames"],
                skip_frames=cfg["skip_frames"],
                status_placeholder=webcam_placeholder,
            )
        else:
            progress = st.progress(0.0, text="Starting vision analysis…")
            vision_results = _process_uploaded_video(
                video_path=video_path,
                max_frames=cfg["max_frames"],
                skip_frames=cfg["skip_frames"],
                progress_bar=progress,
            )

    # ---- Audio analysis ------------------------------------------------
    audio_result: Optional[AudioAnalysisResult] = None

    if cfg["analyse_audio"] and audio_path:
        with st.spinner("Running audio analysis…"):
            engine = AudioEngine(bpm=cfg["bpm"])
            audio_result = engine.analyse_file(audio_path)

    # ------------------------------------------------------------------ #
    #  3. Results dashboard                                                #
    # ------------------------------------------------------------------ #
    st.subheader("📊 Analysis Results")

    if not vision_results:
        st.warning("No frames were processed. Check your input source and try again.")
        return

    # --- Aggregate metrics ---
    scored_results = [r for r in vision_results if r.efficiency_score is not None]
    overall_vision_score = (
        round(float(np.mean([r.efficiency_score for r in scored_results])), 1)
        if scored_results
        else None
    )

    # --- Metric cards ---
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Frames analysed", len(vision_results))
    m2.metric(
        "Overall efficiency",
        f"{overall_vision_score:.1f} / 100" if overall_vision_score is not None else "N/A",
    )
    if audio_result:
        m3.metric("Timing score", f"{audio_result.timing_score:.1f} / 100")
        m4.metric(
            "Detected BPM",
            f"{audio_result.bpm_detected:.1f}" if audio_result.bpm_detected else "N/A",
        )
    else:
        m3.metric("Timing score", "N/A (no audio)")
        m4.metric("Target BPM", f"{cfg['bpm']:.0f}")

    st.divider()

    # --- Charts ---
    tab_vision, tab_audio, tab_frames = st.tabs(
        ["📈 Vision charts", "🎵 Audio charts", "🖼️ Sample frames"]
    )

    with tab_vision:
        if scored_results:
            st.plotly_chart(_finger_height_chart(vision_results), use_container_width=True)
            st.plotly_chart(_efficiency_score_chart(vision_results), use_container_width=True)

            # --- CSV Export ---
            csv_bytes = _build_finger_height_csv(vision_results, audio_result)
            st.download_button(
                label="⬇️ Export Finger Height data to CSV",
                data=csv_bytes,
                file_name="syncopateai_finger_height.csv",
                mime="text/csv",
                help=(
                    "Downloads a CSV file with per-frame timestamp, average finger "
                    "height (px), efficiency score, and individual fingertip coordinates. "
                    "When audio analysis has been run, an Audio_Deviation_ms column is "
                    "also included, correlating each frame with the closest note onset."
                ),
            )
        else:
            st.info(
                "No fretboard or hand was detected in the processed frames.  "
                "Make sure the video clearly shows the fretting hand and fretboard."
            )

    with tab_audio:
        if audio_result:
            st.plotly_chart(_rms_envelope_chart(audio_result), use_container_width=True)
            if audio_result.onset_events:
                st.plotly_chart(_tempo_deviation_chart(audio_result), use_container_width=True)
            else:
                st.info("No note onsets were detected in the audio.")

            with st.expander("Audio analysis details"):
                st.json(
                    {
                        "duration_sec": round(audio_result.duration_sec, 2),
                        "sample_rate": audio_result.sample_rate,
                        "bpm_target": audio_result.bpm_target,
                        "bpm_detected": (
                            round(audio_result.bpm_detected, 1)
                            if audio_result.bpm_detected
                            else None
                        ),
                        "num_onsets": len(audio_result.onset_times),
                        "avg_deviation_ms": round(audio_result.avg_deviation_ms, 2),
                        "timing_score": audio_result.timing_score,
                    }
                )
        else:
            st.info("Upload an audio file and enable audio analysis to see timing charts.")

    with tab_frames:
        # Show up to 6 annotated sample frames
        sample_step = max(1, len(vision_results) // 6)
        samples = vision_results[::sample_step][:6]

        cols = st.columns(3)
        for i, res in enumerate(samples):
            if res.annotated_frame is not None:
                rgb = cv2.cvtColor(res.annotated_frame, cv2.COLOR_BGR2RGB)
                cols[i % 3].image(
                    rgb,
                    caption=(
                        f"t={res.timestamp_sec:.2f}s  "
                        f"score={res.efficiency_score if res.efficiency_score is not None else 'N/A'}"
                    ),
                    use_container_width=True,
                )

    # ------------------------------------------------------------------ #
    #  4. Cleanup temp files                                               #
    # ------------------------------------------------------------------ #
    for path in [video_path, audio_path]:
        if path and path != "__webcam__" and os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass


if __name__ == "__main__":
    main()
