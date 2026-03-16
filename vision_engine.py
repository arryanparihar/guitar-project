"""
vision_engine.py
================
Computer-vision module for SyncopateAI.

Responsibilities
----------------
* Detect the guitarist's left-hand fingertips frame-by-frame using
  MediaPipe's Hand Landmarker (Tasks API).
* Locate the guitar fretboard in each frame using the Hough Line
  Transform (OpenCV).
* Compute the Euclidean distance between every tracked fingertip and the
  nearest fretboard line, normalise by a per-frame hand-size reference
  (wrist-to-index-MCP distance), then aggregate those normalised distances
  into a per-frame *efficiency score* (lower average distance → higher
  score).  The normalisation makes the metric invariant to the player's
  distance from the camera.
* Return structured data that the dashboard (main.py) can plot and
  display.

MediaPipe Hand Landmark indices (0-based):
  4  = THUMB_TIP
  8  = INDEX_FINGER_TIP
  12 = MIDDLE_FINGER_TIP
  16 = RING_FINGER_TIP
  20 = PINKY_TIP
"""

from __future__ import annotations

import math
import os
import urllib.request
from dataclasses import dataclass, field
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
from mediapipe import Image as MpImage, ImageFormat
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode,
)

# ---------------------------------------------------------------------------
# Landmark index constants (MediaPipe 21-point hand model)
# ---------------------------------------------------------------------------
# Fingertip landmark indices for the four FRETTING fingers only.
# Landmark 4 (THUMB_TIP) is intentionally excluded: when playing guitar
# the thumb rests behind the neck and is often occluded.  Including it
# would distort the distance-to-fretboard calculation and produce
# inaccurate efficiency scores.
FINGERTIP_IDS = (8, 12, 16, 20)   # INDEX, MIDDLE, RING, PINKY
FINGERTIP_NAMES = ("index", "middle", "ring", "pinky")

# Landmark index that must never be used for scoring (kept as a named
# constant for documentation purposes and to prevent accidental re-addition).
_THUMB_TIP_ID = 4  # excluded – thumb is behind the neck and not scored

# Reference landmarks used to compute a dynamic hand-size scale so that
# distance metrics are invariant to the player's distance from the camera.
_WRIST_ID = 0           # WRIST
_INDEX_MCP_ID = 5       # INDEX_FINGER_MCP

# ---------------------------------------------------------------------------
# Model file management
# ---------------------------------------------------------------------------
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
_MODEL_DEFAULT_PATH = os.path.join(
    os.path.expanduser("~"), ".cache", "syncopateai", "hand_landmarker.task"
)


def _ensure_model(model_path: str = _MODEL_DEFAULT_PATH) -> str:
    """
    Return *model_path* after making sure the model file exists.

    If the file is absent, attempt to download it from the official
    MediaPipe CDN.  Raises ``RuntimeError`` if the download fails.
    """
    if os.path.exists(model_path):
        return model_path

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    try:
        urllib.request.urlretrieve(_MODEL_URL, model_path)
    except Exception as exc:
        raise RuntimeError(
            f"Could not download the MediaPipe Hand Landmarker model.\n"
            f"URL: {_MODEL_URL}\n"
            f"Error: {exc}\n\n"
            "Please download the model manually and place it at:\n"
            f"  {model_path}"
        ) from exc
    return model_path


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class FingertipData:
    """Pixel coordinates of a single fingertip in one frame."""

    name: str
    x: float
    y: float


@dataclass
class FrameResult:
    """All vision analysis results for a single video frame."""

    frame_index: int
    timestamp_sec: float
    fingertips: list[FingertipData] = field(default_factory=list)
    fretboard_y: Optional[float] = None          # y-coordinate of the nearest fretboard line
    avg_finger_height: Optional[float] = None    # mean distance (pixels) of fingertips from fretboard
    efficiency_score: Optional[float] = None     # 0–100 score for this frame
    annotated_frame: Optional[np.ndarray] = None # BGR image with drawn overlays


# ---------------------------------------------------------------------------
# Helper geometry functions
# ---------------------------------------------------------------------------

def _point_to_line_distance(px: float, py: float,
                             x1: float, y1: float,
                             x2: float, y2: float) -> float:
    """Return the perpendicular distance from point (px, py) to line (x1,y1)–(x2,y2)."""
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return math.hypot(px - x1, py - y1)
    t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    nearest_x = x1 + t * dx
    nearest_y = y1 + t * dy
    return math.hypot(px - nearest_x, py - nearest_y)


def _euclidean(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    """Euclidean distance between two 2-D points."""
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


# ---------------------------------------------------------------------------
# Fretboard detection
# ---------------------------------------------------------------------------

def detect_fretboard_lines(gray_frame: np.ndarray) -> list[tuple[float, float, float, float]]:
    """
    Detect near-horizontal lines in *gray_frame* using the Probabilistic
    Hough Line Transform and return them as a list of (x1, y1, x2, y2) tuples.

    Only lines whose angle deviates ≤ 45° from horizontal are kept, so
    that vertical artefacts (e.g. frets) are filtered out while still
    accommodating guitarists who hold the neck at a steep angle.
    """
    blurred = cv2.GaussianBlur(gray_frame, (5, 5), 0)
    edges = cv2.Canny(blurred, threshold1=50, threshold2=150)

    raw_lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=80,
        minLineLength=gray_frame.shape[1] // 4,
        maxLineGap=30,
    )

    lines: list[tuple[float, float, float, float]] = []
    if raw_lines is None:
        return lines

    for line in raw_lines:
        x1, y1, x2, y2 = map(float, line[0])
        angle_deg = abs(math.degrees(math.atan2(y2 - y1, x2 - x1)))
        if angle_deg <= 45 or angle_deg >= 135:
            lines.append((x1, y1, x2, y2))

    return lines


def _nearest_fretboard_line(
    fingertip_y: float,
    lines: list[tuple[float, float, float, float]],
) -> Optional[tuple[float, float, float, float]]:
    """Return the line whose mid-point y-coordinate is closest to *fingertip_y*."""
    if not lines:
        return None
    return min(lines, key=lambda ln: abs((ln[1] + ln[3]) / 2 - fingertip_y))


# ---------------------------------------------------------------------------
# Efficiency scoring
# ---------------------------------------------------------------------------

def _compute_efficiency(avg_distance_norm: float, max_distance_norm: float = 1.0) -> float:
    """
    Map average *normalised* fingertip-to-fretboard distance to a 0–100
    efficiency score.

    Distances are expressed as multiples of the wrist-to-index-MCP
    reference length so that the metric is invariant to the player's
    distance from the camera.

    A normalised distance of 0 → 100 (fingertips resting on the strings).
    A normalised distance ≥ *max_distance_norm* → 0.

    The default threshold of 1.0 means that once the average fingertip
    is one full wrist-to-MCP length away from the fretboard, the score
    bottoms out at 0.
    """
    clamped = min(avg_distance_norm, max_distance_norm)
    return round((1.0 - clamped / max_distance_norm) * 100.0, 2)


# ---------------------------------------------------------------------------
# Core processing class
# ---------------------------------------------------------------------------

class VisionEngine:
    """
    Wraps MediaPipe's Hand Landmarker (Tasks API) and OpenCV to analyse
    each video frame.

    Parameters
    ----------
    model_path : str, optional
        Path to the ``hand_landmarker.task`` model file.  If the file does
        not exist at this path, it is downloaded automatically from the
        MediaPipe CDN on first instantiation.
    max_num_hands : int
        Maximum number of hands to detect (default 1 – only the fretting hand).
    min_hand_detection_confidence : float
        Minimum confidence for the initial detection.
    min_tracking_confidence : float
        Minimum confidence for subsequent tracking.
    """

    def __init__(
        self,
        model_path: str = _MODEL_DEFAULT_PATH,
        max_num_hands: int = 1,
        min_hand_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        resolved_model = _ensure_model(model_path)

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=resolved_model),
            running_mode=RunningMode.IMAGE,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_hand_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = HandLandmarker.create_from_options(options)
        self._last_fret_lines: list[tuple[float, float, float, float]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        frame_index: int = 0,
        fps: float = 30.0,
    ) -> FrameResult:
        """
        Analyse a single BGR video frame.

        Parameters
        ----------
        frame_bgr : np.ndarray
            BGR image (as returned by ``cv2.VideoCapture.read()``).
        frame_index : int
            Zero-based index of this frame in the video.
        fps : float
            Video frames-per-second (used to compute ``timestamp_sec``).

        Returns
        -------
        FrameResult
            Populated result object.  ``annotated_frame`` contains the
            original frame with landmarks and fretboard lines drawn.
        """
        h, w = frame_bgr.shape[:2]
        timestamp_sec = frame_index / fps if fps > 0 else 0.0
        result = FrameResult(frame_index=frame_index, timestamp_sec=timestamp_sec)

        annotated = frame_bgr.copy()
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        # --- 1. Fretboard detection ------------------------------------------
        fret_lines = detect_fretboard_lines(gray)
        if fret_lines:
            self._last_fret_lines = fret_lines
        elif self._last_fret_lines:
            fret_lines = self._last_fret_lines
        for x1, y1, x2, y2 in fret_lines:
            cv2.line(annotated, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 1)

        # --- 2. Hand landmark detection (MediaPipe Tasks API) ----------------
        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = MpImage(image_format=ImageFormat.SRGB, data=rgb_frame)
        detection = self._landmarker.detect(mp_image)

        if not detection.hand_landmarks:
            result.annotated_frame = annotated
            return result

        # Use the first detected hand (fretting hand)
        hand = detection.hand_landmarks[0]

        # Compute dynamic hand-size reference (wrist → index-finger MCP)
        wrist_lm = hand[_WRIST_ID]
        mcp_lm = hand[_INDEX_MCP_ID]
        hand_ref_scale = _euclidean(
            (wrist_lm.x * w, wrist_lm.y * h),
            (mcp_lm.x * w, mcp_lm.y * h),
        )

        # Draw skeleton connections
        _draw_hand_landmarks(annotated, hand, w, h)

        # --- 3. Extract fingertip coordinates ---------------------------------
        fingertips: list[FingertipData] = []
        for tip_id, tip_name in zip(FINGERTIP_IDS, FINGERTIP_NAMES):
            lm = hand[tip_id]
            fx, fy = lm.x * w, lm.y * h
            fingertips.append(FingertipData(name=tip_name, x=fx, y=fy))
            cv2.circle(annotated, (int(fx), int(fy)), 6, (0, 0, 255), -1)

        result.fingertips = fingertips

        # --- 4. Compute distances to fretboard --------------------------------
        if fret_lines and fingertips:
            distances: list[float] = []
            for ft in fingertips:
                nearest_line = _nearest_fretboard_line(ft.y, fret_lines)
                if nearest_line:
                    dist = _point_to_line_distance(ft.x, ft.y, *nearest_line)
                    distances.append(dist)
                    # Project fingertip vertically onto the nearest fretboard line
                    x1, y1, x2, y2 = nearest_line
                    dx = x2 - x1
                    dy = y2 - y1
                    denom = dx * dx + dy * dy
                    if denom > 0:
                        t = ((ft.x - x1) * dx + (ft.y - y1) * dy) / denom
                        t = max(0.0, min(1.0, t))
                        proj_x = int(x1 + t * dx)
                        proj_y = int(y1 + t * dy)
                    else:
                        proj_x, proj_y = int(x1), int(y1)
                    cv2.line(
                        annotated,
                        (int(ft.x), int(ft.y)),
                        (proj_x, proj_y),
                        (255, 0, 255),
                        1,
                    )

            if distances:
                avg_dist = float(np.mean(distances))
                result.avg_finger_height = avg_dist
                result.fretboard_y = float(
                    np.mean([(ln[1] + ln[3]) / 2 for ln in fret_lines])
                )

                # Normalise distances by hand reference scale
                if hand_ref_scale > 0:
                    norm_distances = [d / hand_ref_scale for d in distances]
                else:
                    norm_distances = distances
                avg_norm_dist = float(np.mean(norm_distances))
                result.efficiency_score = _compute_efficiency(avg_norm_dist)

                # Overlay score on frame
                cv2.putText(
                    annotated,
                    f"Efficiency: {result.efficiency_score:.1f}%",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )

        result.annotated_frame = annotated
        return result

    def process_video(
        self,
        video_path: str,
        max_frames: Optional[int] = None,
        skip_frames: int = 0,
    ) -> list[FrameResult]:
        """
        Process every frame of a video file and return a list of
        :class:`FrameResult` objects.

        Parameters
        ----------
        video_path : str
            Path to the input video file.
        max_frames : int, optional
            Stop after this many *processed* frames (useful for previews).
        skip_frames : int
            Number of frames to skip between processed frames (0 = process
            every frame, 1 = process every other frame, …).

        Returns
        -------
        list[FrameResult]
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        results: list[FrameResult] = []
        frame_index = 0
        processed = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if skip_frames > 0 and frame_index % (skip_frames + 1) != 0:
                    frame_index += 1
                    continue

                results.append(self.process_frame(frame, frame_index, fps))
                processed += 1
                frame_index += 1

                if max_frames is not None and processed >= max_frames:
                    break
        finally:
            cap.release()

        return results

    def close(self) -> None:
        """Release MediaPipe resources."""
        self._landmarker.close()

    def __enter__(self) -> "VisionEngine":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Drawing helper
# ---------------------------------------------------------------------------

# MediaPipe hand connections (pairs of landmark indices)
_HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),           # index
    (0, 9), (9, 10), (10, 11), (11, 12),      # middle
    (0, 13), (13, 14), (14, 15), (15, 16),    # ring
    (0, 17), (17, 18), (18, 19), (19, 20),    # pinky
    (5, 9), (9, 13), (13, 17),                # palm
]


def _draw_hand_landmarks(
    image: np.ndarray,
    landmarks: list,
    width: int,
    height: int,
    connection_color: tuple = (0, 255, 0),
    landmark_color: tuple = (255, 0, 0),
) -> None:
    """Draw hand skeleton connections and landmark dots on *image* in-place."""
    # Convert normalised coords to pixel coords
    pts = [(int(lm.x * width), int(lm.y * height)) for lm in landmarks]

    for start_idx, end_idx in _HAND_CONNECTIONS:
        if start_idx < len(pts) and end_idx < len(pts):
            cv2.line(image, pts[start_idx], pts[end_idx], connection_color, 2)

    for pt in pts:
        cv2.circle(image, pt, 4, landmark_color, -1)

