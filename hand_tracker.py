"""Hand landmark tracking for guitar playing analysis.

Uses the MediaPipe Tasks ``HandLandmarker`` API and OpenCV to track the
four fretting-hand fingertips (index, middle, ring, pinky) relative to
a dynamically detected guitar string reference line and logs distances
to CSV.
"""

import argparse
import csv
import os
import ssl
import time
import urllib.request

import cv2
import mediapipe as mp
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# MediaPipe hand-landmark indices (Tasks API uses plain int indices)
FINGER_TIPS = {
    "Index": 8,
    "Middle": 12,
    "Ring": 16,
    "Pinky": 20,
}

CSV_HEADER = ["timestamp"] + [f"{name}_distance" for name in FINGER_TIPS]

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)
DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task"
)

# ---------------------------------------------------------------------------
# Hough-based reference-line detection
# ---------------------------------------------------------------------------


def detect_reference_line(frame, fallback_y=None):
    """Detect the most prominent horizontal line in *frame*.

    Uses a Canny edge detector followed by a Probabilistic Hough Line
    Transform to find near-horizontal lines (e.g. guitar strings).
    The median Y-coordinate of all qualifying line endpoints is returned
    as the reference-line position.

    Parameters
    ----------
    frame : numpy.ndarray
        BGR video frame.
    fallback_y : int or None
        Value returned when no line is detected.  When *None* the
        vertical centre of the frame is used.

    Returns
    -------
    int or None
        Y-coordinate of the detected reference line, or *fallback_y*.
    """
    if fallback_y is None:
        fallback_y = frame.shape[0] // 2

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=80,
        minLineLength=frame.shape[1] // 4,
        maxLineGap=20,
    )

    if lines is None:
        return fallback_y

    # Keep only near-horizontal lines (slope within ~5°)
    horizontal_ys = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        if dx > 1 and (dy / dx) < 0.09:  # ~5 degrees
            horizontal_ys.extend([y1, y2])

    if not horizontal_ys:
        return fallback_y

    return int(np.median(horizontal_ys))


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------


def compute_distances_from_landmarks(hand_landmarks, frame_width, frame_height,
                                     reference_y):
    """Return finger distances using correct width/height mapping.

    Parameters
    ----------
    hand_landmarks : list[mediapipe.tasks.components.containers.NormalizedLandmark]
        List of 21 normalised hand landmarks returned by
        ``HandLandmarkerResult.hand_landmarks[i]``.
    frame_width : int
        Width of the video frame in pixels.
    frame_height : int
        Height of the video frame in pixels.
    reference_y : int
        Y-coordinate (in pixels) of the horizontal reference line.

    Returns
    -------
    dict[str, tuple[int, int, int]]
        ``{finger_name: (pixel_x, pixel_y, distance)}``
    """
    distances = {}
    for name, landmark_id in FINGER_TIPS.items():
        lm = hand_landmarks[landmark_id]
        px = int(lm.x * frame_width)
        py = int(lm.y * frame_height)
        distances[name] = (px, py, abs(py - reference_y))
    return distances


def draw_overlay(frame, distances, reference_y):
    """Draw the reference line and per-finger distance labels on *frame*.

    Parameters
    ----------
    frame : numpy.ndarray
        BGR video frame (modified in-place).
    distances : dict[str, tuple[int, int, int]]
        Output of :func:`compute_distances_from_landmarks`.
    reference_y : int
        Y-coordinate of the horizontal reference line.
    """
    h, w = frame.shape[:2]
    # Draw the reference line (green)
    cv2.line(frame, (0, reference_y), (w, reference_y), (0, 255, 0), 2)

    for name, (px, py, dist) in distances.items():
        # Fingertip dot (red)
        cv2.circle(frame, (px, py), 6, (0, 0, 255), -1)
        # Distance label above the fingertip
        label = f"{name}: {dist}px"
        cv2.putText(frame, label, (px - 30, py - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                    cv2.LINE_AA)


def write_csv_row(writer, timestamp, distances):
    """Append one row of distance data to the CSV *writer*.

    Parameters
    ----------
    writer : csv.writer
        An open CSV writer.
    timestamp : float
        Elapsed seconds since tracking started.
    distances : dict[str, tuple[int, int, int]]
        Output of :func:`compute_distances_from_landmarks`.
    """
    row = [f"{timestamp:.4f}"]
    for name in FINGER_TIPS:
        _, _, dist = distances.get(name, (0, 0, 0))
        row.append(str(dist))
    writer.writerow(row)


# ---------------------------------------------------------------------------
# CLI & video helpers
# ---------------------------------------------------------------------------


def build_arg_parser():
    """Build and return the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Track hand landmarks over a guitar string reference line."
    )
    parser.add_argument(
        "--source", type=str, default="0",
        help="Video source: camera index (e.g. 0) or path to a video file. "
             "Default: 0 (webcam)."
    )
    parser.add_argument(
        "--csv-output", type=str, default="finger_distances.csv",
        help="Path for the output CSV file. Default: finger_distances.csv"
    )
    parser.add_argument(
        "--model-path", type=str, default=DEFAULT_MODEL_PATH,
        help="Path to the hand_landmarker.task model file. "
             "Downloaded automatically if missing."
    )
    parser.add_argument(
        "--max-hands", type=int, default=1, choices=[1, 2],
        help="Maximum number of hands to detect. Default: 1"
    )
    parser.add_argument(
        "--min-detection-confidence", type=float, default=0.7,
        help="Minimum hand detection confidence [0-1]. Default: 0.7"
    )
    parser.add_argument(
        "--min-tracking-confidence", type=float, default=0.5,
        help="Minimum tracking confidence [0-1]. Default: 0.5"
    )
    return parser


def open_video_source(source):
    """Open a video capture from *source* (camera index or file path).

    Returns
    -------
    cv2.VideoCapture
        The opened capture object.

    Raises
    ------
    RuntimeError
        If the source cannot be opened.
    """
    try:
        source = int(source)
    except ValueError:
        pass
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {source}")
    return cap


def ensure_model(model_path):
    """Download the hand-landmarker model if it does not exist locally.

    Parameters
    ----------
    model_path : str
        Desired local path for the ``.task`` model file.

    Returns
    -------
    str
        The validated *model_path*.

    Raises
    ------
    RuntimeError
        If the model cannot be downloaded.
    """
    if os.path.isfile(model_path):
        return model_path

    print(f"Downloading hand_landmarker.task to {model_path} …")
    try:
        ssl_ctx = ssl.create_default_context()
        req = urllib.request.Request(MODEL_URL)
        with urllib.request.urlopen(req, context=ssl_ctx) as resp, \
             open(model_path, "wb") as out:
            out.write(resp.read())
    except Exception as exc:
        # Clean up partial download
        if os.path.isfile(model_path):
            os.remove(model_path)
        raise RuntimeError(
            f"Failed to download model from {MODEL_URL}: {exc}\n"
            "Download it manually and pass --model-path."
        ) from exc
    return model_path


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def run(args=None):
    """Main tracking loop.

    Parameters
    ----------
    args : argparse.Namespace or None
        Parsed CLI arguments.  When *None*, arguments are parsed from
        ``sys.argv``.
    """
    if args is None:
        args = build_arg_parser().parse_args()

    model_path = ensure_model(args.model_path)

    cap = open_video_source(args.source)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Build HandLandmarker via the Tasks API
    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=args.max_hands,
        min_hand_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    )
    landmarker = HandLandmarker.create_from_options(options)

    csv_file = open(args.csv_output, "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(CSV_HEADER)

    start_time = time.time()
    prev_time = start_time
    fps = 0.0
    reference_y = frame_height // 2  # initial fallback
    _EMA_ALPHA = 0.1                 # smoothing factor for reference-line EMA

    print(f"CSV output: {args.csv_output}")
    print("Press 'q' to quit.\n")

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            current_time = time.time()
            elapsed = current_time - start_time
            timestamp_ms = int(elapsed * 1000)

            # Dynamically detect reference line via Hough transform and apply
            # an Exponential Moving Average to suppress frame-to-frame jitter.
            raw_y = detect_reference_line(frame, fallback_y=reference_y)
            reference_y = int(
                _EMA_ALPHA * raw_y + (1.0 - _EMA_ALPHA) * reference_y
            )

            # Convert BGR → RGB and wrap in mp.Image
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB, data=rgb
            )

            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            # FPS calculation (exponential moving average)
            dt = current_time - prev_time
            if dt > 0:
                instant_fps = 1.0 / dt
                fps = 0.9 * fps + 0.1 * instant_fps
            prev_time = current_time

            if result.hand_landmarks:
                for landmarks in result.hand_landmarks:
                    distances = compute_distances_from_landmarks(
                        landmarks, frame_width, frame_height, reference_y
                    )
                    draw_overlay(frame, distances, reference_y)
                    write_csv_row(writer, elapsed, distances)

            # Show FPS on frame
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
                        cv2.LINE_AA)

            cv2.imshow("Guitar Hand Tracker", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        csv_file.close()
        landmarker.close()
        cap.release()
        cv2.destroyAllWindows()

    print(f"\nTracking complete. Data saved to {args.csv_output}")


if __name__ == "__main__":
    run()
