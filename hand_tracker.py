"""Hand landmark tracking for guitar playing analysis.

Uses OpenCV and MediaPipe to track fingertip positions relative to a
horizontal reference line (guitar string) and logs distances to CSV.
"""

import argparse
import csv
import time

import cv2
import mediapipe as mp

# MediaPipe hand landmark indices for each fingertip
FINGER_TIPS = {
    "Thumb": mp.solutions.hands.HandLandmark.THUMB_TIP,
    "Index": mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP,
    "Middle": mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP,
    "Ring": mp.solutions.hands.HandLandmark.RING_FINGER_TIP,
    "Pinky": mp.solutions.hands.HandLandmark.PINKY_TIP,
}

CSV_HEADER = ["timestamp"] + [f"{name}_distance" for name in FINGER_TIPS]


def compute_distances_from_landmarks(hand_landmarks, frame_width, frame_height,
                                     reference_y):
    """Return finger distances using correct width/height mapping.

    Parameters
    ----------
    hand_landmarks : mediapipe NormalizedLandmarkList
        The ``.landmark`` attribute of a detected hand.
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
        lm = hand_landmarks.landmark[landmark_id]
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
        "--ref-y", type=int, default=None,
        help="Y-coordinate for the reference line in pixels. "
             "Default: middle of the frame."
    )
    parser.add_argument(
        "--csv-output", type=str, default="finger_distances.csv",
        help="Path for the output CSV file. Default: finger_distances.csv"
    )
    parser.add_argument(
        "--max-hands", type=int, default=1, choices=[1, 2],
        help="Maximum number of hands to detect. Default: 1"
    )
    parser.add_argument(
        "--min-detection-confidence", type=float, default=0.7,
        help="Minimum detection confidence [0-1]. Default: 0.7"
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
    # Try to interpret as an integer camera index
    try:
        source = int(source)
    except ValueError:
        pass
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {source}")
    return cap


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

    cap = open_video_source(args.source)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    reference_y = args.ref_y if args.ref_y is not None else frame_height // 2

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=args.max_hands,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    )

    csv_file = open(args.csv_output, "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(CSV_HEADER)

    start_time = time.time()
    prev_time = start_time
    fps = 0.0

    print(f"Tracking started – reference line at y={reference_y}")
    print(f"CSV output: {args.csv_output}")
    print("Press 'q' to quit.\n")

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert BGR -> RGB for MediaPipe
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            current_time = time.time()
            elapsed = current_time - start_time

            # FPS calculation (exponential moving average)
            dt = current_time - prev_time
            if dt > 0:
                instant_fps = 1.0 / dt
                fps = 0.9 * fps + 0.1 * instant_fps
            prev_time = current_time

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    distances = compute_distances_from_landmarks(
                        hand_landmarks, frame_width, frame_height, reference_y
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
        hands.close()
        cap.release()
        cv2.destroyAllWindows()

    print(f"\nTracking complete. Data saved to {args.csv_output}")


if __name__ == "__main__":
    run()
