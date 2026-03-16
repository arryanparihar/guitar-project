"""Unit tests for hand_tracker core logic.

These tests exercise the pure-computation helpers without requiring a
camera or GPU – MediaPipe landmarks are replaced by lightweight stubs.
"""

import csv
import io
import types
import unittest

import hand_tracker


# ---------------------------------------------------------------------------
# Helpers – lightweight landmark stubs
# ---------------------------------------------------------------------------

class _FakeLandmark:
    """Mimics a single mediapipe NormalizedLandmark."""

    def __init__(self, x: float, y: float, z: float = 0.0):
        self.x = x
        self.y = y
        self.z = z


def _make_hand_landmarks(tip_positions):
    """Build a stub ``hand_landmarks`` object.

    Parameters
    ----------
    tip_positions : dict[int, tuple[float, float]]
        Mapping from landmark index to ``(norm_x, norm_y)`` values in
        the [0, 1] range.
    """
    # MediaPipe hands have 21 landmarks (indices 0-20).
    landmarks = [_FakeLandmark(0.0, 0.0)] * 21

    # Replace the entries we care about.
    landmarks = list(landmarks)  # ensure mutable list
    for idx, (nx, ny) in tip_positions.items():
        landmarks[idx] = _FakeLandmark(nx, ny)

    obj = types.SimpleNamespace()
    obj.landmark = landmarks
    return obj


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestComputeDistances(unittest.TestCase):
    """Tests for compute_distances_from_landmarks."""

    def test_all_fingers_on_reference_line(self):
        """Distance should be 0 when fingertips sit exactly on the line."""
        ref_y = 240
        frame_w, frame_h = 640, 480
        norm_y = ref_y / frame_h  # 0.5

        positions = {
            4: (0.1, norm_y),   # Thumb
            8: (0.3, norm_y),   # Index
            12: (0.5, norm_y),  # Middle
            16: (0.7, norm_y),  # Ring
            20: (0.9, norm_y),  # Pinky
        }
        hand = _make_hand_landmarks(positions)
        result = hand_tracker.compute_distances_from_landmarks(
            hand, frame_w, frame_h, ref_y
        )

        for name, (_px, _py, dist) in result.items():
            self.assertEqual(dist, 0, f"{name} should be 0")

    def test_known_distances(self):
        """Check pixel distances against hand-calculated values."""
        ref_y = 200
        frame_w, frame_h = 640, 480

        positions = {
            4: (0.5, 100 / 480),   # Thumb at y=100 → dist=100
            8: (0.5, 300 / 480),   # Index at y=300 → dist=100
            12: (0.5, 200 / 480),  # Middle at y=200 → dist=0
            16: (0.5, 50 / 480),   # Ring at y=50 → dist=150
            20: (0.5, 400 / 480),  # Pinky at y=400 → dist=200
        }
        hand = _make_hand_landmarks(positions)
        result = hand_tracker.compute_distances_from_landmarks(
            hand, frame_w, frame_h, ref_y
        )

        expected = {
            "Thumb": 100,
            "Index": 100,
            "Middle": 0,
            "Ring": 150,
            "Pinky": 200,
        }
        for name, exp_dist in expected.items():
            _, _, dist = result[name]
            self.assertAlmostEqual(dist, exp_dist, delta=1,
                                   msg=f"{name} distance mismatch")

    def test_returns_all_five_fingers(self):
        """Result should contain exactly the five expected finger names."""
        positions = {4: (0.5, 0.5), 8: (0.5, 0.5), 12: (0.5, 0.5),
                     16: (0.5, 0.5), 20: (0.5, 0.5)}
        hand = _make_hand_landmarks(positions)
        result = hand_tracker.compute_distances_from_landmarks(
            hand, 640, 480, 240
        )
        self.assertEqual(set(result.keys()),
                         {"Thumb", "Index", "Middle", "Ring", "Pinky"})

    def test_pixel_coordinates(self):
        """Pixel x/y should be landmark * frame dimension."""
        frame_w, frame_h = 640, 480
        positions = {
            4: (0.25, 0.75),
            8: (0.25, 0.75),
            12: (0.25, 0.75),
            16: (0.25, 0.75),
            20: (0.25, 0.75),
        }
        hand = _make_hand_landmarks(positions)
        result = hand_tracker.compute_distances_from_landmarks(
            hand, frame_w, frame_h, 0
        )
        for _, (px, py, _) in result.items():
            self.assertEqual(px, int(0.25 * frame_w))
            self.assertEqual(py, int(0.75 * frame_h))


class TestWriteCsvRow(unittest.TestCase):
    """Tests for write_csv_row."""

    def _write_one_row(self, timestamp, distances):
        buf = io.StringIO()
        writer = csv.writer(buf)
        hand_tracker.write_csv_row(writer, timestamp, distances)
        buf.seek(0)
        return list(csv.reader(buf))[0]

    def test_column_order_matches_header(self):
        distances = {
            "Thumb": (0, 0, 10),
            "Index": (0, 0, 20),
            "Middle": (0, 0, 30),
            "Ring": (0, 0, 40),
            "Pinky": (0, 0, 50),
        }
        row = self._write_one_row(1.2345, distances)
        self.assertEqual(row[0], "1.2345")
        # Values must follow the FINGER_TIPS key order
        expected_dists = [str(d) for _, d in
                          zip(hand_tracker.FINGER_TIPS,
                              [10, 20, 30, 40, 50])]
        self.assertEqual(row[1:], expected_dists)

    def test_missing_finger_defaults_to_zero(self):
        """If a finger is absent from *distances*, distance should be 0."""
        row = self._write_one_row(0.0, {})
        self.assertTrue(all(v == "0" for v in row[1:]))


class TestCsvHeader(unittest.TestCase):
    """Verify the CSV header matches expectations."""

    def test_header_structure(self):
        self.assertEqual(hand_tracker.CSV_HEADER[0], "timestamp")
        self.assertEqual(len(hand_tracker.CSV_HEADER),
                         1 + len(hand_tracker.FINGER_TIPS))

    def test_header_finger_names(self):
        for name in hand_tracker.FINGER_TIPS:
            self.assertIn(f"{name}_distance", hand_tracker.CSV_HEADER)


class TestBuildArgParser(unittest.TestCase):
    """Smoke-test the CLI argument parser."""

    def test_defaults(self):
        parser = hand_tracker.build_arg_parser()
        args = parser.parse_args([])
        self.assertEqual(args.source, "0")
        self.assertIsNone(args.ref_y)
        self.assertEqual(args.csv_output, "finger_distances.csv")
        self.assertEqual(args.max_hands, 1)
        self.assertAlmostEqual(args.min_detection_confidence, 0.7)
        self.assertAlmostEqual(args.min_tracking_confidence, 0.5)

    def test_custom_values(self):
        parser = hand_tracker.build_arg_parser()
        args = parser.parse_args([
            "--source", "video.mp4",
            "--ref-y", "300",
            "--csv-output", "out.csv",
            "--max-hands", "2",
        ])
        self.assertEqual(args.source, "video.mp4")
        self.assertEqual(args.ref_y, 300)
        self.assertEqual(args.csv_output, "out.csv")
        self.assertEqual(args.max_hands, 2)


if __name__ == "__main__":
    unittest.main()
