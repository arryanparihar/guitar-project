# Guitar Hand Tracker

Track fingertip positions over a guitar string using OpenCV and MediaPipe.

## Features

- Detects index, middle, ring and pinky fingertips via the MediaPipe Tasks `HandLandmarker` API and the `hand_landmarker.task` model
- **Dynamic fretboard detection** – uses a Probabilistic Hough Line Transform (OpenCV) to automatically locate the guitar string reference line each frame
- Measures pixel distance from each fingertip to the detected reference line
- Saves timestamped distances to a CSV file for graphing "Finger Height" over time
- Overlays real-time distance labels on the video feed
- Optimised for **30+ FPS** to capture fast guitar playing

## Requirements

- Python 3.9+
- A webcam or video file

## Setup

```bash
pip install -r requirements.txt
```

The `hand_landmarker.task` model is **downloaded automatically** on first run. To supply your own copy, use the `--model-path` flag.

## Usage

**Webcam (default):**

```bash
python hand_tracker.py
```

**Video file:**

```bash
python hand_tracker.py --source video.mp4
```

**All options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--source` | `0` (webcam) | Camera index or video file path |
| `--csv-output` | `finger_distances.csv` | Output CSV path |
| `--model-path` | `hand_landmarker.task` | Path to the `.task` model file |
| `--max-hands` | `1` | Max hands to detect (1 or 2) |
| `--min-detection-confidence` | `0.7` | Detection confidence threshold |
| `--min-tracking-confidence` | `0.5` | Tracking confidence threshold |

Press **q** to stop tracking.

## CSV Output Format

| timestamp | Index_distance | Middle_distance | Ring_distance | Pinky_distance |
|-----------|----------------|-----------------|---------------|----------------|
| 0.0331 | 98 | 67 | 105 | 180 |

## Running Tests

```bash
python -m unittest test_hand_tracker -v
```