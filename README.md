# Guitar Hand Tracker

Track fingertip positions over a guitar string using OpenCV and MediaPipe.

## Features

- Detects thumb, index, middle, ring and pinky fingertips via MediaPipe Hands
- Measures pixel distance from each fingertip to a configurable horizontal reference line (guitar string)
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

## Usage

**Webcam (default):**

```bash
python hand_tracker.py
```

**Video file with custom reference line:**

```bash
python hand_tracker.py --source video.mp4 --ref-y 300
```

**All options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--source` | `0` (webcam) | Camera index or video file path |
| `--ref-y` | frame midpoint | Y-coordinate of the reference line (pixels) |
| `--csv-output` | `finger_distances.csv` | Output CSV path |
| `--max-hands` | `1` | Max hands to detect (1 or 2) |
| `--min-detection-confidence` | `0.7` | Detection confidence threshold |
| `--min-tracking-confidence` | `0.5` | Tracking confidence threshold |

Press **q** to stop tracking.

## CSV Output Format

| timestamp | Thumb_distance | Index_distance | Middle_distance | Ring_distance | Pinky_distance |
|-----------|---------------|----------------|-----------------|---------------|----------------|
| 0.0331 | 142 | 98 | 67 | 105 | 180 |

## Running Tests

```bash
python -m unittest test_hand_tracker -v
```