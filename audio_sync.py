"""
audio_sync.py
=============
Production-grade module for extracting precise note timestamps from
acoustic fingerstyle guitar recordings.

Typical usage::

    syncer = AudioSyncer()
    timestamps = syncer.extract_onsets("my_recording.wav")
    print(timestamps)
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import librosa
import numpy as np
from scipy.ndimage import median_filter


class AudioSyncer:
    """Extracts precise note-onset timestamps from guitar audio recordings.

    Uses a spectral-flux onset detector tuned for the high dynamic range
    and wide tonal palette of fingerstyle guitar.

    Args:
        sample_rate: Target sample rate used when loading audio (Hz).
            Defaults to 22050.
        hop_length: Hop length in samples for short-time analyses.
            Smaller values yield finer time resolution at higher CPU cost.
            Defaults to 512.
        wait: Minimum number of frames between successive onsets.
            Helps suppress double-triggering on percussive thumb slaps.
            Defaults to 8.
        pre_max: Number of frames before a candidate peak that must all
            be lower than the peak (used by ``librosa.onset.onset_detect``
            for peak-picking).  Increasing this value makes the detector
            less sensitive to fast transients and avoids false positives
            on sustain tails.  Defaults to 3.
        median_filter_size: Window size (in frames) for median filtering
            applied to the onset-strength envelope before peak-picking.
            Filters out slow resonance/sustain energy that would otherwise
            produce spurious detections.  Defaults to 3.
    """

    def __init__(
        self,
        sample_rate: int = 22050,
        hop_length: int = 512,
        wait: int = 8,
        pre_max: int = 3,
        median_filter_size: int = 3,
    ) -> None:
        if sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {sample_rate}")
        if hop_length <= 0:
            raise ValueError(f"hop_length must be positive, got {hop_length}")
        if wait < 0:
            raise ValueError(f"wait must be non-negative, got {wait}")
        if pre_max < 0:
            raise ValueError(f"pre_max must be non-negative, got {pre_max}")
        if median_filter_size < 1:
            raise ValueError(
                f"median_filter_size must be at least 1, got {median_filter_size}"
            )

        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.wait = wait
        self.pre_max = pre_max
        self.median_filter_size = median_filter_size

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_onsets(self, file_path: str) -> List[float]:
        """Extract note-onset timestamps from an MP3 or WAV recording.

        The pipeline:

        1. Loads the file at ``self.sample_rate`` Hz, mixing down to mono.
        2. Applies peak normalisation so that the loudest sample reaches
           ±1.0 – this keeps the onset-strength envelope on a consistent
           scale regardless of the recording's overall gain.
        3. Computes a spectral-flux onset-strength envelope over the full
           Mel spectrogram.
        4. Applies a median filter to the envelope to suppress slow
           sustain/resonance energy.
        5. Runs ``librosa.onset.onset_detect`` with ``backtrack=True`` so
           that each onset is snapped back to the nearest preceding local
           minimum – giving sample-accurate attack times rather than
           peak-centred estimates.
        6. Converts detected frames to seconds and returns them in
           ascending order.

        Args:
            file_path: Path to an MP3 or WAV audio file.

        Returns:
            A chronologically sorted list of onset timestamps in seconds.

        Raises:
            FileNotFoundError: If *file_path* does not point to an
                existing file.
            audioread.exceptions.NoBackendError: If no audio decoding
                backend (e.g. ffmpeg) is available to read the file.
        """
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(
                f"Audio file not found: '{file_path}'"
            )

        # --- Step 1: Load and mix to mono --------------------------------
        y, sr = librosa.load(str(path), sr=self.sample_rate, mono=True)

        # --- Step 2: Peak normalisation ----------------------------------
        peak = float(np.abs(y).max())
        if peak > 1e-9:
            y = y / peak

        # --- Step 3: Spectral-flux onset-strength envelope ---------------
        onset_env: np.ndarray = librosa.onset.onset_strength(
            y=y,
            sr=sr,
            hop_length=self.hop_length,
            aggregate=np.median,
        )

        # --- Step 4: Median filtering to suppress sustain/resonance ------
        if self.median_filter_size > 1:
            onset_env = median_filter(
                onset_env,
                size=self.median_filter_size,
                mode="nearest",
            )

        # --- Step 5: Peak-pick with backtracking -------------------------
        onset_frames: np.ndarray = librosa.onset.onset_detect(
            onset_envelope=onset_env,
            sr=sr,
            hop_length=self.hop_length,
            backtrack=True,
            wait=self.wait,
            pre_max=self.pre_max,
            units="frames",
        )

        # --- Step 6: Convert to seconds and sort -------------------------
        onset_times: List[float] = sorted(
            librosa.frames_to_time(
                onset_frames,
                sr=sr,
                hop_length=self.hop_length,
            ).tolist()
        )

        return onset_times


# ---------------------------------------------------------------------------
# Verification entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    target = "test_cover.wav"
    syncer = AudioSyncer()

    try:
        timestamps = syncer.extract_onsets(target)
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] Could not process '{target}': {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Total notes detected: {len(timestamps)}")
    print("First 10 timestamps (seconds):")
    for ts in timestamps[:10]:
        print(f"  {ts:.3f}")
