"""
audio_engine.py
===============
Audio analysis module for SyncopateAI.

Responsibilities
----------------
* Load audio from a file (wav, mp3, etc.) using Librosa.
* Detect note *onsets* (the moment a note starts).
* Compare the detected onsets against a user-defined BPM to compute a
  per-onset *tempo deviation* in milliseconds.
* Return structured data that the dashboard (main.py) can plot.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import librosa
import numpy as np


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class OnsetEvent:
    """A single detected note onset."""

    index: int           # sequential onset number (0-based)
    time_sec: float      # onset time in seconds
    expected_sec: float  # expected time based on BPM grid
    deviation_ms: float  # (time_sec − expected_sec) × 1000  (+ = late, − = early)


@dataclass
class AudioAnalysisResult:
    """Full result of analysing one audio file / stream."""

    sample_rate: int
    duration_sec: float
    bpm_target: float
    bpm_detected: Optional[float]       # as estimated by librosa.beat.beat_track
    onset_times: list[float]            # raw onset times (seconds)
    onset_events: list[OnsetEvent]      # aligned events with deviation
    rms_envelope: list[float]           # per-frame RMS (normalised 0–1)
    rms_times: list[float]              # corresponding time stamps for rms_envelope
    avg_deviation_ms: float             # mean absolute deviation (ms)
    timing_score: float                 # 0–100: 100 = perfect timing


# ---------------------------------------------------------------------------
# Core engine
# ---------------------------------------------------------------------------

class AudioEngine:
    """
    Analyses audio to detect note onsets and compare them against a
    target BPM metronome grid.

    Parameters
    ----------
    bpm : float
        Target tempo in beats per minute (the "metronome" setting).
    hop_length : int
        Hop length (in samples) used for librosa's short-time analyses.
        Smaller values give finer time resolution but are slower.
    """

    def __init__(self, bpm: float = 120.0, hop_length: int = 512) -> None:
        if bpm <= 0:
            raise ValueError(f"BPM must be positive, got {bpm}")
        self.bpm = float(bpm)
        self.hop_length = hop_length

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyse_file(self, audio_path: str) -> AudioAnalysisResult:
        """
        Load *audio_path* and run the full analysis pipeline.

        Parameters
        ----------
        audio_path : str
            Path to a supported audio file (wav, mp3, flac, ogg, …).

        Returns
        -------
        AudioAnalysisResult
        """
        y, sr = librosa.load(audio_path, sr=None, mono=True)
        return self._analyse(y, sr)

    def analyse_array(
        self,
        y: np.ndarray,
        sr: int,
    ) -> AudioAnalysisResult:
        """
        Run the full analysis pipeline on an in-memory audio array.

        Parameters
        ----------
        y : np.ndarray
            Mono audio time-series (float32 or float64).
        sr : int
            Sample rate of *y*.

        Returns
        -------
        AudioAnalysisResult
        """
        return self._analyse(y, int(sr))

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _analyse(self, y: np.ndarray, sr: int) -> AudioAnalysisResult:
        duration_sec = librosa.get_duration(y=y, sr=sr)

        # --- Onset detection (harmonic-aware) --------------------------
        onset_strength = self._harmonic_onset_strength(y, sr)
        onset_frames = librosa.onset.onset_detect(
            onset_envelope=onset_strength,
            sr=sr,
            hop_length=self.hop_length,
            units="frames",
        )
        onset_times = librosa.frames_to_time(
            onset_frames, sr=sr, hop_length=self.hop_length
        ).tolist()

        # --- BPM estimation --------------------------------------------
        tempo, _ = librosa.beat.beat_track(
            y=y, sr=sr, hop_length=self.hop_length
        )
        # librosa ≥ 0.10 returns an ndarray; older versions return a scalar
        tempo_arr = np.asarray(tempo).ravel()
        bpm_detected = float(tempo_arr[0]) if len(tempo_arr) > 0 else None

        # --- Align onsets to BPM grid ----------------------------------
        beat_period_sec = 60.0 / self.bpm
        onset_events = self._align_onsets(onset_times, beat_period_sec)

        # --- RMS energy envelope (for waveform display) ----------------
        rms = librosa.feature.rms(y=y, hop_length=self.hop_length)[0]
        rms_norm = (rms / (rms.max() + 1e-9)).tolist()
        rms_times = librosa.frames_to_time(
            np.arange(len(rms)), sr=sr, hop_length=self.hop_length
        ).tolist()

        # --- Aggregate metrics -----------------------------------------
        if onset_events:
            deviations = [abs(ev.deviation_ms) for ev in onset_events]
            avg_dev = float(np.mean(deviations))
        else:
            avg_dev = 0.0

        timing_score = self._timing_score(avg_dev)

        return AudioAnalysisResult(
            sample_rate=sr,
            duration_sec=duration_sec,
            bpm_target=self.bpm,
            bpm_detected=bpm_detected,
            onset_times=onset_times,
            onset_events=onset_events,
            rms_envelope=rms_norm,
            rms_times=rms_times,
            avg_deviation_ms=avg_dev,
            timing_score=timing_score,
        )

    def _harmonic_onset_strength(self, y: np.ndarray, sr: int) -> np.ndarray:
        """
        Compute a combined onset strength envelope sensitive to guitar harmonics.

        Guitar harmonics lack a strong broadband transient but produce a
        concentrated spike in high-frequency energy.  This method:

        1. Builds a full Mel spectrogram.
        2. Restricts it to Mel bins whose centre frequency is above 2000 Hz
           (the high-frequency band where harmonics bloom).
        3. Computes spectral flux (``librosa.onset.onset_strength``) over those
           bins alone.
        4. Normalises the high-frequency flux to [0, 1].
        5. Normalises the per-frame RMS energy to [0, 1].
        6. Returns the element-wise mean of the two normalised signals so that
           both broadband energy changes *and* harmonic blooms contribute to
           onset detection.

        Parameters
        ----------
        y : np.ndarray
            Mono audio time-series.
        sr : int
            Sample rate.

        Returns
        -------
        np.ndarray
            1-D onset strength envelope, one value per analysis frame.
        """
        n_mels = 128

        # --- High-frequency Mel spectrogram flux ----------------------------
        mel_spec = librosa.feature.melspectrogram(
            y=y, sr=sr, hop_length=self.hop_length, n_mels=n_mels
        )
        mel_freqs = librosa.mel_frequencies(n_mels=n_mels, fmin=0.0, fmax=sr / 2.0)
        hf_mask = mel_freqs > 2000.0
        hf_mel_spec = mel_spec[hf_mask, :]  # (n_hf_bins, n_frames)

        hf_onset_env = librosa.onset.onset_strength(
            S=librosa.power_to_db(hf_mel_spec),
            sr=sr,
            hop_length=self.hop_length,
        )

        # --- RMS energy envelope -------------------------------------------
        rms = librosa.feature.rms(y=y, hop_length=self.hop_length)[0]

        # Align lengths (onset_strength may produce a different frame count)
        min_len = min(len(hf_onset_env), len(rms))
        hf_onset_env = hf_onset_env[:min_len]
        rms = rms[:min_len]

        # --- Normalise both signals to [0, 1] and combine ------------------
        def _norm(x: np.ndarray) -> np.ndarray:
            peak = float(x.max())
            if peak < 1e-9:
                return np.zeros_like(x)
            return x / peak

        combined = (_norm(hf_onset_env) + _norm(rms)) / 2.0
        return combined

    @staticmethod
    def _align_onsets(
        onset_times: list[float],
        beat_period_sec: float,
    ) -> list[OnsetEvent]:
        """
        Snap each onset to the nearest beat in a uniform metronome grid
        and return the deviation from that beat.

        The first onset is used as the grid anchor (phase offset).
        """
        if not onset_times:
            return []

        anchor = onset_times[0]  # treat the first onset as beat 1
        events: list[OnsetEvent] = []

        for i, t in enumerate(onset_times):
            # How many beats since the anchor?
            beats_since_anchor = (t - anchor) / beat_period_sec
            # Round to nearest integer beat
            nearest_beat = round(beats_since_anchor)
            expected_sec = anchor + nearest_beat * beat_period_sec
            deviation_ms = (t - expected_sec) * 1000.0

            events.append(
                OnsetEvent(
                    index=i,
                    time_sec=t,
                    expected_sec=expected_sec,
                    deviation_ms=deviation_ms,
                )
            )

        return events

    @staticmethod
    def _timing_score(avg_deviation_ms: float, max_deviation_ms: float = 200.0) -> float:
        """
        Map average absolute deviation (ms) to a 0–100 timing score.

        0 ms deviation → 100 (perfect).
        ≥ *max_deviation_ms* → 0.

        The threshold of 200 ms is grounded in music-perception research:
        deviations beyond ~200 ms are clearly audible and considered
        unacceptable timing errors at most practice tempos.
        """
        clamped = min(avg_deviation_ms, max_deviation_ms)
        return round((1.0 - clamped / max_deviation_ms) * 100.0, 2)
