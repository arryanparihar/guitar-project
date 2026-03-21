"""
audio_sync.py
=============
AudioSyncer: maps AudioEngine onsets to TabParser time slices for the
Guitar Tab Player.

Responsibilities
----------------
* Accept raw audio bytes (WAV / MP3) from a Streamlit file uploader.
* Run librosa-based onset detection via the existing AudioEngine.
* Map detected onset times 1-to-1 onto TabParser time slices with an
  optional global offset, handling mismatched list lengths gracefully.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from typing import List, Optional

from audio_engine import AudioEngine


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

@dataclass
class SyncedSlice:
    """A tab time-slice paired with its linked audio onset.

    Attributes
    ----------
    slice_index:
        0-based position of this slice in the original TabParser output list.
    time_index:
        Column index (from TabParser) representing the horizontal position
        of this slice in the raw tab text.  Can be mutated during Sync
        Refinement Mode to re-link an onset to a different tab column.
    onset_time_sec:
        The audio onset time (seconds) linked to this slice, or ``None``
        when there are fewer onsets than tab slices.
    """

    slice_index: int
    time_index: int
    onset_time_sec: Optional[float]


# ---------------------------------------------------------------------------
# AudioSyncer
# ---------------------------------------------------------------------------

class AudioSyncer:
    """Extract audio onsets and map them 1-to-1 to TabParser time slices.

    Parameters
    ----------
    bpm : float
        Target BPM forwarded to the underlying :class:`AudioEngine` for
        grid-alignment and deviation scoring.
    """

    #: Hard upper limit on accepted audio payloads (50 MiB).
    _MAX_FILE_BYTES: int = 52_428_800

    def __init__(self, bpm: float = 120.0) -> None:
        if bpm <= 0:
            raise ValueError(
                f"BPM must be positive (typical range: 40–240), got {bpm}."
            )
        self._engine: AudioEngine = AudioEngine(bpm=bpm)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyse_bytes(
        self,
        audio_bytes: bytes,
        suffix: str = ".wav",
    ) -> List[float]:
        """Detect onsets in raw audio bytes.

        Parameters
        ----------
        audio_bytes : bytes
            Raw contents of a WAV or MP3 file.
        suffix : str
            File extension used when writing to a temporary file
            (e.g. ``".mp3"`` or ``".wav"``).

        Returns
        -------
        List[float]
            Onset times in seconds, ascending order.

        Raises
        ------
        ValueError
            If *audio_bytes* exceeds the 50 MiB size limit.
        """
        if len(audio_bytes) > self._MAX_FILE_BYTES:
            size_mb = len(audio_bytes) / 1_048_576
            raise ValueError(
                f"Audio file too large ({size_mb:.1f} MB). "
                "Maximum supported size is 50 MB."
            )

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        try:
            result = self._engine.analyse_file(tmp_path)
        finally:
            os.unlink(tmp_path)

        return result.onset_times

    @staticmethod
    def map_to_slices(
        onset_times: List[float],
        tab_slices: list,
        global_offset_ms: float = 0.0,
    ) -> List[SyncedSlice]:
        """Pair onset times with tab slices 1-to-1.

        Handles mismatched list lengths:

        * **More onsets than slices** – excess onsets are silently dropped.
        * **More slices than onsets** – surplus slices receive
          ``onset_time_sec=None``.

        Parameters
        ----------
        onset_times : List[float]
            Raw onset times in seconds (ascending), e.g. from
            :meth:`analyse_bytes`.
        tab_slices : list
            Output of :meth:`TabParser.parse` – a list of
            ``{"time_index": int, "notes": [...]}`` dicts.
        global_offset_ms : float
            Constant offset added to every onset time before mapping.
            Positive values shift onsets later; negative shifts earlier.

        Returns
        -------
        List[SyncedSlice]
            One :class:`SyncedSlice` per tab slice, in order.
        """
        offset_sec = global_offset_ms / 1000.0
        adjusted: List[float] = [t + offset_sec for t in onset_times]

        return [
            SyncedSlice(
                slice_index=i,
                time_index=slc["time_index"],
                onset_time_sec=adjusted[i] if i < len(adjusted) else None,
            )
            for i, slc in enumerate(tab_slices)
        ]
