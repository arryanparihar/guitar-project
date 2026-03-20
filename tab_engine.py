"""
tab_engine.py
=============
Production-grade ASCII guitar tablature parser for SyncopateAI.

Responsibilities
----------------
* Detect valid string lines from raw, messy tab text (dynamic tuning support).
* Parse each string line in a single pass using ``re.finditer`` – never
  ``str.split('-')`` – so that multi-character tokens (``<12>``, ``[12]``,
  ``12``) are anchored to their first character's column index, preserving
  cross-string time alignment.
* Classify every token into one of the recognised fingerstyle techniques.
* Return a chronologically sorted list of "time slices", where each slice
  groups all notes that start at the same column index.
"""

from __future__ import annotations

import json
import re
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Compiled patterns
# ---------------------------------------------------------------------------

# Ordered from most-specific to least-specific so that multi-character tokens
# are consumed before their constituent digits can be re-matched.
_TOKEN_RE = re.compile(
    r"<\d+>"        # natural harmonic  – e.g. <12>  or <7>
    r"|\[\d+\]"     # artificial harm.  – e.g. [12]
    r"|\(\d+\)"     # artificial harm.  – e.g. (12)
    r"|\d{1,2}"     # fret number 0-24  – 1 or 2 digits
    r"|[xX]"        # dead note / slap
    r"|[hp]"        # hammer-on / pull-off
    r"|[/\\]"       # slide up / slide down
)

# A valid tab string line must:
#   – start with optional whitespace
#   – be followed by a string name (a letter, optionally with # or b suffix)
#   – be immediately followed by a pipe character '|'
#   – have tab content (after the pipe) that contains at least one dash,
#     which distinguishes it from chord-name lines like "Am|chord info"
_STRING_LINE_RE = re.compile(
    r"^(?P<ws>\s*)(?P<name>[A-Za-z][#b]?)\|(?P<content>.*)$"
)


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

# A single note within a time slice
NoteDict = Dict[str, str]

# One time slice
SliceDict = Dict  # {"time_index": int, "notes": List[NoteDict]}


# ---------------------------------------------------------------------------
# TabParser
# ---------------------------------------------------------------------------

class TabParser:
    """Parse raw ASCII guitar tablature into a chronological list of time slices.

    Usage::

        parser = TabParser()
        slices = parser.parse(raw_tab_string)
    """

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def parse(self, tab_string: str) -> List[SliceDict]:
        """Parse raw ASCII tab text into a list of time slices.

        Parameters
        ----------
        tab_string:
            Multi-line string containing the raw tab.  Lyrics, chord names,
            blank lines and other non-tab content are silently ignored.

        Returns
        -------
        List[SliceDict]
            A chronologically sorted list of dicts, each of the form::

                {
                    "time_index": 5,
                    "notes": [
                        {"string": "A", "fret": "0",  "technique": "standard"},
                        {"string": "D", "fret": "x",  "technique": "slap"},
                    ]
                }
        """
        lines: List[str] = tab_string.splitlines()
        blocks: List[List[_StringLine]] = self._group_tab_blocks(lines)

        # Accumulate notes keyed by their column (= time) index
        time_map: Dict[int, List[NoteDict]] = {}
        for block in blocks:
            self._parse_block(block, time_map)

        # Build and return the sorted list of time slices
        return [
            {"time_index": idx, "notes": notes}
            for idx, notes in sorted(time_map.items())
        ]

    # ------------------------------------------------------------------
    # Block grouping
    # ------------------------------------------------------------------

    def _group_tab_blocks(
        self, lines: List[str]
    ) -> List[List["_StringLine"]]:
        """Group consecutive tab string lines into blocks.

        Non-tab lines (lyrics, blank lines, chord names) break the current
        block, so multi-system tabs produce one block per system.

        Returns
        -------
        List of blocks, each block being a list of ``_StringLine`` named
        tuples.
        """
        blocks: List[List[_StringLine]] = []
        current: List[_StringLine] = []

        for line in lines:
            parsed = self._parse_string_line(line)
            if parsed is not None:
                current.append(parsed)
            else:
                if current:
                    blocks.append(current)
                    current = []

        if current:
            blocks.append(current)

        return blocks

    def _parse_string_line(self, line: str) -> Optional["_StringLine"]:
        """Attempt to parse *line* as a tab string line.

        Returns a ``_StringLine`` on success, ``None`` otherwise.

        A valid string line must have tab content (after ``|``) that contains
        at least one ``-`` character; this filters out chord-name lines such as
        ``Am|something`` which also start with a letter followed by ``|``.
        """
        m = _STRING_LINE_RE.match(line)
        if m is None:
            return None

        content: str = m.group("content")
        # Require at least one dash – the backbone of any tab line.
        if "-" not in content:
            return None

        return _StringLine(
            name=m.group("name"),
            content=content,
        )

    # ------------------------------------------------------------------
    # Block / string parsing
    # ------------------------------------------------------------------

    def _parse_block(
        self,
        block: List["_StringLine"],
        time_map: Dict[int, List[NoteDict]],
    ) -> None:
        """Parse every string line in *block* and accumulate into *time_map*."""
        for sl in block:
            self._parse_string(sl, time_map)

    def _parse_string(
        self,
        sl: "_StringLine",
        time_map: Dict[int, List[NoteDict]],
    ) -> None:
        """Single-pass scan of one string line using ``re.finditer``.

        ``match.start()`` gives the column offset within the tab content, which
        acts as the time index for cross-string alignment.  Multi-character
        tokens (``<12>``, ``[12]``, ``12``) are consumed whole, so their
        trailing characters never shift the alignment of later tokens.
        """
        for match in _TOKEN_RE.finditer(sl.content):
            token: str = match.group()
            time_index: int = match.start()

            fret, technique = self._classify_token(token)
            note: NoteDict = {
                "string": sl.name,
                "fret": fret,
                "technique": technique,
            }

            if time_index not in time_map:
                time_map[time_index] = []
            time_map[time_index].append(note)

    # ------------------------------------------------------------------
    # Token classification
    # ------------------------------------------------------------------

    def _classify_token(self, token: str) -> Tuple[str, str]:
        """Classify a raw token into ``(fret, technique)``.

        Parameters
        ----------
        token:
            The raw matched string, e.g. ``"<12>"``, ``"[7]"``, ``"12"``,
            ``"x"``, ``"h"``, etc.

        Returns
        -------
        Tuple[str, str]
            A ``(fret_str, technique_str)`` pair.
        """
        # Natural harmonic  <12>  or  <7>
        if token.startswith("<") and token.endswith(">"):
            return (token[1:-1], "natural_harmonic")

        # Artificial harmonic  [12]  or  (12)
        if (
            (token.startswith("[") and token.endswith("]"))
            or (token.startswith("(") and token.endswith(")"))
        ):
            return (token[1:-1], "artificial_harmonic")

        # Dead note / slap
        if token in ("x", "X"):
            return (token, "slap")

        # Hammer-on
        if token == "h":
            return (token, "hammer_on")

        # Pull-off
        if token == "p":
            return (token, "pull_off")

        # Slide up
        if token == "/":
            return (token, "slide_up")

        # Slide down
        if token == "\\":
            return (token, "slide_down")

        # Standard fret number (1-2 digit decimal)
        if re.fullmatch(r"\d{1,2}", token):
            return (token, "standard")

        # Fallback – should not be reached with the current _TOKEN_RE
        return (token, "unknown")  # pragma: no cover


# ---------------------------------------------------------------------------
# Internal data container
# ---------------------------------------------------------------------------

class _StringLine:
    """Lightweight container for a parsed tab string line."""

    __slots__ = ("name", "content")

    def __init__(self, name: str, content: str) -> None:
        self.name: str = name          # e.g. "e", "A", "F#"
        self.content: str = content    # everything after the '|'


# ---------------------------------------------------------------------------
# Demonstration / smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # A complex, multi-technique tablature that exercises every feature:
    #
    #  Column 2  : bass note (A-string 0) and slap (E-string x) simultaneously
    #  Column 10 : natural harmonic <12> on e-string
    #  Column 11 : double-digit fret 12 on B-string (different column from above)
    #  Column 21 : artificial harmonic [12] on e-string
    #
    SAMPLE_TAB = (
        "e|--0-------<12>-------[12]-|\n"
        "B|--1--------12---------1---|\n"
        "G|--0---------0---------0---|\n"
        "D|--2---------2---------2---|\n"
        "A|--0---------0---------0---|\n"
        "E|--x---------x---------x---|\n"
    )

    parser = TabParser()
    result = parser.parse(SAMPLE_TAB)
    print(json.dumps(result, indent=2))
