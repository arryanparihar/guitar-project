"""Unit tests for tab_engine.py.

Tests validate the TabParser's ability to:
- detect tab string lines and ignore non-tab content
- perform single-pass column-aligned parsing
- handle all recognised techniques
- handle multi-character tokens (double-digit frets, harmonics)
- produce correctly sorted, structured output
"""

import unittest
from typing import Dict, List

from tab_engine import TabParser


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _find_slice(slices: List[Dict], time_index: int) -> Dict:
    """Return the time slice at *time_index*, raising AssertionError if absent."""
    for s in slices:
        if s["time_index"] == time_index:
            return s
    raise AssertionError(
        f"No time slice found for time_index={time_index}. "
        f"Available indices: {[s['time_index'] for s in slices]}"
    )


def _note(slc: Dict, string: str) -> Dict:
    """Return the note for *string* within *slc*, raising AssertionError if absent."""
    for n in slc["notes"]:
        if n["string"] == string:
            return n
    raise AssertionError(
        f"No note for string '{string}' in slice {slc}."
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestParseEmptyAndNonTab(unittest.TestCase):
    """Edge-case inputs that should produce no output."""

    def setUp(self) -> None:
        self.parser = TabParser()

    def test_empty_string(self) -> None:
        self.assertEqual(self.parser.parse(""), [])

    def test_only_blank_lines(self) -> None:
        self.assertEqual(self.parser.parse("\n\n\n"), [])

    def test_lyrics_only(self) -> None:
        # Lines that start with letters but have no dashes after '|'
        tab = "Am|Em|G\nSome lyrics here\n"
        self.assertEqual(self.parser.parse(tab), [])

    def test_chord_names_ignored(self) -> None:
        # "Am|something" has no '-' in content → not a tab line
        tab = "Am|Em G D\n"
        self.assertEqual(self.parser.parse(tab), [])


class TestSingleStringParsing(unittest.TestCase):
    """Tests with a single string line."""

    def setUp(self) -> None:
        self.parser = TabParser()

    def test_open_string(self) -> None:
        tab = "e|--0--|\n"
        result = self.parser.parse(tab)
        self.assertEqual(len(result), 1)
        slc = result[0]
        self.assertEqual(slc["time_index"], 2)
        self.assertEqual(len(slc["notes"]), 1)
        note = slc["notes"][0]
        self.assertEqual(note["string"], "e")
        self.assertEqual(note["fret"], "0")
        self.assertEqual(note["technique"], "standard")

    def test_double_digit_fret(self) -> None:
        tab = "e|--12--|\n"
        result = self.parser.parse(tab)
        self.assertEqual(len(result), 1)
        note = result[0]["notes"][0]
        self.assertEqual(note["fret"], "12")
        self.assertEqual(note["technique"], "standard")

    def test_natural_harmonic(self) -> None:
        tab = "e|--<12>--|\n"
        result = self.parser.parse(tab)
        self.assertEqual(len(result), 1)
        note = result[0]["notes"][0]
        self.assertEqual(note["fret"], "12")
        self.assertEqual(note["technique"], "natural_harmonic")

    def test_artificial_harmonic_square_brackets(self) -> None:
        tab = "e|--[12]--|\n"
        result = self.parser.parse(tab)
        note = result[0]["notes"][0]
        self.assertEqual(note["fret"], "12")
        self.assertEqual(note["technique"], "artificial_harmonic")

    def test_artificial_harmonic_parentheses(self) -> None:
        tab = "e|--(12)--|\n"
        result = self.parser.parse(tab)
        note = result[0]["notes"][0]
        self.assertEqual(note["fret"], "12")
        self.assertEqual(note["technique"], "artificial_harmonic")

    def test_slap_lowercase(self) -> None:
        tab = "E|--x--|\n"
        result = self.parser.parse(tab)
        note = result[0]["notes"][0]
        self.assertEqual(note["fret"], "x")
        self.assertEqual(note["technique"], "slap")

    def test_slap_uppercase(self) -> None:
        tab = "E|--X--|\n"
        result = self.parser.parse(tab)
        note = result[0]["notes"][0]
        self.assertEqual(note["fret"], "X")
        self.assertEqual(note["technique"], "slap")

    def test_hammer_on(self) -> None:
        tab = "e|--5h7--|\n"
        result = self.parser.parse(tab)
        techniques = [s["notes"][0]["technique"] for s in result]
        self.assertIn("standard", techniques)
        self.assertIn("hammer_on", techniques)

    def test_pull_off(self) -> None:
        tab = "e|--7p5--|\n"
        result = self.parser.parse(tab)
        techniques = [s["notes"][0]["technique"] for s in result]
        self.assertIn("pull_off", techniques)

    def test_slide_up(self) -> None:
        tab = "e|--5/7--|\n"
        result = self.parser.parse(tab)
        techniques = [s["notes"][0]["technique"] for s in result]
        self.assertIn("slide_up", techniques)

    def test_slide_down(self) -> None:
        tab = "e|--7\\5--|\n"
        result = self.parser.parse(tab)
        techniques = [s["notes"][0]["technique"] for s in result]
        self.assertIn("slide_down", techniques)

    def test_multiple_frets_ordered(self) -> None:
        tab = "e|--0---5---12--|\n"
        result = self.parser.parse(tab)
        indices = [s["time_index"] for s in result]
        self.assertEqual(indices, sorted(indices))
        frets = [s["notes"][0]["fret"] for s in result]
        self.assertEqual(frets, ["0", "5", "12"])


class TestDoubleDigitAlignment(unittest.TestCase):
    """Verify that double-digit frets do not shift alignment on other strings."""

    def setUp(self) -> None:
        self.parser = TabParser()

    def test_double_digit_does_not_drift(self) -> None:
        # Both strings have a note at column 2.
        # The e-string uses double-digit '12'; B-string uses single-digit '0'.
        # They must appear in the same time slice (same column).
        tab = (
            "e|--12--|\n"
            "B|--0---|\n"
        )
        result = self.parser.parse(tab)
        # Both tokens start at column 2
        slc = _find_slice(result, 2)
        strings_present = {n["string"] for n in slc["notes"]}
        self.assertIn("e", strings_present)
        self.assertIn("B", strings_present)

    def test_natural_harmonic_does_not_drift(self) -> None:
        # <12> on e-string at column 2; '0' on B-string at column 2.
        tab = (
            "e|--<12>--|\n"
            "B|--0-----|\n"
        )
        result = self.parser.parse(tab)
        slc = _find_slice(result, 2)
        strings_present = {n["string"] for n in slc["notes"]}
        self.assertIn("e", strings_present)
        self.assertIn("B", strings_present)
        e_note = _note(slc, "e")
        self.assertEqual(e_note["technique"], "natural_harmonic")

    def test_artificial_harmonic_does_not_drift(self) -> None:
        tab = (
            "e|--[12]--|\n"
            "B|--0-----|\n"
        )
        result = self.parser.parse(tab)
        slc = _find_slice(result, 2)
        strings_present = {n["string"] for n in slc["notes"]}
        self.assertIn("e", strings_present)
        self.assertIn("B", strings_present)
        e_note = _note(slc, "e")
        self.assertEqual(e_note["technique"], "artificial_harmonic")


class TestSimultaneousNotes(unittest.TestCase):
    """Notes in the same column must be grouped into one time slice."""

    def setUp(self) -> None:
        self.parser = TabParser()

    def test_bass_and_slap_simultaneously(self) -> None:
        # A-string open (bass note) and E-string slap at the same column
        tab = (
            "A|--0--|\n"
            "E|--x--|\n"
        )
        result = self.parser.parse(tab)
        slc = _find_slice(result, 2)
        self.assertEqual(len(slc["notes"]), 2)
        a_note = _note(slc, "A")
        e_note = _note(slc, "E")
        self.assertEqual(a_note["technique"], "standard")
        self.assertEqual(e_note["technique"], "slap")

    def test_all_strings_same_column(self) -> None:
        tab = (
            "e|--0--|\n"
            "B|--1--|\n"
            "G|--0--|\n"
            "D|--2--|\n"
            "A|--2--|\n"
            "E|--0--|\n"
        )
        result = self.parser.parse(tab)
        slc = _find_slice(result, 2)
        self.assertEqual(len(slc["notes"]), 6)
        strings_present = {n["string"] for n in slc["notes"]}
        self.assertEqual(strings_present, {"e", "B", "G", "D", "A", "E"})


class TestDynamicTuning(unittest.TestCase):
    """Parser must handle non-standard tuning string names."""

    def setUp(self) -> None:
        self.parser = TabParser()

    def test_drop_d_tuning(self) -> None:
        # Standard strings but with D as lowest instead of E
        tab = (
            "e|--0--|\n"
            "B|--0--|\n"
            "G|--0--|\n"
            "D|--0--|\n"
            "A|--0--|\n"
            "D|--0--|\n"
        )
        result = self.parser.parse(tab)
        slc = _find_slice(result, 2)
        self.assertEqual(len(slc["notes"]), 6)

    def test_sharp_string_name(self) -> None:
        tab = "F#|--5--|\n"
        result = self.parser.parse(tab)
        self.assertEqual(len(result), 1)
        note = result[0]["notes"][0]
        self.assertEqual(note["string"], "F#")

    def test_flat_string_name(self) -> None:
        tab = "Bb|--3--|\n"
        result = self.parser.parse(tab)
        self.assertEqual(len(result), 1)
        note = result[0]["notes"][0]
        self.assertEqual(note["string"], "Bb")

    def test_lowercase_string_name(self) -> None:
        tab = "e|--7--|\n"
        result = self.parser.parse(tab)
        note = result[0]["notes"][0]
        self.assertEqual(note["string"], "e")


class TestNonTabContentIgnored(unittest.TestCase):
    """Lines without dashes or not matching the string-line pattern are ignored."""

    def setUp(self) -> None:
        self.parser = TabParser()

    def test_blank_lines_between_tab_blocks(self) -> None:
        tab = (
            "e|--0--|\n"
            "\n"
            "e|--5--|\n"
        )
        result = self.parser.parse(tab)
        # Each block is parsed independently; same column index from two blocks
        # still ends up in the same time_index
        self.assertGreaterEqual(len(result), 1)

    def test_lyrics_above_tab(self) -> None:
        tab = (
            "Verse 1:\n"
            "e|--0--|\n"
            "B|--1--|\n"
        )
        result = self.parser.parse(tab)
        self.assertGreater(len(result), 0)
        strings_present = {
            n["string"] for slc in result for n in slc["notes"]
        }
        self.assertEqual(strings_present, {"e", "B"})

    def test_chord_header_ignored(self) -> None:
        tab = (
            "Am    G     C\n"
            "e|--0---3---0--|\n"
            "B|--1---3---1--|\n"
        )
        result = self.parser.parse(tab)
        strings_present = {
            n["string"] for slc in result for n in slc["notes"]
        }
        # Only tab strings, no spurious 'A', 'G', 'C' string entries
        self.assertNotIn("G", strings_present)

    def test_no_split_on_dash(self) -> None:
        # Regression: parser must NOT use str.split('-').
        # Verify by ensuring frets after long dash runs are at correct indices.
        tab = "e|-----5-----12-----|\n"
        result = self.parser.parse(tab)
        self.assertEqual(len(result), 2)
        idx_5 = result[0]["time_index"]
        idx_12 = result[1]["time_index"]
        self.assertLess(idx_5, idx_12)
        self.assertEqual(result[0]["notes"][0]["fret"], "5")
        self.assertEqual(result[1]["notes"][0]["fret"], "12")


class TestComplexTab(unittest.TestCase):
    """Integration test with a full, multi-technique tab block."""

    SAMPLE_TAB = (
        "e|--0-------<12>-------[12]-|\n"
        "B|--1--------12---------1---|\n"
        "G|--0---------0---------0---|\n"
        "D|--2---------2---------2---|\n"
        "A|--0---------0---------0---|\n"
        "E|--x---------x---------x---|\n"
    )

    def setUp(self) -> None:
        self.parser = TabParser()
        self.result = self.parser.parse(self.SAMPLE_TAB)

    def test_output_is_sorted_by_time_index(self) -> None:
        indices = [s["time_index"] for s in self.result]
        self.assertEqual(indices, sorted(indices))

    def test_simultaneous_bass_and_slap_at_first_chord(self) -> None:
        # At column 2 we expect A-string (bass, fret 0) and E-string (slap, x)
        slc = _find_slice(self.result, 2)
        a_note = _note(slc, "A")
        e_note = _note(slc, "E")
        self.assertEqual(a_note["technique"], "standard")
        self.assertEqual(e_note["technique"], "slap")

    def test_natural_harmonic_detected(self) -> None:
        # <12> on e-string
        found = any(
            n["technique"] == "natural_harmonic"
            for slc in self.result
            for n in slc["notes"]
        )
        self.assertTrue(found, "Expected a natural_harmonic but none found")

    def test_artificial_harmonic_detected(self) -> None:
        # [12] on e-string
        found = any(
            n["technique"] == "artificial_harmonic"
            for slc in self.result
            for n in slc["notes"]
        )
        self.assertTrue(found, "Expected an artificial_harmonic but none found")

    def test_double_digit_fret_detected(self) -> None:
        # 12 on B-string
        found = any(
            n["fret"] == "12" and n["technique"] == "standard"
            for slc in self.result
            for n in slc["notes"]
        )
        self.assertTrue(found, "Expected a standard double-digit fret 12")

    def test_no_split_usage(self) -> None:
        """Regression: parse must never call str.split('-') internally.

        We validate indirectly: a double-digit fret immediately adjacent to a
        single-digit fret on a neighbouring string must NOT share a time slice
        unless their start columns genuinely match.
        """
        # B-string '12' starts at column 11; e-string '<12>' starts at column 10.
        # They must NOT be in the same time slice.
        idx_e_harmonic = None
        idx_b_double = None
        for slc in self.result:
            for n in slc["notes"]:
                if n["string"] == "e" and n["technique"] == "natural_harmonic":
                    idx_e_harmonic = slc["time_index"]
                if n["string"] == "B" and n["fret"] == "12" and n["technique"] == "standard":
                    idx_b_double = slc["time_index"]
        self.assertIsNotNone(idx_e_harmonic)
        self.assertIsNotNone(idx_b_double)
        # They start one column apart (<12> is 4 chars wide, but 12 is 2)
        # so they MUST be in different slices.
        self.assertNotEqual(idx_e_harmonic, idx_b_double)


if __name__ == "__main__":
    unittest.main()
