"""Tests for the line cleaner and window indexing (round 4: the professor's concern
that the code "may struggle with line indexing and stage directions").

The corpus tests need Gutenberg #100 at data/pg100.txt and are skipped without it:
    curl -L -o data/pg100.txt https://www.gutenberg.org/files/100/100-0.txt
Run:  pytest tests/
"""
import difflib
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import sonnet_toolkit as tk                      # noqa: E402
from known_sonnets import manual_play_sonnets, play_sonnet_specs   # noqa: E402

DATA = os.path.join(ROOT, "data", "pg100.txt")
NOT_IN_GUTENBERG_100 = {"Edward III, scene 2.322–3."}   # Edward III is not in the collection


# ---------- small cases, no corpus needed ------------------------------------
def test_bare_speaker_tag_without_period():
    lines = ["THESEUS.", "Let him approach.", "PROLOGUE", "If we offend, it is with our good will."]
    clean, _ = tk.clean_play_lines(lines)
    assert clean == [("Let him approach.", "THESEUS"),
                     ("If we offend, it is with our good will.", "PROLOGUE")]


def test_bare_tag_needs_a_dotted_use_in_the_same_play():
    clean, _ = tk.clean_play_lines(["KING.", "Ay me!", "FORD", "Well, I hope it be not so."])
    assert clean[-1] == ("Well, I hope it be not so.", "KING")     # no "FORD." anywhere -> not a tag
    clean, _ = tk.clean_play_lines(["FORD.", "Hi.", "PAGE.", "Ho.", "FORD", "Well, I hope it be not so."])
    assert clean[-1] == ("Well, I hope it be not so.", "FORD")


def test_inline_and_unclosed_directions():
    lines = ["KING.", "[_Reads_.] [_So sweet a kiss the golden sun gives not",
             "To those fresh morning drops upon the rose,"]
    clean, _ = tk.clean_play_lines(lines)
    assert [t for t, _ in clean] == ["So sweet a kiss the golden sun gives not",
                                     "To those fresh morning drops upon the rose,"]
    lines = ["ROMEO.", "[_To Juliet._] If I profane with my unworthiest hand",
             "[_Exeunt Capulet and others,", "ladies following._]", "Next line."]
    clean, _ = tk.clean_play_lines(lines)
    assert [t for t, _ in clean] == ["If I profane with my unworthiest hand", "Next line."]


# ---------- the real corpus ---------------------------------------------------
@pytest.fixture(scope="module")
def works():
    if not os.path.exists(DATA):
        pytest.skip("data/pg100.txt not downloaded")
    text = open(DATA, encoding="utf-8-sig").read()
    return tk.parse_works(tk.strip_gutenberg(text))


@pytest.fixture(scope="module")
def plays(works):
    return {t: l for t, l in works.items() if t not in tk.NON_PLAY_TITLES}


def _first_line(spec):
    lab = spec["label"]
    return manual_play_sonnets[lab][0] if lab in manual_play_sonnets else spec["first_line"]


def test_every_locatable_known_sonnet_is_found(works):
    missing = set()
    for spec in play_sonnet_specs:
        hit = tk.locate_passage(works, _first_line(spec))
        if hit is None:
            missing.add(spec["label"])
            continue
        play, i = hit
        found = tk.normalize_for_match(tk.clean_play_lines(works[play])[0][i][0])
        want = tk.normalize_for_match(_first_line(spec))
        # exact for all but Pericles 10 ("y-slackèd" vs Gutenberg's "yslaked"): >= 85% similar
        assert difflib.SequenceMatcher(None, found[:30], want[:30]).ratio() >= 0.85, spec["label"]
    assert missing == NOT_IN_GUTENBERG_100


def test_romeo_and_juliet_1_5_starts_at_if_i_profane(works):
    play, i = tk.locate_passage(works, "If I profane with my unworthiest hand")
    assert play == "THE TRAGEDY OF ROMEO AND JULIET"
    text, speaker = tk.clean_play_lines(works[play])[0][i]
    assert text.startswith("If I profane")
    assert speaker == "ROMEO"


def test_no_cleaned_line_contains_brackets(plays):
    bad = [(t, s) for t, l in plays.items() for s, _ in tk.clean_play_lines(l)[0] if "[" in s or "]" in s]
    assert bad == []


def test_midsummer_prologue_speaker(works):
    play, i = tk.locate_passage(works, "If we offend, it is with our good will.")
    assert play == "A MIDSUMMER NIGHT’S DREAM"
    clean = tk.clean_play_lines(works[play])[0]
    assert [sp for _, sp in clean[i:i + 10]] == ["PROLOGUE"] * 10
    assert clean[i + 10][1] == "THESEUS"          # "This fellow doth not stand upon points."


def test_window_index_is_the_same_text_everywhere(works):
    """Window (play, i) must be the same 14 lines in scan_work (rule detector and plots),
    the classifier windows (cell 6) and the evaluation cell's located start lines (8b)."""
    texts, meta = tk.build_play_windows(works, size=14)
    classifier = {(t, i): txt for txt, (t, i, _) in zip(texts, meta)}
    checked = 0
    for spec in play_sonnet_specs:
        hit = tk.locate_passage(works, _first_line(spec))
        if hit is None:
            continue
        play, i = hit
        scan = tk.scan_work(play, works[play])
        from_scan = "\n".join(tk.window_lines(scan["clean_data"], i, 14))
        from_cleaner = "\n".join(t for t, _ in tk.clean_play_lines(works[play])[0][i:i + 14])
        assert classifier[(play, i)] == from_scan == from_cleaner, spec["label"]
        checked += 1
    assert checked == len(play_sonnet_specs) - len(NOT_IN_GUTENBERG_100)
