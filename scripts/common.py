"""Shared setup for the round-4 scripts: load the corpus, scan every play with the rule
detector (cached in data/), and locate the known sonnets exactly as cell 8b does.
Everything comes from src/ — no scoring or cleaning is re-implemented here."""
import os
import pickle
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import sonnet_toolkit as tk                                          # noqa: E402
from known_sonnets import manual_play_sonnets, play_sonnet_specs     # noqa: E402

DATA = os.path.join(ROOT, "data", "pg100.txt")
CACHE = os.path.join(ROOT, "data", "rule_scans.pkl")


def load_works():
    text = open(DATA, encoding="utf-8-sig").read()
    return text, tk.parse_works(tk.strip_gutenberg(text))


def play_titles(works):
    return [t for t in works if t not in tk.NON_PLAY_TITLES]


def rule_scans(works):
    """scan_work for every play; cached because a full scan takes ~30 s."""
    key = (tuple(tk.WINDOW_SIZES), tk.SHORT_FORM_CREDIT, os.path.getmtime(os.path.join(ROOT, "src", "sonnet_toolkit.py")))
    if os.path.exists(CACHE):
        cached = pickle.load(open(CACHE, "rb"))
        if cached["key"] == key:
            return cached["scans"]
    scans = {t: tk.scan_work(t, works[t]) for t in play_titles(works)}
    pickle.dump({"key": key, "scans": scans}, open(CACHE, "wb"))
    return scans


def known_sonnets(works, specs=play_sonnet_specs, manual=manual_play_sonnets):
    """Cell 8b's located table: label, play, line, n_lines (unlocated ones dropped)."""
    rows = []
    for spec in specs:
        lab = spec["label"]
        first = manual[lab][0] if lab in manual else spec["first_line"]
        hit = tk.locate_passage(works, first)
        if hit:
            rows.append({"label": lab, "play": hit[0], "line": hit[1],
                         "n_lines": len(manual[lab]) if lab in manual else spec["num_lines"]})
    return pd.DataFrame(rows)


def all_windows(scans, weights=None):
    """Every (play, start) with its best score and window size, best first."""
    rows = []
    for t, scan in scans.items():
        sc, sizes = tk.scores_from_scan(scan, weights)
        for i in range(len(sc)):
            rows.append((t, i, int(sizes[i]), float(sc[i])))
    df = pd.DataFrame(rows, columns=["play", "start", "size", "score"])
    return df.sort_values("score", ascending=False, kind="stable").reset_index(drop=True)


def distinct_passages(windows, radius=tk.NMS_RADIUS):
    """Greedy NMS over a ranked window table (per play, same rule as tk.nms_top_k):
    keep a window unless a better kept one in the same play starts within `radius` lines."""
    kept, taken = [], {}
    for row in windows.itertuples(index=False):
        starts = taken.setdefault(row.play, [])
        if any(abs(row.start - s) < radius for s in starts):
            continue
        starts.append(row.start)
        kept.append(row)
    return pd.DataFrame(kept, columns=windows.columns).reset_index(drop=True)


def overlaps(play, start, size, known):
    """Labels of known sonnets whose span overlaps lines [start, start+size)."""
    k = known[known["play"] == play]
    return [r.label for r in k.itertuples() if start < r.line + r.n_lines and r.line < start + size]


def act_scene(scan, line):
    """Last ACT / SCENE heading at or before a cleaned line index ('' if none)."""
    act = scene = ""
    for idx, label, kind in scan["boundaries"]:
        if idx > line:
            break
        if kind == "act":
            act, scene = label, ""
        else:
            scene = label
    return " ".join(x for x in (act, scene) if x)
