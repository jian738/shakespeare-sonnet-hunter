"""Round 4, step 3: blind annotation sheet.

annotation/top_candidates.csv  what annotators see: the top 100 distinct passages of the
                               rule ranking plus the known play sonnets, shuffled, with
                               speaker names and no scores.
annotation/key.csv             what they don't see: source, rank, score, and which items
                               are known sonnets (to measure annotator accuracy).

The fused (rule + PU) ranking is left out: in run 3 it was no better than the rule
ranking (all 21: recall@50 0.43 [0.24, 0.62] vs 0.67 [0.48, 0.86]).
Run:  python scripts/export_annotation.py   (needs data/pg100.txt)
"""
import os

import numpy as np
import pandas as pd

from common import ROOT, tk, load_works, rule_scans, known_sonnets, act_scene, overlaps

TOP = 100
SEED = 2026
OUT = os.path.join(ROOT, "annotation")


def passage_text(clean_data, start, size):
    """Lines with [SPEAKER] at the start and at every change of speaker."""
    out, prev = [], None
    for text, sp in clean_data[start:start + size]:
        out.append(f"[{sp}] {text}" if sp != prev else text)
        prev = sp
    return "\n".join(out)


_, works = load_works()
scans = rule_scans(works)
known = known_sonnets(works)
scores = {t: tk.scores_from_scan(s) for t, s in scans.items()}
hits = tk.ranked_hits({t: sc for t, (sc, _) in scores.items()})[:TOP]

items = []
covered = set()
for rank, (play, start, score) in enumerate(hits, 1):
    size = int(scores[play][1][start])
    labs = overlaps(play, start, size, known)
    covered.update(labs)
    items.append({"play": play, "start_line": start, "n_lines": size, "source": "rule_top100",
                  "rule_rank": rank, "rule_score": round(score, 4), "window_size": size,
                  "is_known_sonnet": bool(labs), "known_label": "; ".join(labs)})

# known sonnets not already overlapped by a top-100 passage: add them at their own span
for r in known.itertuples():
    if r.label in covered:
        continue
    sc = scores[r.play][0]
    items.append({"play": r.play, "start_line": int(r.line), "n_lines": int(r.n_lines),
                  "source": "known_sonnet", "rule_rank": np.nan,
                  "rule_score": round(float(sc[r.line]), 4) if r.line < len(sc) else np.nan,
                  "window_size": np.nan, "is_known_sonnet": True, "known_label": r.label})

df = pd.DataFrame(items).sample(frac=1, random_state=SEED).reset_index(drop=True)
df.insert(0, "id", [f"P{k + 1:03d}" for k in range(len(df))])
df["act_scene"] = [act_scene(scans[p], s) for p, s in zip(df.play, df.start_line)]
df["text"] = [passage_text(scans[p]["clean_data"], s, n) for p, s, n in zip(df.play, df.start_line, df.n_lines)]

os.makedirs(OUT, exist_ok=True)
sheet = df[["id", "play", "act_scene", "start_line", "n_lines", "text"]].copy()
sheet["label"] = ""      # sonnet / sonnet-like / neither  (see RUBRIC.md)
sheet["confidence"] = "" # 1-3
sheet["notes"] = ""
sheet.to_csv(os.path.join(OUT, "top_candidates.csv"), index=False)
df[["id", "source", "rule_rank", "rule_score", "window_size", "is_known_sonnet", "known_label",
    "play", "start_line"]].to_csv(os.path.join(OUT, "key.csv"), index=False)

print(f"{len(df)} items: {int((df.source == 'rule_top100').sum())} rule top-{TOP} passages "
      f"({int(df[df.source == 'rule_top100'].is_known_sonnet.sum())} of them overlap a known sonnet) "
      f"+ {int((df.source == 'known_sonnet').sum())} known sonnets added")
print(f"Known sonnets represented: {len(covered) + int((df.source == 'known_sonnet').sum())} of {len(known)}")
print("Plays:", df.play.nunique())
