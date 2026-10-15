"""Round 4, step 2: how many DISTINCT passages are in the rule detector's top 3000
windows, and which known play sonnets fall inside them?

An earlier LLM check read the top 3000 windows. Overlapping windows are mostly the same
passage shifted by a line or two, so this counts what was really there to read.
Run:  python scripts/top3000_recheck.py   (needs data/pg100.txt)
"""
import numpy as np

from common import (tk, load_works, rule_scans, known_sonnets, all_windows,
                    distinct_passages, overlaps)

TOP = 3000
_, works = load_works()
scans = rule_scans(works)
known = known_sonnets(works)
win = all_windows(scans)
top = win.head(TOP)
dist = distinct_passages(top, radius=tk.NMS_RADIUS)

print(f"Windows scored: {len(win)} (every start line of {len(scans)} plays; best of sizes {tk.WINDOW_SIZES})")
print(f"Top {TOP} windows: score >= {top.score.min():.3f}; plays represented: {top.play.nunique()}")
print(f"Distinct passages after NMS (radius {tk.NMS_RADIUS}): {len(dist)}")
print(f"Windows per distinct passage: {TOP / len(dist):.1f}")

# known sonnets: (a) any top-3000 window overlaps their span; (b) rank among distinct passages
hits = {lab: [] for lab in known.label}
for k, r in enumerate(top.itertuples(index=False), 1):
    for lab in overlaps(r.play, r.start, r.size, known):
        hits[lab].append(k)
ranks = tk.known_ranks([(r.play, r.start, r.score) for r in dist.itertuples(index=False)], known)
print(f"\nKnown play sonnets located: {len(known)} of 22 (Edward III is not in Gutenberg #100)")
print(f"Overlapped by at least one top-{TOP} window: {sum(bool(v) for v in hits.values())}")
print(f"Matched by a distinct passage start (cell 8b tolerance): {int(np.isfinite(ranks).sum())}")
print("\nlabel | windows overlapping | best window rank | distinct-passage rank")
for (lab, w), rk in zip(hits.items(), ranks):
    print(f"{lab} | {len(w)} | {min(w) if w else '-'} | {int(rk) if np.isfinite(rk) else '-'}")

print("\nDistinct passages per play (top 3000):")
print(dist.play.value_counts().to_string())
