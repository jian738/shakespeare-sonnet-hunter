# Top-3000 recheck (round 4, step 2)

**The rule detector's top 3000 windows contain 489 distinct passages, not 3000. 19 of the 21 locatable known play sonnets overlap at least one of those windows.** A reader who checked 3000 windows was looking at each passage about 6 times, shifted by a line or two.

Reproduce with `python scripts/top3000_recheck.py`. It needs `data/pg100.txt`, and all scoring comes from `src/sonnet_toolkit.py` at v8. The rule ranks are identical to the run-3 evaluation cell (8b).

## Why this matters for the earlier LLM check

The earlier check read the top 3000 windows. It reported 0 sonnets and about 20 sonnet-like passages. Three facts constrain how to read that result:

1. **About 489 passages, not 3000.** Window *i* and window *i+1* share 13 of 14 lines, so the 3000 windows collapse to 489 passages after non-maximum suppression with radius 18. That is the same rule the detector uses everywhere.
2. **Known sonnets were in the list.** 19 of the 21 known play sonnets present in Gutenberg #100 overlap at least one top-3000 window. Romeo and Juliet 1.5 is window 1, Comedy of Errors 3.2 is window 2, and LLL 1.1.80 is window 3. A check that found zero sonnets among them missed sonnets that were in front of it. That says the check was unreliable. It says nothing about the plays lacking sonnets.
3. **We don't know which ranking the check used.** If it read the classifier export (cell 7, top 2%), these numbers don't apply. In run 3 the classifiers placed only 2 of 21 known sonnets in their top 50 (recall@50 0.10).

## Counts

| Quantity | Value |
|---|---|
| Windows scored (every start line, best of sizes 10/11/14/15/18) | 107,699 |
| Score cut-off of the top 3000 | 0.627 |
| Plays represented in the top 3000 | 37 of 38 |
| Distinct passages after NMS (radius 18) | **489** |
| Windows per distinct passage | 6.1 |
| Known sonnets located in the corpus | 21 of 22 (Edward III is not in Gutenberg #100) |
| Overlapped by at least one top-3000 window | **19** |
| Matched by a distinct passage's start (cell 8b tolerance) | 17 |

## Each known sonnet

"Windows overlapping" counts how many of the top 3000 windows share at least one line with the sonnet. "Distinct-passage rank" is the cell 8b rank.

| Known sonnet | Windows overlapping | Best window rank | Distinct-passage rank |
|---|---:|---:|---:|
| The Two Gentlemen of Verona, 3.1.140–9 | 5 | 87 | 38 |
| The Comedy of Errors, 3.2.41–54 | 15 | 2 | 2 |
| Love's Labour's Lost, 1.1.80–9 | 9 | 3 | 3 |
| Love's Labour's Lost, 1.1.159–74 | 10 | 25 | 15 |
| Love's Labour's Lost, 4.2.106–19 | 3 | 22 | 13 |
| Love's Labour's Lost, 4.3.24–39 | 14 | 13 | 10 |
| Love's Labour's Lost, 4.3.57–70 | 13 | 39 | – (suppressed by a peak 4 lines away) |
| Love's Labour's Lost, 5.2.401–15 | 12 | 8 | 6 |
| Romeo and Juliet, Prologue | 5 | 9 | 7 |
| Romeo and Juliet, 1.5.92–109 | 5 | 1 | 1 |
| Romeo and Juliet, 2.0 Chorus | 5 | 4 | 4 |
| A Midsummer Night's Dream, 5.1.108–17 | 5 | 102 | 46 |
| Much Ado About Nothing, 3.1.107–16 | 0 | – | – |
| Henry V, Epilogue | 5 | 92 | 42 |
| As You Like It, 3.2.1–10 | 5 | 107 | 49 |
| Troilus and Cressida, 1.2.278–91 | 5 | 507 | 124 |
| All's Well That Ends Well, 1.1.212–225 | 1 | 1087 | 209 |
| All's Well That Ends Well, 3.4.4–17 | 5 | 7 | 5 |
| Pericles, scene 10.1–14 | 0 | – | – |
| Pericles, scene 21.225–35 | 2 | 548 | 130 |
| Cymbeline, 5.5.187–204 | 6 | 737 | – (suppressed by a peak 7 lines away) |

The two sonnets with no overlapping window score too low on their own. Much Ado is in the top 7.1% of windows and uses short tetrameter-like lines. Pericles 10 is in the top 58%; it is octosyllabic Gower verse, which the pentameter score penalizes.

## Distinct passages per play (top 3000)

The 489 passages are spread over 37 plays. The rhymed-verse plays lead: Love's Labour's Lost 45, Richard II 44, Midsummer 40, Romeo and Juliet 38, Pericles 30. Then 3 Henry VI, Richard III and Comedy of Errors have 22 each, and 1 Henry VI 21. Every other play has 17 or fewer.

## What this does and does not show

- It shows that the "top 3000" list held about 489 passages and nearly all the known sonnets.
- It does **not** say how many of the other ~470 passages are sonnets. That needs human judgement, and the blind annotation sheet (`annotation/`) is built for it. Question (3), whether the plays contain few sonnets, should be answered from those labels, not from this count.
