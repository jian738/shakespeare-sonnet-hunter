# Changelog

## v8 — answering the review: tested cleaner, top-3000 recheck, blind annotation

Why: the professor asked whether line indexing and stage directions are handled, why an LLM
pass over the top 3000 windows found no sonnets, and whether the plays simply hold few.
Run 3 (v7) located 21 of 22 known sonnets (Edward III is not in Gutenberg #100); rule test
recall@50 0.73, median rank 13. The fused ranking was worse (0.43 vs 0.67 over all 21).

- Cleaner: speaker tags printed without a period (PROLOGUE, FORD, SNOUT...) are kept;
  435 lines had the wrong speaker, including Midsummer's Prologue as THESEUS. Text and indices unchanged.
- Toolkit moved to `src/sonnet_toolkit.py` (plus parse_works, locate_passage, ranked_hits,
  build_play_windows); known sonnets in `src/known_sonnets.py`; cell 1 clones the repo in Colab.
- `tests/test_cleaner.py`: known sonnets found, R&J 1.5 start, no brackets, Prologue speaker,
  one window index everywhere.
- Top-3000 recheck: 489 distinct passages; 19 of 21 known sonnets inside (`docs/results/top3000-recheck.md`).
- Blind annotation sheet: top 100 rule passages + known sonnets, shuffled, no scores; draft rubric.
- `DISCOVERED_SONNETS` (reported by `report_discovered()`, never tuned on).
- Run 3 recorded in `docs/results/run-3.md` with every figure.

## v7 — fixes from the second run

Why: the second run showed TF-IDF returning by an AUC-based switch (Friar Laurence 0.06 -> 0.69),
a too-lenient clustering verdict, label errors, and an AR(1) null that still flagged 46% of the spectrum.

- TF-IDF off by default (`USE_TFIDF = False`).
- Clustering verdict needs silhouette > 0.05 and stability ARI >= 0.5.
- Slant rhymes score 0.45 so scheme labels skip them; foregone/begone/bemoan added to historical rhymes.
- Reduced-vowel final syllables count as weak rhymes (ornament/content).
- `show_raw()` prints raw Gutenberg lines for unlocated sonnets; unclosed `[` only skips if `]` is within 4 lines.
- `explain_misses()`: low score vs NMS suppression for each missed sonnet.
- `SHORT_FORM_CREDIT` setting and a 0.75 vs 1.0 comparison for 10-line sonnets.
- Fused ranking (rule + PU, reciprocal rank fusion) evaluated like any detector.
- Wavelet significance vs phase-randomized surrogates (localized power).

## v6 — fixes from the first run, dependent-data statistics

Why: the first run located only 15 of 22 known sonnets (a cleaner bug), and several
analyses used tests that assume independent lines.

- Cleaner keeps verse after `[...]` directions (5 sonnets were hidden); PROLOGUE/EPILOGUE as speakers.
- Fuzzy location of known sonnets (Pericles 10).
- PU ranked by unclipped g/c (no ties at the top).
- Scheme labels accept weak rhymes.
- Bootstrap 95% intervals on recall@50 and median rank; documented weights kept (tuning gave no test gain).
- Wavelet significance vs an AR(1) red-noise null.
- Rare words: window permutation test (t-test kept, labelled naive).
- Clustering: silhouette vs a shuffled-word null, plus bootstrap stability.
- Embeddings: all-MiniLM-L6-v2, mean-pooled, centered; validated on known sonnets first.

## v5 — first review implemented

Why: nothing measured whether the hunt worked, and the rhyme rule called 32% of random line-end pairs a rhyme.

- Shared toolkit (cell 4): one copy of every helper; old copies removed.
- Tokenising: curly apostrophes and italic underscores handled (sister's, o'er, `tongue._`).
- Rhyme: graded levels (1 / 0.5 / 0) replace letter matching; random pairs called full rhymes 32% -> 1.5%.
- Meter: alignment against iambic templates; mean meter on real verse 0.62 -> 0.87.
- One scorer and one set of weights (25/15/35/25) for the rule and neural detectors.
- One line cleaner for every cell; PLAY SONNETS excluded from training and scanning.
- Classifiers: 5-fold held-out AUC, TF-IDF tested, PU c estimated on held-out positives.
- New evaluation cell (8b): recall@k and median rank on the known play sonnets, tune/test split.
- Speed: cached line features and a vectorized scan (~10x faster).
- Bugs 4-11 from the review fixed (feature count, duplicate title, PCA order, undefined
  variables, overwritten exports, duplicated rows, no-NMS candidates).
