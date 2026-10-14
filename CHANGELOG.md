# Changelog

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
