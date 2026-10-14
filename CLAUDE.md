# CLAUDE.md — Shakespeare Sonnet Hunter

Context for Claude Code. Read this before changing anything.

## What the project is
`notebooks/sonnet_hunter.ipynb` scans every window of Shakespeare's plays (Gutenberg #100)
for passages that behave like sonnets, using a rule-based score (meter, syllables, rhyme
scheme, couplet) and two classifiers (supervised and PU). It is judged by how well it
finds 22 known play sonnets (`play_sonnet_specs` in cell 3).

## Ground rules
- **Cell 4 (shared toolkit) is the single source** of tokenising, pronunciation, rhyme,
  meter, `score_window`, `clean_play_lines`, `scan_work`, NMS and parsing helpers.
  Never re-implement these in another cell; import from the toolkit.
- **Every change is judged by the evaluation cell (8b)**: recall@k and median rank on
  the TEST half of the known sonnets, with bootstrap intervals. A change that only
  helps the tune half, or moves recall within the interval, is not an improvement.
- **Validate a measure before using it** (as the coherence and distinctness cells do):
  if it can't tell known sonnets from random windows, it can't support a claim.
- Statistical tests treat the **window**, not the line, as the unit.
- The notebook runs in **Google Colab**; Gutenberg is downloaded at run time.
- Keep the user's own "Cell N" labels in code comments.

## Workflow for each round
1. The user runs the notebook in Colab and saves it with outputs.
2. Save the key text outputs to `docs/results/run-N.md`.
3. Add a round to `docs/walkthrough.md`, in this order:
   results -> walkthrough (cell table with outputs) -> what works well ->
   bugs found (table: # / Cell / Bug / Effect / Fix) -> method improvements ->
   suggested order of work -> "Changes made in version N+1"
   (each change as **Before / Problem / Now / Why it matters for the hunter**) ->
   what to check on the next run.
4. Make the code changes in `notebooks/sonnet_hunter.ipynb` (jupytext keeps the .py twin in sync).
5. Add a `CHANGELOG.md` entry, commit with a message that explains *why*, and tag `vN+1`.

## Writing style for docs
Lead with the result and its number. Short sentences, specific numbers, no hype.
State null results as findings.
