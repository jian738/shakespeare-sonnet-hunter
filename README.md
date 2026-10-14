# Shakespeare Sonnet Hunter

Searches every window of Shakespeare's plays (Project Gutenberg #100) for passages that
behave like sonnets, and measures how well it does by trying to find 22 sonnets already
known to be embedded in the plays.

**Current version: v7.** Latest measured results (second full run, v6): the rule-based
detector finds 65% of the 20 locatable known sonnets in its top 50 (95% interval 45-85%);
the supervised and PU classifiers find 5% and 10%. See `docs/results/`.

## Layout

| Path | What it is |
| --- | --- |
| `notebooks/sonnet_hunter.ipynb` | The notebook. Every version is a git commit; tags `v4`-`v7`. |
| `notebooks/sonnet_hunter.py` | Plain-Python twin (jupytext) for readable diffs. |
| `docs/walkthrough.md` | Full review: original walkthrough, then each round of results, bugs, fixes. |
| `docs/walkthrough.pdf` | Same, as a PDF. |
| `docs/results/run-N.md` | Key text output from each full Colab run. |
| `CHANGELOG.md` | One entry per version: what changed and why. |
| `CLAUDE.md` | Instructions Claude Code reads every session. |

## Run it

Open `notebooks/sonnet_hunter.ipynb` in Google Colab and run all cells (Runtime > Restart
and run all). It downloads the text itself. Allow 30-60 minutes for the full run.

## See what changed between versions

```bash
git log --oneline --decorate
git diff v6 v7 -- notebooks/sonnet_hunter.py
```

## Local setup (once)

```bash
pip install jupytext nbstripout
nbstripout --install          # strips notebook outputs on every commit
```

After editing the notebook, run `jupytext --sync notebooks/sonnet_hunter.ipynb` so the
`.py` twin matches.
