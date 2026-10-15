# Annotation rubric (DRAFT — edit before use)

> Draft written for the project owner to revise. Every threshold below is a proposal, not
> a decision. Settle the definitions **before** anyone labels, and don't change them after
> seeing the scores in `key.csv`.

## The task

`top_candidates.csv` has 106 passages from Shakespeare's plays (Gutenberg #100), in random order. For each one, fill in:

| Column | Values |
|---|---|
| `label` | `sonnet`, `sonnet-like` or `neither` |
| `confidence` | 1 = guess, 2 = fairly sure, 3 = certain |
| `notes` | Optional. Give the rhyme scheme you hear, the line count you'd use, or why you chose the label. |

Label the passage shown. If the real passage starts a line or two earlier or later, say so in `notes`, for example "sonnet if you start one line later", and label what the passage would be after that shift.

**Don't open `key.csv`** until every row is labelled. It holds the detector scores and marks which items are already-known sonnets. Some passages are known sonnets mixed in to measure how well annotators agree with the established list. Some of them may look familiar. Label them anyway, by the rubric, not by memory.

`[NAME]` marks the speaker at the start of the passage and at every change of speaker. Stage directions have been removed. `start_line` counts cleaned spoken lines, not Gutenberg file lines.

## Definitions (proposed)

### `sonnet`: all of the following
1. **Length: 14 lines.** *Option to decide: allow 13–15, so that a half-line or a cue line split across speakers can count.*
2. **Meter: iambic pentameter** in most lines (about 10 syllables, mostly rising stress). Occasional feminine endings and inversions are fine.
3. **Rhyme scheme:** one of the standard sonnet schemes.
   - Shakespearean: ABAB CDCD EFEF GG
   - Petrarchan octave (ABBAABBA) with a sestet
   - Spenserian (ABAB BCBC CDCD EE)
   - Slant and historical rhymes count, for example *love/prove* and *come/doom*.
4. **Unity:** it reads as one bounded unit, with a closing couplet or turn that ends it.
5. **Speakers:** *option to decide:*
   - (a) **single speaker only**, or
   - (b) **shared speakers allowed**, as in Romeo and Juliet 1.5, where Romeo and Juliet split one sonnet.

   The 22 known play sonnets include shared ones, so (b) matches the established list.

### `sonnet-like`: clearly organised verse that misses one or two of the above
Examples:
- **Length 10–18 lines.** This includes the 10-line forms ABAB CDCD EE (Two Gentlemen 3.1, Much Ado 3.1) and the 18-line Cymbeline 5.5.
- **Tetrameter or octosyllabic lines** with a sonnet rhyme scheme (Pericles scene 10, Gower).
- A sonnet rhyme scheme that runs on into dialogue with no clear boundary.
- Quatrains plus a couplet that don't form a closed unit.

### `neither`
Blank verse, prose, couplet runs (AABB…) with no quatrain structure, songs, or anything else that isn't one of the above.

*Decision to make: is a run of heroic couplets (AABBCC…) `neither` or `sonnet-like`? This draft says `neither`, because the detector already rewards couplets, and calling them sonnet-like would inflate its apparent precision.*

## How the labels will be used
- **Annotator accuracy.** Compare each annotator's labels on the known sonnets in `key.csv` with `sonnet` or `sonnet-like`.
- **Agreement.** With at least two annotators, report Cohen's kappa on the three-way label.
- **Detector precision.** Count the share of non-known passages labelled `sonnet` or `sonnet-like`. Report each rank band (1–25, 26–50, 51–100) separately, with a binomial interval.
- **New finds.** A passage labelled `sonnet` by every annotator goes into `DISCOVERED_SONNETS` (`src/known_sonnets.py`). It is reported there and never used for tuning.
