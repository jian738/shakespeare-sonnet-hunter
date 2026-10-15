"""Shared sonnet toolkit: the single source for tokenising, pronunciation, rhyme,
meter, score_window, clean_play_lines, scan_work, NMS and parsing helpers.

Notebook cell 4 imports everything from here; tests/ test it directly.
Moved out of the notebook in version 8 so the cleaner and the window indexing
can be unit-tested (round 4, the professor's line-indexing concern).

Settings such as SHORT_FORM_CREDIT are module globals: change them with
``sonnet_toolkit.SHORT_FORM_CREDIT = x``, not by assigning the imported name.
"""
import re
import difflib
from collections import defaultdict
from functools import lru_cache

import numpy as np
import pywt
from scipy.ndimage import gaussian_filter

import nltk
from nltk.corpus import cmudict
try:
    d = cmudict.dict()
except LookupError:
    nltk.download("cmudict", quiet=True)
    d = cmudict.dict()

FULL_TEXT = None    # raw Gutenberg text, for show_raw(); cell 4 sets it

# ---------- 0. Configuration -------------------------------------------------
# One set of weights used by EVERY rule-based detector (bug 3).
# These are the weights your docstring documents.
# First full run (Oct 13): tune_weights() on the 7 tune-half sonnets chose
# meter 0.4 / syllable 0 / rhyme 0.3 / couplet 0.3, but the test half did not move
# (recall@50 0.62 -> 0.62, median rank 28 -> 28). With n this small that is noise,
# so the documented weights are kept. Re-tune once more known sonnets are located.
WEIGHTS = {"meter": 0.25, "syllable": 0.15, "rhyme": 0.35, "couplet": 0.25}
COMPONENTS = ("meter", "syllable", "rhyme", "couplet")
WINDOW_SIZES = (10, 11, 14, 15, 18)
# Rhyme credit for a body that is not exactly three quatrains. Two quatrains + couplet
# (10 lines, ABAB CDCD EE) is a real form: 6 of the 22 known play sonnets have 10 lines.
# The second run ranked several of them low (Much Ado 3.1 at 951). The evaluation cell
# compares 0.75 with 1.0 on the known sonnets; change this only if the TEST half improves.
SHORT_FORM_CREDIT = 0.75
NMS_RADIUS = max(WINDOW_SIZES)          # suppress hits closer than one window

# ---------- 1. Tokenising (bug 2) --------------------------------------------
# Gutenberg uses curly apostrophes (’) and _underscores_ for italics.
# r"\w+" split "sister’s" into "sister" + "s" (an extra stressed beat) and
# kept "_" as a word, so an italic line ending "tongue._" ended in "_".
_APOS = str.maketrans({"’": "'", "‘": "'", "ʼ": "'"})
_WORD_RE = re.compile(r"'?[A-Za-z]+(?:'[A-Za-z]+)*'?")

def normalize_line(line):
    return line.translate(_APOS).replace("_", "")

def tokenize_words(line):
    """Words of a verse line, apostrophes kept inside words ("o'er", "'tis")."""
    return _WORD_RE.findall(normalize_line(line))

def end_word(line):
    words = tokenize_words(line)
    return words[-1].lower().strip("'") if words else ""

# ---------- 2. Pronunciation lookup ------------------------------------------
# A few Early Modern forms CMU lacks.
_EME_PRONS = {
    "o'er": [["OW1", "R"]], "e'er": [["EH1", "R"]], "ne'er": [["N", "EH1", "R"]],
    "th'": [["DH"]], "i'": [["IH0"]], "o'": [["AH0"]], "ta'en": [["T", "EY1", "N"]],
    "eyne": [["AY1", "N"]], "hie": [["HH", "AY1"]], "wot": [["W", "AA1", "T"]],
}
_CONS = {"s": "Z", "st": "S T", "d": "D", "t": "T"}

@lru_cache(maxsize=None)
def _cmu_prons(word):
    """All CMU pronunciations of word (lower-case), handling elisions; None if unknown."""
    w = word.lower()
    if w in d: return d[w]
    if w in _EME_PRONS: return _EME_PRONS[w]
    bare = w.strip("'")
    if bare in d: return d[bare]
    # possessive / elided endings: sister's, liv'st, lov'd  -> base + consonant, no extra syllable
    for suf in ("'st", "'d", "'s", "'t"):
        if w.endswith(suf):
            base = w[: -len(suf)]
            for cand in (base, base + "e"):
                if cand in d:
                    tail = _CONS[suf[1:]].split()
                    return [p + tail for p in d[cand]]
    joined = w.replace("'", "")
    if joined in d: return d[joined]
    return None

_G2P_MODEL = None
@lru_cache(maxsize=None)
def _g2p_prons(word):
    """Neural G2P for out-of-vocabulary words only (g2p_en itself uses CMU for known words)."""
    global _G2P_MODEL
    if _G2P_MODEL is None:
        from g2p_en import G2p          # imported lazily: only mode='g2p' needs it
        _G2P_MODEL = G2p()
    phones = [p for p in _G2P_MODEL(word) if p.strip()]
    return [phones] if phones else None

def word_prons(word, mode="cmu"):
    prons = _cmu_prons(word)
    if prons is None and mode == "g2p":
        prons = _g2p_prons(word.lower().strip("'"))
    return prons

def _fallback_syllables(word):
    w = word.lower().replace("'", "")
    n = len(re.findall(r"[aeiouy]+", w))
    if w.endswith("e") and not w.endswith(("le", "ee", "ye")) and n > 1:
        n -= 1                                   # silent final e
    return max(n, 1) if w else 0

def count_syllables(word, mode="cmu"):
    prons = word_prons(word, mode)
    if prons:
        return sum(p[-1].isdigit() for p in prons[0])
    return _fallback_syllables(word)

# ---------- 3. Stress and meter (method 4) -----------------------------------
# CMU marks nearly every monosyllable as stressed ("my", "do", "with"), which
# made ordinary iambic lines look irregular. Monosyllabic function words are
# now unstressed, other monosyllables stressed, and both are "flexible":
# a flexible syllable in the "wrong" slot costs 0.5 instead of 1.
FUNCTION_WORDS = set("""
a an the and but or nor for so yet if as than that though while when where whence
whose who whom which what i me my mine we us our you your ye thou thee thy thine
he him his she her it its they them their this these those there here
to of in on at by from with without into upon unto o'er through
am is are was were be been being do does did doth dost hath hast have has had
shall shalt should will wilt would can canst could may mayst might must
not no all such some each own 'tis 'twas t' th' i' o' 'gainst
""".split())

@lru_cache(maxsize=None)
def word_stress(word, mode="cmu"):
    """Tuple of (stress, flexible) per syllable."""
    w = word.lower()
    prons = word_prons(w, mode)
    if prons:
        stresses = [int(p[-1]) for p in prons[0] if p[-1].isdigit()]
    else:
        stresses = [None] * _fallback_syllables(w)          # unknown word
    if len(stresses) == 1:
        return ((0, True),) if w.strip("'") in FUNCTION_WORDS or w in FUNCTION_WORDS else ((1, True),)
    out = []
    for s in stresses:
        if s is None: out.append((1, True))                  # unknown: no hard claim
        elif s == 1: out.append((1, False))
        elif s == 2: out.append((1, True))                   # secondary stress is flexible
        else: out.append((0, False))
    return tuple(out)

def get_stress_pattern(line, mode="cmu"):
    """0/1 pattern (kept for old plotting code); see line_stress for flexibility."""
    return [s for s, _ in line_stress(line, mode)]

@lru_cache(maxsize=None)
def line_stress(line, mode="cmu"):
    out = []
    for w in tokenize_words(line):
        out.extend(word_stress(w, mode))
    return tuple(out)

_IAMB = [0, 1] * 5
_TEMPLATES = [
    _IAMB,                        # regular pentameter
    [1, 0] + _IAMB[2:],           # trochaic first foot
    _IAMB + [0],                  # feminine ending
    [1, 0] + _IAMB[2:] + [0],     # both
]

def _align_cost(pattern, template):
    """Edit distance: hard mismatch 1, flexible mismatch 0.5, missing/extra syllable 1."""
    n, m = len(pattern), len(template)
    prev = [float(j) for j in range(m + 1)]
    for i in range(1, n + 1):
        s, flex = pattern[i - 1]
        cur = [float(i)] + [0.0] * m
        for j in range(1, m + 1):
            sub = 0.0 if s == template[j - 1] else (0.5 if flex else 1.0)
            cur[j] = min(prev[j - 1] + sub, prev[j] + 1.0, cur[j - 1] + 1.0)
        prev = cur
    return prev[m]

def meter_score(pattern):
    """Best alignment of a line's stresses against iambic pentameter variants (0-1)."""
    if len(pattern) < 7 or len(pattern) > 14:
        return 0.0
    cost = min(_align_cost(pattern, t) for t in _TEMPLATES)
    return max(0.0, 1.0 - cost / 10.0)

def is_iambic(pattern):
    """Backward-compatible wrapper: accepts a plain 0/1 list."""
    return meter_score(tuple((s, False) for s in pattern))

def syllable_score(pattern):
    n = len(pattern)
    if n == 10 or (n == 11 and pattern[-1][0] == 0):     # 11 only with an unstressed (feminine) ending
        return 1.0
    return max(0.0, 1.0 - 0.1 * abs(n - 10))

@lru_cache(maxsize=None)
def line_features(line, mode="cmu"):
    """Everything the rule score needs from ONE line, computed once and cached
    (a line sits in ~68 overlapping windows of sizes 10-18)."""
    pat = line_stress(line, mode)
    return meter_score(pat), syllable_score(pat), end_word(line)

# ---------- 4. Rhyme (bug 1, method 4) ---------------------------------------
_HISTORICAL_GROUPS = [
    ["love", "move", "prove", "remove", "approve", "reprove"],
    ["gone", "foregone", "begone", "alone", "moan", "bemoan", "one", "none", "done", "shown", "own", "upon", "on"],
    ["forth", "worth"],
    ["wind", "mind", "kind", "find", "behind", "blind"],
    ["come", "doom", "tomb", "home", "roam"],
    ["heaven", "even", "given", "driven", "riven"],
    ["great", "defeat", "seat", "eat", "conceit"],
    ["tongue", "young", "song", "wrong", "long", "sung", "strong", "among"],
    ["war", "are", "far", "star", "jar"],
    ["blood", "good", "flood", "stood", "food", "brood", "mood"],
    ["eye", "melancholy", "remedy", "majesty", "die", "lie", "history"],
]
HISTORICAL_RHYMES = defaultdict(set)
for group in _HISTORICAL_GROUPS:
    for a in group:
        HISTORICAL_RHYMES[a].update(g for g in group if g != a)

# Same stressed vowel, different final consonants (moan/o'er). Slightly below 0.5 so that
# scheme LABELS (check_rhyme, min_level 0.5) skip it: the second run labelled Sonnet 30
# "ABBA" because foregone/before and o'er/moan matched on vowels alone.
SLANT_LEVEL = 0.45

def _strip(phones):
    return tuple(re.sub(r"\d", "", ph) for ph in phones)

def _rhyme_parts(prons):
    """Last STRESSED vowel to the end, stress digits removed (so AY1 matches AY2)."""
    parts = set()
    for p in prons:
        idx = next((i for i in range(len(p) - 1, -1, -1) if p[i][-1] in "12"), None)
        if idx is None:
            idx = next((i for i in range(len(p) - 1, -1, -1) if p[i][-1].isdigit()), len(p) - 1)
        parts.add(_strip(p[idx:]))
    return parts

def _final_syllable_parts(prons):
    """Last vowel (stressed or not) to the end: dignity/mutiny, argument/punishment."""
    parts = set()
    for p in prons:
        idx = next((i for i in range(len(p) - 1, -1, -1) if p[i][-1].isdigit()), None)
        if idx is not None:
            parts.add(_strip(p[idx:]))
    return parts

_REDUCED = {"AH0", "IH0", "ER0", "UH0"}

def _final_vowel_and_coda(prons):
    out = set()
    for p in prons:
        idx = next((i for i in range(len(p) - 1, -1, -1) if p[i][-1].isdigit()), None)
        if idx is not None and idx < len(p) - 1:          # needs a final consonant
            out.add((p[idx], tuple(p[idx + 1:])))
    return out

def _reduced_final_match(p1, p2):
    """Final syllables share their consonants and at least one vowel is a reduced schwa:
    CMU writes 'ornament' with AH0 where Shakespeare's rhyme with 'content' is audible."""
    for v1, c1 in _final_vowel_and_coda(p1):
        for v2, c2 in _final_vowel_and_coda(p2):
            if c1 == c2 and (v1 in _REDUCED or v2 in _REDUCED):
                return True
    return False

def _n_syll(prons):
    return sum(ph[-1].isdigit() for ph in prons[0])

@lru_cache(maxsize=None)
def rhyme_level(w1, w2, mode="cmu"):
    """
    1.0  perfect rhyme (same stressed vowel + everything after it) or a listed historical rhyme
    0.5  weaker rhyme: identical word; same final syllable in a polysyllable (dignity/mutiny);
         Early Modern '-y' = 'eye' (perjury/eye); same stressed vowel, different coda (slant)
    0.0  otherwise.
    The old rule also accepted 'same last two letters' or 'same last vowel LETTER', which made
    ~30% of random line-end pairs 'rhyme' (love/time).
    """
    w1, w2 = w1.lower().strip("'"), w2.lower().strip("'")
    if not w1 or not w2:
        return 0.0
    if w1 == w2:
        return 0.5
    if w2 in HISTORICAL_RHYMES.get(w1, ()):
        return 1.0
    p1, p2 = word_prons(w1, mode), word_prons(w2, mode)
    if p1 and p2:
        r1, r2 = _rhyme_parts(p1), _rhyme_parts(p2)
        if r1 & r2:
            return 1.0
        if max(_n_syll(p1), _n_syll(p2)) > 1 and _final_syllable_parts(p1) & _final_syllable_parts(p2):
            return 0.5
        if max(_n_syll(p1), _n_syll(p2)) > 1 and _reduced_final_match(p1, p2):
            return 0.5        # ornament/content: same final consonants, one vowel reduced by CMU
        ay1 = any(r[0] == "AY" for r in r1); ay2 = any(r[0] == "AY" for r in r2)
        if (ay1 and w2.endswith("y") and _n_syll(p2) > 1) or (ay2 and w1.endswith("y") and _n_syll(p1) > 1):
            return 0.5
        v1 = {r[0] for r in r1 if r}; v2 = {r[0] for r in r2 if r}
        return SLANT_LEVEL if v1 & v2 else 0.0
    # unknown word: only a 3-letter spelling match, and only as a weak rhyme
    if len(w1) >= 3 and len(w2) >= 3 and w1[-3:] == w2[-3:] and re.search(r"[aeiouy]", w1[-3:]):
        return 0.5
    return 0.0

def check_rhyme(w1, w2, mode="cmu", min_level=0.5):
    """Yes/no rhyme, used ONLY for scheme labels (ABAB / ABBA ...), never for scoring.
    Accepts weak rhymes: with min_level=1.0 the first run labelled Sonnet 1 'Irregular'
    because die/memory is an Early Modern '-y' rhyme (level 0.5)."""
    return rhyme_level(w1, w2, mode) >= min_level

# ---------- 5. The one window scorer -----------------------------------------
_SCHEMES = [((0, 2), (1, 3), 1.0),   # ABAB
            ((0, 3), (1, 2), 0.8),   # ABBA
            ((0, 1), (2, 3), 0.6)]   # AABB  (old AAAA branch was unreachable: AAAA also matches AABB)

def quatrain_score(w, mode="cmu"):
    best = 0.2 * max(rhyme_level(w[0], w[2], mode), rhyme_level(w[1], w[3], mode))
    for a, b, val in _SCHEMES:
        best = max(best, val * min(rhyme_level(w[a[0]], w[a[1]], mode),
                                   rhyme_level(w[b[0]], w[b[1]], mode)))
    return best

def _shape_credit(nq, rem):
    if nq == 3 and rem == 0:
        return 1.0
    if nq == 2 and rem == 0:
        return SHORT_FORM_CREDIT
    return 0.75

def window_parts(lines, mode="cmu"):
    """Component scores (meter, syllable, rhyme, couplet) for one window, each 0-1."""
    if len(lines) < 8:
        return np.zeros(4)
    feats = [line_features(l, mode) for l in lines]
    meter = np.mean([f[0] for f in feats])
    syll = np.mean([f[1] for f in feats])
    ends = [f[2] for f in feats]
    body = ends[:-2]
    nq, rem = divmod(len(body), 4)
    q = [quatrain_score(body[i:i + 4], mode) for i in range(0, nq * 4, 4)]
    rhyme = (np.mean(q) if q else 0.0) * _shape_credit(nq, rem)
    couplet = rhyme_level(ends[-2], ends[-1], mode)
    return np.array([meter, syll, rhyme, couplet])

def weights_vector(weights=None):
    weights = weights or WEIGHTS
    return np.array([weights[c] for c in COMPONENTS])

def score_window(lines, weights=None, mode="cmu"):
    return float(window_parts(lines, mode) @ weights_vector(weights))

# ---------- 6. One line cleaner for every cell (bug 6, method 2) ------------
_SPEAKER_RE = re.compile(r"^[A-Z][A-Z’'\-,&\. ]*[A-Z]\.$")
_DIRECTION_RE = re.compile(r"^(Enter|Exit|Exeunt|Re-enter|Flourish|Alarum|Sennet|Music|Thunder)\b")
_INLINE_DIRECTION_RE = re.compile(r"\[[^\]]*\]")
_NUMBER_RE = re.compile(r"^[0-9IVXLC]+\.?$")

def clean_play_lines(lines):
    """
    Turn a work's raw lines into spoken text only.
    Returns clean_data = [(text, speaker)], boundaries = [(clean_idx, label, 'act'|'scene')].
    Removes: blank lines, ACT/SCENE headings, speaker tags (ROMEO.), stage directions
    ([Exit.], Enter ..., inline [_Aside_]) and italic underscores.
    Every scan, classifier window and plot uses THIS function, so line index i means
    the same line everywhere.
    """
    clean_data, boundaries = [], []
    speaker = "UNKNOWN"
    # Third run: Gutenberg sometimes drops the period after a speaker tag ("PROLOGUE",
    # "FORD", "SNOUT"), so the tag was discarded and the speech credited to the previous
    # speaker (MND's Prologue came out as THESEUS). A bare tag counts if the same play
    # also uses that name with a period, or it is PROLOGUE/EPILOGUE/CHORUS (line 0 is the title).
    dotted = {l.strip().translate(_APOS)[:-1] for l in lines
              if len(l.strip()) < 40 and _SPEAKER_RE.match(l.strip().translate(_APOS))}
    dotted |= {"PROLOGUE", "EPILOGUE", "CHORUS"}
    in_direction = False                      # inside a [ ... ] that spans several lines
    for li, raw in enumerate(lines):
        s = raw.strip()
        if not s:
            continue
        if in_direction:                      # skip until the closing bracket
            if "]" in s:
                in_direction = False
                s = s.split("]", 1)[1].strip()
                if not s:
                    continue
            else:
                continue
        up = s.upper()
        if up.startswith("ACT ") or up.startswith("SCENE "):
            kind = "act" if up.startswith("ACT ") else "scene"
            boundaries.append((len(clean_data), s.split(".")[0], kind))
            continue
        if up in ("PROLOGUE.", "EPILOGUE."):  # a speaker in MND 5.1, a section elsewhere: both
            boundaries.append((len(clean_data), s.rstrip("."), "scene"))
            speaker = s.rstrip(".")
            continue
        if len(s) < 40 and _SPEAKER_RE.match(s.translate(_APOS)):
            speaker = s.rstrip(".")
            continue
        if li > 0 and s.translate(_APOS) in dotted and li + 1 < len(lines) and lines[li + 1].strip():
            speaker = s
            continue
        if _DIRECTION_RE.match(s):
            continue
        # First run: lines like "[_To Juliet._] If I profane with my unworthiest hand"
        # were dropped whole because they start with "[". Strip the bracketed part
        # and keep whatever verse follows it.
        s = _INLINE_DIRECTION_RE.sub("", s)
        if "[" in s:                          # an opening bracket with no close on this line
            # Second run: an unclosed "[" could swallow verse until the next "]" anywhere
            # below. Only treat it as a multi-line direction if "]" appears within 4 lines;
            # otherwise it's a stray bracket: drop the character, keep the text.
            if any("]" in nxt for nxt in lines[li + 1: li + 5]):
                s, in_direction = s.split("[", 1)[0], True
            else:
                s = s.replace("[", "")
        s = s.replace("_", "").strip()
        if not s or s.isupper() or _NUMBER_RE.match(s):   # titles, sonnet numbers, empty
            continue
        clean_data.append((s, speaker))
    return clean_data, boundaries

# ---------- 7. Scanning with caching -----------------------------------------
SCAN_CACHE = {}

def scan_work(title, lines, mode="cmu", window_sizes=WINDOW_SIZES, use_cache=True):
    """
    Score every start position with every window size, fast.
    Per-line features are computed once; per-quatrain and per-couplet rhyme scores are
    computed once per start line; a window's score is then assembled from those pieces.
    (Gives exactly the same numbers as window_parts on each window.)
    Returns dict with:
      parts  : array (n_lines, n_sizes, 4) of component scores (NaN where the window doesn't fit)
      clean_data, boundaries, window_sizes
    Re-weighting later is a dot product, so tuning WEIGHTS never needs a re-scan.
    """
    key = (title, mode, tuple(window_sizes), SHORT_FORM_CREDIT)
    if use_cache and key in SCAN_CACHE:
        return SCAN_CACHE[key]
    clean_data, boundaries = clean_play_lines(lines)
    text = [t for t, _ in clean_data]
    n = len(text)
    feats = [line_features(t, mode) for t in text]
    meter = np.array([f[0] for f in feats]); syll = np.array([f[1] for f in feats])
    ends = [f[2] for f in feats]
    cm = np.concatenate([[0.0], np.cumsum(meter)]); cs = np.concatenate([[0.0], np.cumsum(syll)])
    quat = np.array([quatrain_score(ends[j:j + 4], mode) if j + 4 <= n else 0.0 for j in range(n)])
    coup = np.array([rhyme_level(ends[j], ends[j + 1], mode) if j + 1 < n else 0.0 for j in range(n)])
    parts = np.full((n, len(window_sizes), 4), np.nan)
    for k, w in enumerate(window_sizes):
        if w < 8 or w > n:
            continue
        nq, rem = divmod(w - 2, 4)
        mult = _shape_credit(nq, rem)
        starts = np.arange(0, n - w + 1)
        parts[starts, k, 0] = (cm[starts + w] - cm[starts]) / w
        parts[starts, k, 1] = (cs[starts + w] - cs[starts]) / w
        if nq:
            parts[starts, k, 2] = np.mean([quat[starts + 4 * q] for q in range(nq)], axis=0) * mult
        else:
            parts[starts, k, 2] = 0.0
        parts[starts, k, 3] = coup[starts + w - 2]
    res = {"parts": parts, "clean_data": clean_data, "boundaries": boundaries,
           "window_sizes": tuple(window_sizes)}
    SCAN_CACHE[key] = res
    return res

def scores_from_scan(scan, weights=None):
    """Best score over window sizes at each start line, and which size won."""
    s = np.nan_to_num(scan["parts"] @ weights_vector(weights), nan=0.0)
    if s.size == 0:
        return np.zeros(0), np.zeros(0, dtype=int)
    best = s.argmax(axis=1)
    return s.max(axis=1), np.array(scan["window_sizes"])[best]

def analyze_work(title, lines, weights=None, mode="cmu", verbose=True):
    """Drop-in replacement for the old analyze_work copies: (scores, clean_data, boundaries)."""
    if verbose:
        print(f"Analyzing {title}...")
    scan = scan_work(title, lines, mode)
    scores, _ = scores_from_scan(scan, weights)
    return list(scores), scan["clean_data"], scan["boundaries"]

# ---------- 8. Non-maximum suppression (one copy) ----------------------------
def nms_top_k(scores, k=10, radius=NMS_RADIUS):
    """Indices of the k best positions, none within `radius` lines of a better one."""
    scores = np.asarray(scores)
    taken = np.zeros(len(scores), dtype=bool)
    picked = []
    for idx in np.argsort(scores)[::-1]:
        if k is not None and len(picked) >= k:
            break
        if taken[idx]:
            continue
        picked.append(int(idx))
        taken[max(0, idx - radius + 1): idx + radius] = True
    return picked

def window_lines(clean_data, idx, size=14):
    """Lines of a window, clipped at the end of the work (bug 11)."""
    return [t for t, _ in clean_data[idx: min(idx + size, len(clean_data))]]


# ---------- 9. Shared parsing helpers (one copy each) ------------------------
NON_PLAY_TITLES = {"THE SONNETS", "A LOVER’S COMPLAINT", "THE PASSIONATE PILGRIM",
                   "THE PHOENIX AND THE TURTLE", "THE RAPE OF LUCRECE", "VENUS AND ADONIS",
                   "PLAY SONNETS"}      # PLAY SONNETS is a synthetic copy of text already in the plays
_SONNET_HEADER_RE = re.compile(r"^(\d+|[XVILC]+)\.?$")

def parse_sonnets_map(works, min_lines=10):
    """{sonnet number: [lines]} for THE SONNETS (replaces 4 separate parsing loops)."""
    out, cur, num = {}, [], 0
    for line in works.get("THE SONNETS", []):
        s = line.strip()
        if not s or s == "THE SONNETS":
            continue
        if _SONNET_HEADER_RE.match(s):
            if len(cur) >= min_lines:
                out[num] = cur
            cur = []
            try: num = int(s.strip("."))
            except ValueError: num += 1
        else:
            cur.append(s)
    if len(cur) >= min_lines:
        out[num] = cur
    return out

def parse_play_sonnets(works):
    """[(label, [lines])] for the synthetic PLAY SONNETS work, speaker tags removed."""
    out, label, cur = [], None, []
    for line in works.get("PLAY SONNETS", []):
        m = re.match(r"^=== PLAY SONNET \d+: (.*) \(\d+ lines\) ===$", line)
        if m:
            if label is not None: out.append((label, cur))
            label, cur = m.group(1), []
        elif line.strip() and not _SPEAKER_RE.match(line.strip().translate(_APOS)):
            cur.append(line.strip())
    if label is not None: out.append((label, cur))
    return out

def get_sonnet_date(num):
    """Approximate composition date of a sonnet (one copy; used by cells 9 and 10b)."""
    if 1 <= num <= 60: return 1596
    elif 61 <= num <= 77: return 1594.5
    elif 78 <= num <= 86: return 1599
    elif 87 <= num <= 103: return 1594.5
    elif 104 <= num <= 126: return 1602
    elif 127 <= num <= 144: return 1593
    elif num == 145: return 1582
    elif 146 <= num <= 154: return 1593
    return None

# ---------- 10. Statistics for dependent data (method 5) --------------------
def ar1_wavelet_threshold(sig, scales, wavelet="cmor1.5-1.0", smooth_sigma=(2, 2),
                          n_surr=30, q=0.95, seed=0):
    """
    Red-noise significance for a wavelet power spectrum (Torrence & Compo 1998, by simulation).

    The detection signal comes from overlapping windows, so neighbouring values are strongly
    correlated: it behaves like red noise, which naturally has lots of power at long periods.
    A white-noise threshold (variance * chi2) therefore calls far too much "significant".

    Here we fit AR(1) (lag-1 autocorrelation alpha), simulate n_surr AR(1) series with the
    same alpha and variance, push each through the SAME wavelet + smoothing, and take the
    q-quantile of their power at each scale. Simulating (instead of using the analytic
    formula) makes the threshold automatically match pywt's normalisation and the smoothing.

    Returns (threshold per scale, alpha). Significant where power_smooth / threshold[:, None] > 1.
    """
    sig = np.asarray(sig, float) - np.mean(sig)
    n = len(sig)
    var = np.var(sig)
    alpha = float(np.corrcoef(sig[:-1], sig[1:])[0, 1]) if n > 2 and var > 0 else 0.0
    alpha = min(max(alpha, 0.0), 0.99)
    rng = np.random.default_rng(seed)
    pooled = []
    for _ in range(n_surr):
        e = rng.normal(0, np.sqrt(var * (1 - alpha ** 2)), n)
        x = np.empty(n); x[0] = rng.normal(0, np.sqrt(var))
        for t in range(1, n):
            x[t] = alpha * x[t - 1] + e[t]
        c, _ = pywt.cwt(x - x.mean(), scales, wavelet, method="fft")
        pooled.append(gaussian_filter(np.abs(c) ** 2, sigma=smooth_sigma))
    pooled = np.stack(pooled)                        # surrogates x scales x time
    # Use only points inside the cone of influence: edge-affected power is artificially low
    # and would drag the threshold down.
    t = np.arange(n); edge = np.minimum(t, n - 1 - t)
    thr = np.empty(len(scales))
    for k, sc in enumerate(scales):
        inside = edge >= np.sqrt(2) * sc
        vals = pooled[:, k, inside] if inside.sum() >= 10 else pooled[:, k, :]
        thr[k] = np.quantile(vals, q)
    return thr, alpha

def window_permutation_test(values_per_line, weights_per_line, starts, size=14,
                            n_perm=5000, seed=0, min_gap=None):
    """
    Is the rate in the chosen windows unusual for this play?  The unit is the WINDOW, not
    the line: lines inside one window share speaker, scene and topic, so treating 280 lines
    as 280 independent samples (a t-test) overstates the evidence.

    values_per_line: e.g. rare-word count per line;  weights_per_line: tokens per line.
    observed = mean over the chosen windows of (sum values / sum weights).
    null     = the same statistic for len(starts) randomly placed, non-overlapping windows.
    Returns dict(observed, null_mean, null_sd, p_two_sided, p_greater).
    """
    v = np.concatenate([[0.0], np.cumsum(values_per_line)])
    w = np.concatenate([[0.0], np.cumsum(weights_per_line)])
    n = len(values_per_line)
    def rate(st):
        st = np.asarray(st)
        num, den = v[st + size] - v[st], w[st + size] - w[st]
        return float(np.mean(num / np.maximum(den, 1)))
    starts = [s for s in starts if s + size <= n]
    k = len(starts)
    obs = rate(starts)
    rng = np.random.default_rng(seed)
    gap = min_gap or size
    null = np.empty(n_perm)
    for b in range(n_perm):
        picked, taken = [], np.zeros(n, bool)
        for s0 in rng.permutation(n - size + 1):
            if not taken[s0]:
                picked.append(s0)
                taken[max(0, s0 - gap + 1): s0 + gap] = True
                if len(picked) == k:
                    break
        null[b] = rate(picked)
    mu = null.mean()
    return {"observed": obs, "null_mean": mu, "null_sd": null.std(), "n_windows": k,
            "p_two_sided": (1 + np.sum(np.abs(null - mu) >= abs(obs - mu))) / (1 + n_perm),
            "p_greater": (1 + np.sum(null >= obs)) / (1 + n_perm)}

def phase_wavelet_threshold(sig, scales, wavelet="cmor1.5-1.0", smooth_sigma=(2, 2),
                            n_surr=30, q=0.95, seed=0):
    """
    Second run: the AR(1) null still called 46% of Midsummer's spectrum significant.
    AR(1) has a short memory, but a max-over-18-line-windows signal stays correlated for
    ~18 lines, so AR(1) underestimates its long-period power.

    Phase-randomized surrogates keep the signal's OWN average power spectrum and scramble
    only the phases (Theiler et al. 1992). Anything significant against them is power that
    is LOCALIZED in time (a burst at one place in the play), not a property of the whole
    signal -- which is what "a sonnet-rich passage here" would look like.
    Returns threshold per scale (95th percentile inside the cone of influence).
    """
    sig = np.asarray(sig, float) - np.mean(sig)
    n = len(sig)
    amp = np.abs(np.fft.rfft(sig))
    rng = np.random.default_rng(seed)
    pooled = []
    for _ in range(n_surr):
        ph = rng.uniform(0, 2 * np.pi, len(amp)); ph[0] = 0
        if n % 2 == 0: ph[-1] = 0
        x = np.fft.irfft(amp * np.exp(1j * ph), n)
        c, _ = pywt.cwt(x, scales, wavelet, method="fft")
        pooled.append(gaussian_filter(np.abs(c) ** 2, sigma=smooth_sigma))
    pooled = np.stack(pooled)
    t = np.arange(n); edge = np.minimum(t, n - 1 - t)
    thr = np.empty(len(scales))
    for k, sc in enumerate(scales):
        inside = edge >= np.sqrt(2) * sc
        vals = pooled[:, k, inside] if inside.sum() >= 10 else pooled[:, k, :]
        thr[k] = np.quantile(vals, q)
    return thr

def show_raw(phrase, before=3, after=6):
    """Print the RAW Gutenberg lines around a phrase, exactly as the cleaner receives them.
    Use it when a known sonnet is 'not in the corpus' or a speaker label looks wrong."""
    key = re.sub(r"[^a-z0-9]", "", phrase.lower())[:25]
    if FULL_TEXT is None:
        raise RuntimeError("set sonnet_toolkit.FULL_TEXT = full_text first (cell 4 does this)")
    raw = FULL_TEXT.split("\n")
    hits = [i for i, l in enumerate(raw) if key and key in re.sub(r"[^a-z0-9]", "", l.lower())]
    if not hits:
        print(f"'{phrase}' not found in the raw text"); return
    for i in hits[:3]:
        print(f"--- raw lines {i - before}..{i + after} ---")
        for j in range(max(0, i - before), min(len(raw), i + after + 1)):
            print(f"{j:7d} | {raw[j]!r}")


# ---------- 11. Windows and known-sonnet location (version 8) ---------------
# One window builder for every consumer: scan_work's clean_data, the classifier
# windows (cell 6) and the evaluation cell (8b) all index the SAME cleaned lines,
# so window i of a play is the same text everywhere (tests/test_cleaner.py).
def normalize_for_match(s):
    """Lowercase, strip everything that is not a-z or 0-9."""
    return re.sub(r"[^a-z0-9]", "", s.lower())

def build_play_windows(works, size=14):
    """All full `size`-line windows of every play: (texts, meta), meta = (title, start, first line)."""
    texts, meta = [], []
    for title, lines in works.items():
        if title in NON_PLAY_TITLES:
            continue
        clean_data = clean_play_lines(lines)[0]
        for i in range(0, len(clean_data) - size + 1):
            block = window_lines(clean_data, i, size)
            texts.append("\n".join(block))
            meta.append((title, i, block[0]))
    return texts, meta

def locate_passage(works, first_line, min_ratio=0.85):
    """(play, cleaned line index) where a passage starts, or None.
    Exact prefix match on the normalized first line, else the closest line if >= min_ratio
    similar (spelling differs between editions, e.g. "y-slackèd" vs "yslaked")."""
    plays = {t: clean_play_lines(l)[0] for t, l in works.items() if t not in NON_PLAY_TITLES}
    key = normalize_for_match(first_line)[:30]
    for title, cd in plays.items():
        for i, (t, _) in enumerate(cd):
            if normalize_for_match(t).startswith(key):
                return title, i
    best = (0.0, None)
    for title, cd in plays.items():
        for i, (t, _) in enumerate(cd):
            r = difflib.SequenceMatcher(None, key, normalize_for_match(t)[:30]).ratio()
            if r > best[0]:
                best = (r, (title, i))
    return best[1] if best[0] >= min_ratio else None

# ---------- 12. Ranking against known sonnets (moved from cell 8b, version 8) --
TOLERANCE = 3          # a hit within 3 lines of the true start counts (more for 16-18-line passages)

def ranked_hits(score_by_play, radius=NMS_RADIUS):
    """Pool NMS peaks from every play into one ranked list [(play, line, score)]."""
    hits = []
    for play, sc in score_by_play.items():
        sc = np.asarray(sc)
        for idx in nms_top_k(sc, k=None, radius=radius):
            hits.append((float(sc[idx]), play, idx))
    hits.sort(key=lambda h: -h[0])
    return [(p, i, s) for s, p, i in hits]

def known_ranks(hits, known, tol=TOLERANCE):
    """1-based rank of the first hit matching each known sonnet (inf if never hit)."""
    ranks = []
    for _, row in known.iterrows():
        slack = max(tol, int(row["n_lines"]) - 12)     # longer passages (15-18 lines) can peak a few lines in
        r = next((k + 1 for k, (p, i, _) in enumerate(hits)
                  if p == row["play"] and abs(i - row["line"]) <= slack), np.inf)
        ranks.append(r)
    return np.array(ranks, dtype=float)


# ---------- 13. Corpus parsing (moved from cells 2-3, version 8) ------------
def strip_gutenberg(text):
    """Remove the Project Gutenberg header and footer."""
    start_marker = "*** START OF THE PROJECT GUTENBERG EBOOK"
    end_marker = "*** END OF THE PROJECT GUTENBERG EBOOK"
    start_idx = text.find(start_marker)
    end_idx = text.find(end_marker)
    if start_idx != -1:
        text = text[start_idx + len(start_marker):]
    if end_idx != -1:
        text = text[:end_idx]
    return text

works_list = [
    "THE SONNETS", "ALL’S WELL THAT ENDS WELL", "THE TRAGEDY OF ANTONY AND CLEOPATRA",
    "AS YOU LIKE IT", "THE COMEDY OF ERRORS", "THE TRAGEDY OF CORIOLANUS",
    "CYMBELINE", "THE TRAGEDY OF HAMLET, PRINCE OF DENMARK",
    "THE FIRST PART OF KING HENRY THE FOURTH", "THE SECOND PART OF KING HENRY THE FOURTH",
    "THE LIFE OF KING HENRY THE FIFTH", "THE FIRST PART OF HENRY THE SIXTH",
    "THE SECOND PART OF KING HENRY THE SIXTH", "THE THIRD PART OF KING HENRY THE SIXTH",
    "KING HENRY THE EIGHTH", "THE LIFE AND DEATH OF KING JOHN",
    "THE TRAGEDY OF JULIUS CAESAR", "THE TRAGEDY OF KING LEAR",
    "LOVE’S LABOUR’S LOST", "THE TRAGEDY OF MACBETH",
    "MEASURE FOR MEASURE", "THE MERCHANT OF VENICE",
    "THE MERRY WIVES OF WINDSOR", "A MIDSUMMER NIGHT’S DREAM",
    "MUCH ADO ABOUT NOTHING", "THE TRAGEDY OF OTHELLO, THE MOOR OF VENICE",
    "PERICLES, PRINCE OF TYRE", "KING RICHARD THE SECOND",
    "KING RICHARD THE THIRD", "THE TRAGEDY OF ROMEO AND JULIET",
    "THE TAMING OF THE SHREW", "THE TEMPEST",
    "THE LIFE OF TIMON OF ATHENS", "THE TRAGEDY OF TITUS ANDRONICUS",
    "TROILUS AND CRESSIDA", "TWELFTH NIGHT; OR, WHAT YOU WILL",
    "THE TWO GENTLEMEN OF VERONA", "THE TWO NOBLE KINSMEN",
    "THE WINTER’S TALE", "A LOVER’S COMPLAINT",
    "THE PASSIONATE PILGRIM", "THE PHOENIX AND THE TURTLE",
    "THE RAPE OF LUCRECE", "VENUS AND ADONIS",
    "PLAY SONNETS"  # synthetic work we add manually after parsing
]

def parse_works(text):
    # Skip the contents table itself
    contents_end = text.find("VENUS AND ADONIS") + len("VENUS AND ADONIS")
    current_pos = max(0, contents_end)

    sorted_indices = []
    for work in works_list:
        idx = text.find(work, current_pos)
        if idx != -1:
            sorted_indices.append((idx, work))

    sorted_indices.sort()

    parsed_data = {}

    for i in range(len(sorted_indices)):
        start_idx, title = sorted_indices[i]
        end_idx = sorted_indices[i+1][0] if i + 1 < len(sorted_indices) else len(text)

        work_text = text[start_idx:end_idx]
        lines = [l.strip() for l in work_text.split('\n') if l.strip()]
        parsed_data[title] = lines

    return parsed_data
