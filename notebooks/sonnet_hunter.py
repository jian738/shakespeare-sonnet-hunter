# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.6
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown] id="xeiDyHC-fFQh"
# # Import modules

# %% colab={"base_uri": "https://localhost:8080/"} id="dbemM8Hk1mGX" outputId="8c995180-cd9d-4d2c-d3a5-e435cd1f82ee"
# Cell 1: Setup and Imports
# !pip install nltk
# !pip install g2p_en transformers torch scikit-learn
# !pip install --upgrade pip
# !pip cache purge
# !pip install PyWavelets
import os
import re
import csv
import ssl
import nltk
import pywt
import torch
import random
import requests
import zipfile
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection

from functools import lru_cache
from collections import defaultdict
from collections import Counter
import scipy.stats
import scipy.ndimage
from scipy import stats
from scipy.ndimage import gaussian_filter
from scipy.spatial import ConvexHull
from g2p_en import G2p
from transformers import BertTokenizer, BertModel

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler

nltk.download('cmudict')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger_eng')
from nltk.corpus import cmudict
from google.colab import files

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable, desc=None, unit=None):
        print(f"Progress: {desc}")
        return iterable

# Initialize CMU Dict
d = cmudict.dict()


# %% [markdown] id="OHkQozkb9dNa"
# # Part 1 - Introduction

# %% [markdown] id="m4ZfwV22fKcf"
# ## Download Shakespeare's Works

# %% colab={"base_uri": "https://localhost:8080/"} id="Obp7sm5p1mgE" outputId="175174b4-095c-40f8-c47c-9835e2cdc9e6"
# Cell 2: Data Acquisition
def download_shakespeare():
    url = "https://www.gutenberg.org/files/100/100-0.txt"
    response = requests.get(url)
    response.encoding = 'utf-8-sig'
    text = response.text

    # Remove Gutenberg Headers/Footers
    start_marker = "*** START OF THE PROJECT GUTENBERG EBOOK"
    end_marker = "*** END OF THE PROJECT GUTENBERG EBOOK"

    start_idx = text.find(start_marker)
    end_idx = text.find(end_marker)

    if start_idx != -1:
        text = text[start_idx + len(start_marker):]
    if end_idx != -1:
        text = text[:end_idx]

    return text

full_text = download_shakespeare()
print(f"Downloaded {len(full_text)} characters.")

# %% [markdown] id="GCmFNQHkfR9h"
# ## Prepare Texts for Analysis, Add Play Sonnets

# %% colab={"base_uri": "https://localhost:8080/"} id="Tb4sX9J_1muf" outputId="2c7fd1d6-f53e-42f8-ab2e-27a00de1a7e1"
# Cell 3: Parsing Plays, Sonnets, and PLAY SONNETS

# 1. Works list, now including PLAY SONNETS
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

# --- Helpers to build PLAY SONNETS from the full text ---

def _normalize_for_match(s):
    # Lowercase, strip everything that is not a-z or 0-9
    return re.sub(r'[^a-z0-9]', '', s.lower())

def find_sonnet_in_full_text(full_text, first_line, num_lines):
    """
    Find a passage by its first line anywhere in the full text,
    return num_lines non-blank lines starting from the match.
    """
    lines = full_text.split('\n')
    norm_first = _normalize_for_match(first_line)

    for idx, line in enumerate(lines):
        if _normalize_for_match(line).startswith(norm_first):
            collected = []
            j = idx
            while j < len(lines) and len(collected) < num_lines:
                if lines[j].strip():
                    collected.append(lines[j].rstrip())
                j += 1
            return collected

    print(f"WARNING: could not find passage starting with: {first_line!r}")
    return []

# --- Manual overrides for passages that Gutenberg layout breaks ---

manual_play_sonnets = {
    # Two Gentlemen of Verona, full 10-line sonnet
    "The Two Gentlemen of Verona, 3.1.140–9.": [
        "My thoughts do harbour with my Silvia nightly,",
        "And slaves they are to me that send them flying.",
        "O, could their master come and go as lightly,",
        "Himself would lodge where, senseless, they are lying.",
        "My herald thoughts in thy pure bosom rest them,",
        "While I, their king, that thither them importune,",
        "Do curse the grace that with such grace hath blest them,",
        "Because myself do want my servants’ fortune.",
        "I curse myself, for they are sent by me,",
        "That they should harbour where their lord should be.",
    ],

    # Edward III passage, cleaned into verse lines
    "Edward III, scene 2.322–3.": [
        "And let me have her likened to the sun –",
        "Say she hath thrice more splendour than the sun,",
        "That her perfections emulates the sun,",
        "That she breeds sweets as plenteous as the sun,",
        "That she doth thaw cold winter like the sun,",
        "That she doth cheer fresh summer, like the sun,",
        "That she doth dazzle gazers like the sun,",
        "And in this application to the sun",
        "Bid her be free and general as the sun,",
        "Who smiles upon the basest weed that grows",
        "As lovingly as on the fragrant rose.",
    ],

    # Comedy of Errors sonnet including the missing last line
    "The Comedy of Errors, 3.2.41–54.": [
        "But if that I am I, then well I know",
        "Your weeping sister is no wife of mine,",
        "Nor to her bed no homage do I owe.",
        "Far more, far more, to you do I decline.",
        "O, train me not, sweet mermaid, with thy note",
        "To drown me in thy sister’s flood of tears.",
        "Sing, siren, for thyself, and I will dote;",
        "Spread o’er the silver waves thy golden hairs,",
        "And as a bed I’ll take thee, and there lie,",
        "And, in that glorious supposition think",
        "He gains by death that hath such means to die.",
        "Let love, being light, be drowned if she sink!",
        "LUCIANA.",
        "What, are you mad, that you do reason so?",
        "Not mad, but mated – how, I do not know.",
    ],

    # Love’s Labour’s Lost 1.1.80–9, with the extra four lines
    "Love’s Labour’s Lost, 1.1.80–9.": [
        "Study me how to please the eye indeed",
        "By fixing it upon a fairer eye,",
        "Who dazzling so, that eye shall be his heed,",
        "And give him light that it was blinded by.",
        "Study is like the heaven’s glorious sun,",
        "That will not be deep-searched with saucy looks;",
        "Small have continual plodders ever won,",
        "Save base authority from others’ books.",
        "These earthly godfathers of heaven’s lights,",
        "That give a name to every fixed star,",
        "Have no more profit of their shining nights",
        "Than those that walk and wot not what they are.",
        "Too much to know is to know naught but fame,",
        "And every godfather can give a name.",
    ],

    # Love’s Labour’s Lost 1.1.159–74, WITHOUT "BEROWNE."
    "Love’s Labour’s Lost, 1.1.159–74.": [
        "Ay, that there is. Our court, you know, is haunted",
        "With a refined traveller of Spain,",
        "A man in all the world’s new fashion planted,",
        "That hath a mint of phrases in his brain;",
        "One who the music of his own vain tongue",
        "Doth ravish like enchanting harmony,",
        "A man of complements, whom right and wrong",
        "Have chose as umpire of their mutiny.",
        "This child of fancy, that Armado hight,",
        "For interim to our studies shall relate",
        "In high-born words the worth of many a knight",
        "From tawny Spain lost in the world’s debate.",
        "How you delight, my lords, I know not, I,",
        "But I protest I love to hear him lie,",
        "And I will use him for my minstrelsy.",
    ],

    # Love’s Labour’s Lost 4.3.24–39, full poem
    "Love’s Labour’s Lost, 4.3.24–39.": [
        "So sweet a kiss the golden sun gives not",
        "To those fresh morning drops upon the rose,",
        "As thy eye-beams, when their fresh rays have smote",
        "The night of dew that on my cheeks down flows.",
        "Nor shines the silver moon one half so bright",
        "Through the transparent bosom of the deep",
        "As doth thy face, through tears of mine give light.",
        "Thou shin’st in every tear that I do weep.",
        "No drop but as a coach doth carry thee;",
        "So ridest thou triumphing in my woe.",
        "Do but behold the tears that swell in me,",
        "And they thy glory through my grief will show.",
        "But do not love thyself; then thou wilt keep",
        "My tears for glasses, and still make me weep.",
        "O queen of queens, how far dost thou excel",
        "No thought can think, nor tongue of mortal tell.",
    ],

    # Love’s Labour’s Lost 5.2.401–15, WITHOUT "ROSALINE."
    "Love’s Labour’s Lost, 5.2.401–15.": [
        "O, never will I trust to speeches penned,",
        "Nor to the motion of a school-boy’s tongue,",
        "Nor never come in visor to my friend,",
        "Nor woo in rhyme like a blind harper’s song.",
        "Taffeta phrases, silken terms precise,",
        "Three-piled hyperboles, spruce affectation,",
        "Figures pedantical: these summer flies",
        "Have blown me full of maggot ostentation.",
        "I do forswear them, and I here protest,",
        "By this white glove—how white the hand, God knows!—",
        "Henceforth my wooing mind shall be expressed",
        "In russet yeas and honest kersey noes.",
        "And, to begin: wench, so God help me, law,",
        "My love to thee is sound, sans crack or flaw.",
    ],

    # Romeo and Juliet, 1.5.92–109: the shared sonnet (verse-only form)
    "Romeo and Juliet, 1.5.92–109.": [
        "If I profane with my unworthiest hand",
        "This holy shrine, the gentle sin is this:",
        "My lips, two blushing pilgrims, ready stand",
        "To smooth that rough touch with a tender kiss.",
        "Good pilgrim, you do wrong your hand too much,",
        "Which mannerly devotion shows in this;",
        "For saints have hands that pilgrims’ hands do touch,",
        "And palm to palm is holy palmers’ kiss.",
        "Have not saints lips, and holy palmers too?",
        "Ay, pilgrim, lips that they must use in prayer.",
        "O, then, dear saint, let lips do what hands do:",
        "They pray, grant thou, lest faith turn to despair.",
        "Saints do not move, though grant for prayers’ sake.",
        "Then move not while my prayer’s effect I take.",
    ],

    # Much Ado About Nothing, 3.1.107–16
    "Much Ado About Nothing, 3.1.107–16.": [
        "What fire is in mine ears? Can this be true?",
        "Stand I condemn’d for pride and scorn so much?",
        "Contempt, farewell! and maiden pride, adieu!",
        "No glory lives behind the back of such.",
        "And, Benedick, love on; I will requite thee,",
        "Taming my wild heart to thy loving hand:",
        "If thou dost love, my kindness shall incite thee",
        "To bind our loves up in a holy band;",
        "For others say thou dost deserve, and I",
        "Believe it better than reportingly.",
    ],

    # All’s Well That Ends Well, 3.4.4–17
    "All’s Well That Ends Well, 3.4.4–17.": [
        "I am Saint Jaques’ pilgrim, thither gone.",
        "Ambitious love hath so in me offended",
        "That barefoot plod I the cold ground upon,",
        "With sainted vow my faults to have amended.",
        "Write, write, that from the bloody course of war",
        "My dearest master, your dear son, may hie.",
        "Bless him at home in peace, whilst I from far",
        "His name with zealous fervour sanctify.",
        "His taken labours bid him me forgive;",
        "I, his despiteful Juno, sent him forth",
        "From courtly friends, with camping foes to live,",
        "Where death and danger dog the heels of worth.",
        "He is too good and fair for death and me;",
        "Whom I myself embrace to set him free.",
    ],

    # Pericles, scene 10.1–14: expanded full passage
    "Pericles, scene 10.1–14.": [
        "Now sleep y-slackèd hath the rout;",
        "No din but snores about the house,",
        "Made louder by the o’erfed breast",
        "Of this most pompous marriage feast.",
        "The cat, with eyne of burning coal,",
        "Now couches fore the mouse’s hole;",
        "And crickets sing at the oven’s mouth,",
        "Are the blither for their drouth.",
        "Hymen hath brought the bride to bed,",
        "Where, by the loss of maidenhead,",
        "A babe is moulded. Be attent,",
        "And time that is so briefly spent",
        "With your fine fancies quaintly eche:",
        "What’s dumb in show I’ll plain with speech.",
    ],

    # Pericles, scene 21.225–35: explicit 10-line sonnet, NO stage directions
    "Pericles, scene 21.225–35.": [
        "My temple stands in Ephesus. Hie thee thither,",
        "And do upon mine altar sacrifice.",
        "There when my maiden priests are met together,",
        "At large discourse thy fortunes in this wise:",
        "With a full voice before the people all,",
        "Reveal how thou at sea didst lose thy wife.",
        "To mourn thy crosses, with thy daughter’s, call",
        "And give them repetition to the life.",
        "Perform my bidding, or thou liv’st in woe;",
        "Do ’t, and rest happy, by my silver bow.",
    ],
}

# The 22 “play sonnets”, with target counts (used only when not overridden)
play_sonnet_specs = [
    {
        "label": "The Two Gentlemen of Verona, 3.1.140–9.",
        "first_line": "My thoughts do harbour with my Silvia nightly",
        "num_lines": 10,
    },
    {
        "label": "Edward III, scene 2.322–3.",
        "first_line": "And let me have her likened to the sun",
        "num_lines": 11,
    },
    {
        "label": "The Comedy of Errors, 3.2.41–54.",
        "first_line": "But if that I am I, then well I know",
        "num_lines": 15,
    },
    {
        "label": "Love’s Labour’s Lost, 1.1.80–9.",
        "first_line": "Study me how to please the eye indeed",
        "num_lines": 14,
    },
    {
        "label": "Love’s Labour’s Lost, 1.1.159–74.",
        "first_line": "Ay, that there is. Our court, you know, is haunted",
        "num_lines": 16,
    },
    {
        "label": "Love’s Labour’s Lost, 4.2.106–19.",
        "first_line": "If love make me forsworn, how shall I swear to love?",
        "num_lines": 14,
    },
    {
        "label": "Love’s Labour’s Lost, 4.3.24–39.",
        "first_line": "So sweet a kiss the golden sun gives not",
        "num_lines": 16,
    },
    {
        "label": "Love’s Labour’s Lost, 4.3.57–70.",
        "first_line": "Did not the heavenly rhetoric of thine eye,",
        "num_lines": 14,
    },
    {
        "label": "Love’s Labour’s Lost, 5.2.401–15.",
        "first_line": "O, never will I trust to speeches penned,",
        "num_lines": 15,
    },
    {
        "label": "Romeo and Juliet, Prologue.",
        "first_line": "Two households, both alike in dignity",
        "num_lines": 14,
    },
    {
        "label": "Romeo and Juliet, 1.5.92–109.",
        "first_line": "If I profane with my unworthiest hand",
        "num_lines": 14,
    },
    {
        "label": "Romeo and Juliet, 2.0. Introduction to Act Two.",
        "first_line": "Now old desire doth in his deathbed lie,",
        "num_lines": 14,
    },
    {
        "label": "A Midsummer Night’s Dream, 5.1.108–17.",
        "first_line": "If we offend, it is with our good will.",
        "num_lines": 10,
    },
    {
        "label": "Much Ado About Nothing, 3.1.107–16.",
        "first_line": "What fire is in mine ears? Can this be true?",
        "num_lines": 10,
    },
    {
        "label": "Henry V, Epilogue.",
        "first_line": "Thus far, with rough and all-unable pen,",
        "num_lines": 14,
    },
    {
        "label": "As You Like It, 3.2.1–10.",
        "first_line": "Hang there, my verse, in witness of my love.",
        "num_lines": 10,
    },
    {
        "label": "Troilus and Cressida, 1.2.278–91.",
        "first_line": "Words, vows, gifts, tears, and love’s full sacrifice",
        "num_lines": 14,
    },
    {
        "label": "All’s Well That Ends Well, 1.1.212–225.",
        "first_line": "Our remedies oft in ourselves do lie,",
        "num_lines": 14,
    },
    {
        "label": "All’s Well That Ends Well, 3.4.4–17.",
        "first_line": "I am Saint Jaques’ pilgrim, thither gone.",
        "num_lines": 14,
    },
    {
        "label": "Pericles, scene 10.1–14.",
        "first_line": "Now sleep y-slackèd hath the rout,",
        "num_lines": 14,
    },
    {
        "label": "Pericles, scene 21.225–35.",
        "first_line": "My temple stands in Ephesus. Hie thee thither,",
        "num_lines": 10,
    },
    {
        "label": "Cymbeline, 5.5.187–204.",
        "first_line": "No more, you petty spirits of region low,",
        "num_lines": 18,
    },
]

def build_play_sonnets(full_text, specs, manual_passages=None):
    """
    Construct the PLAY SONNETS work as a list of lines.
    Each passage is preceded by a header line.
    """
    if manual_passages is None:
        manual_passages = {}

    play_sonnets_lines = []

    for i, spec in enumerate(specs, 1):
        label = spec["label"]

        if label in manual_passages:
            sonnet_lines = manual_passages[label]
        else:
            sonnet_lines = find_sonnet_in_full_text(
                full_text,
                spec["first_line"],
                num_lines=spec["num_lines"],
            )

        # Normalize: strip leading/trailing whitespace from every line
        sonnet_lines = [l.strip() for l in sonnet_lines]

        actual_len = len(sonnet_lines)
        header = f"=== PLAY SONNET {i}: {label} ({actual_len} lines) ==="
        play_sonnets_lines.append(header)
        play_sonnets_lines.extend(sonnet_lines)
        play_sonnets_lines.append("")  # blank separator

    return play_sonnets_lines

# --- Build works, then synthesize PLAY SONNETS ---

works = parse_works(full_text)

# Create the synthetic PLAY SONNETS entry using search + manual overrides
works["PLAY SONNETS"] = build_play_sonnets(
    full_text,
    play_sonnet_specs,
    manual_passages=manual_play_sonnets
)

print(f"Parsed {len(works)} works (including PLAY SONNETS).")
print(f"First few lines of Sonnets: {works.get('THE SONNETS', [])[:5]}")

# Print out all PLAY SONNETS so we can be sure they were captured
print("\n" + "=" * 60)
print("ALL PLAY SONNETS")
print("=" * 60 + "\n")

for line in works["PLAY SONNETS"]:
    print(line)



# %% [markdown] id="QFPLUxK3fVI6"
# ## Prepare Natural Language Processing Tools

# %% colab={"base_uri": "https://localhost:8080/"} id="YhmLesjY1m9l" outputId="1efb992e-80c0-4f06-adf2-2721407048a8"
# Cell 4: NLP Tools (Syllables, Stress, Rhyme)
def count_syllables(word):
    word = word.lower().strip(".,;:!?()[]\"'")
    if not word: return 0
    if word in d:
        return [len(list(y for y in x if y[-1].isdigit())) for x in d[word]][0]
    else:
        # Fallback: count vowels
        return len(re.findall(r'[aeiouy]+', word))

def get_rhyme_part(word):
    word = word.lower().strip(".,;:!?()[]\"'")
    if not word or word not in d:
        return None

    pronunciation = d[word][0]
    # Find the last stressed vowel
    for i in range(len(pronunciation) - 1, -1, -1):
        phone = pronunciation[i]
        if phone[-1].isdigit() and int(phone[-1]) > 0: # Stressed
            return tuple(pronunciation[i:])
    return tuple(pronunciation[-1:]) # Fallback to last sound

def get_stress_pattern(line):
    # Simplified stress pattern: 1 for stressed, 0 for unstressed
    words = re.findall(r"\w+", line)
    pattern = []
    for w in words:
        w_lower = w.lower()
        if w_lower in d:
            # Take first pronunciation
            pron = d[w_lower][0]
            # Extract stresses
            stresses = [int(p[-1]) for p in pron if p[-1].isdigit()]
            # Simplify: 1 or 2 is stress (1), 0 is unstressed (0)
            pattern.extend([1 if s > 0 else 0 for s in stresses])
        else:
            # Guess: 1 syllable = 1, >1 = 101...
            syls = count_syllables(w)
            pattern.extend([1] * syls) # Very rough guess
    return pattern

def is_iambic(pattern):
    # Ideal iambic pentameter: 0101010101 (10 syllables)
    # We allow variations.
    if len(pattern) < 8 or len(pattern) > 12:
        return 0.0

    # Compare with ideal 0101010101...
    ideal = [0, 1] * 6
    ideal = ideal[:len(pattern)]

    matches = sum(1 for x, y in zip(pattern, ideal) if x == y)
    return matches / len(pattern)

print("NLP Tools Ready.")


# %% [markdown] id="HcUytcCionud"
# # Part 2 - Classifier

# %% [markdown] id="KV_gp-CmosAH"
# ## Preparation text & Detector

# %% colab={"base_uri": "https://localhost:8080/"} id="JJFFwmZ_ox1_" outputId="be95656e-f045-4dc4-e2e4-05ee7848a80b"
# Cell 5: Structural Feature Extraction & Improved Sonnet Detector
# Note: We extend the existing NLP tools with a class-based detector for feature extraction.

class SonnetDetectorWrapper:
    """
    Wrapper around the global NLP functions and dictionary 'd'
    to provide a consistent interface for the StructuralFeatures transformer.
    Includes the RELAXED rhyme check logic.
    """
    def __init__(self, cmu_dict):
        self.d = cmu_dict

    def count_syllables(self, word):
        # Re-uses the global function logic or implements it directly
        word = word.lower().strip(".,;:!?()[]\"'")
        if not word: return 0
        if word in self.d:
            return [len(list(y for y in x if y[-1].isdigit())) for x in self.d[word]][0]
        else:
            return len(re.findall(r'[aeiouy]+', word))

    def get_rhyme_part(self, word):
        word = word.lower().strip(".,;:!?()[]\"'")
        if not word or word not in self.d: return None
        pronunciation = self.d[word][0]
        # Strategy 1: Stressed vowel to end
        for i in range(len(pronunciation) - 1, -1, -1):
            phone = pronunciation[i]
            if phone[-1].isdigit() and int(phone[-1]) > 0:
                return tuple(pronunciation[i:])
        # Strategy 2: Fallback to last vowel (RELAXED LOGIC)
        for i in range(len(pronunciation) - 1, -1, -1):
             phone = pronunciation[i]
             if phone[-1].isdigit():
                 return tuple(pronunciation[i:])
        return tuple(pronunciation[-1:])

    def check_rhyme(self, w1, w2):
        r1 = self.get_rhyme_part(w1)
        r2 = self.get_rhyme_part(w2)
        if r1 and r2:
            if r1 == r2: return True
            # Relaxed: Allow last phone match if it's a vowel (weak rhyme/slant rhyme)
            if r1[-1] == r2[-1] and r1[-1][-1].isdigit(): return True
        return False

    def get_stress_pattern(self, line):
        words = re.findall(r"\w+", line)
        pattern = []
        for w in words:
            w_lower = w.lower()
            if w_lower in self.d:
                pron = self.d[w_lower][0]
                stresses = [int(p[-1]) for p in pron if p[-1].isdigit()]
                pattern.extend([1 if s > 0 else 0 for s in stresses])
            else:
                syls = self.count_syllables(w)
                pattern.extend([1] * syls)
        return pattern

    def is_iambic(self, pattern):
        if len(pattern) < 8 or len(pattern) > 12: return 0.0
        ideal = [0, 1] * 6
        ideal = ideal[:len(pattern)]
        matches = sum(1 for x, y in zip(pattern, ideal) if x == y)
        return matches / len(pattern)

# Initialize the wrapper with the global dictionary 'd' from Cell 1
detector_wrapper = SonnetDetectorWrapper(d)

class StructuralFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, detector):
        self.detector = detector

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        features = []
        for text in X:
            lines = text.split('\n')
            if not lines:
                features.append([0, 0, 0, 0])
                continue
            lines_14 = lines[:14]
            if len(lines_14) < 14:
                lines_14 += [""] * (14 - len(lines_14))

            # 1. Rhyme Density
            last_words = []
            for l in lines_14:
                words = re.findall(r"\w+", l)
                last_words.append(words[-1] if words else "")

            rhyme_pairs = 0
            for i in range(len(last_words)):
                for j in range(i + 1, len(last_words)):
                    if self.detector.check_rhyme(last_words[i], last_words[j]):
                        rhyme_pairs += 1
            rhyme_density = rhyme_pairs / 20.0

            # 2. Line Length Variance
            syllable_counts = []
            for l in lines_14:
                words = re.findall(r"\w+", l)
                count = sum(self.detector.count_syllables(w) for w in words)
                syllable_counts.append(count)
            length_var = np.var(syllable_counts) if syllable_counts else 100.0

            # 3. Iambic Score
            iambic_scores = []
            for l in lines_14:
                pattern = self.detector.get_stress_pattern(l)
                iambic_scores.append(self.detector.is_iambic(pattern))
            avg_iambic = np.mean(iambic_scores) if iambic_scores else 0.0

            features.append([rhyme_density, length_var, avg_iambic])

        return np.array(features)

print("Structural Feature Extractor Ready.")


# %% [markdown] id="Hqejl2dKo7Yz"
# ## Train Supervised and PU Models

# %% colab={"base_uri": "https://localhost:8080/"} id="lOELrvOgo-ig" outputId="7fb6dd95-0759-45c8-ff8d-79650eee0047"
# Cell 6: Training Hybrid Classifiers (Supervised & PU)

# 1. Helper to extract Canonical Sonnets (Positives)
def extract_canonical_sonnets(works):
    if "THE SONNETS" not in works: return []
    lines = works["THE SONNETS"]
    sonnets = []
    current = []
    for line in lines:
        if re.match(r'^[XVI]+\.?$', line) or re.match(r'^\d+\.?$', line):
            if len(current) >= 12: sonnets.append("\n".join(current))
            current = []
        else:
            current.append(line)
    if len(current) >= 12: sonnets.append("\n".join(current))
    return sonnets

all_positive_texts = extract_canonical_sonnets(works)
print(f"Extracted {len(all_positive_texts)} positive sonnets.")

# 2. Helper to make Hard Negatives (Real Blank Verse)
def make_hard_negatives(works, n_samples=200, block_len=14, random_state=42):
    rng = np.random.RandomState(random_state)
    all_play_lines = []
    exclude_titles = {"THE SONNETS", "A LOVER’S COMPLAINT", "THE PASSIONATE PILGRIM", "THE PHOENIX AND THE TURTLE", "THE RAPE OF LUCRECE", "VENUS AND ADONIS", "PLAY SONNETS"}

    for title, lines in works.items():
        if title in exclude_titles: continue
        all_play_lines.extend([l for l in lines if l.strip()])

    blocks = []
    for i in range(0, len(all_play_lines) - block_len, block_len // 2):
        block = all_play_lines[i : i + block_len]
        blocks.append("\n".join(block))

    if len(blocks) > n_samples:
        indices = rng.choice(len(blocks), size=n_samples, replace=False)
        return [blocks[i] for i in indices]
    return blocks

# 3. Train Supervised Model (Pos vs Hard Negatives)
print("\n--- Training Supervised Classifier (Pos vs Hard Negatives) ---")
hard_negatives = make_hard_negatives(works, n_samples=len(all_positive_texts), random_state=123)
y_sup = np.array([1] * len(all_positive_texts) + [0] * len(hard_negatives))
X_sup = all_positive_texts + hard_negatives

pipeline_sup = Pipeline([
    ('features', FeatureUnion([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=100)), # Low dim TF-IDF
        ('structural', StructuralFeatures(detector_wrapper))
    ])),
    ('scaler', StandardScaler(with_mean=False)),
    ('clf', LogisticRegression(max_iter=1000, class_weight='balanced', C=0.5))
])
pipeline_sup.fit(X_sup, y_sup)
print("Supervised Model Trained.")

# 4. Train PU Classifier (Pos vs Unlabeled Plays)
print("\n--- Training PU Classifier (Pos vs Unlabeled) ---")
# Build Unlabeled Pool
play_windows = []
play_meta = []
exclude_titles = {"THE SONNETS", "A LOVER’S COMPLAINT", "THE PASSIONATE PILGRIM", "THE PHOENIX AND THE TURTLE", "THE RAPE OF LUCRECE", "VENUS AND ADONIS"}

for title, lines in works.items():
    if title in exclude_titles: continue
    clean_lines = [l.strip() for l in lines if l.strip()]
    for i in range(0, len(clean_lines) - 14 + 1):
        block = clean_lines[i:i+14]
        play_windows.append("\n".join(block))
        play_meta.append((title, i, block[0]))

# Subsample unlabeled
rng = np.random.RandomState(123)
n_unlabeled = 3000
if len(play_windows) > n_unlabeled:
    indices = rng.choice(len(play_windows), size=n_unlabeled, replace=False)
    unlabeled_texts = [play_windows[i] for i in indices]
else:
    unlabeled_texts = play_windows

y_pu = np.array([1] * len(all_positive_texts) + [0] * len(unlabeled_texts))
X_pu = all_positive_texts + unlabeled_texts

pipeline_pu = Pipeline([
    ('features', FeatureUnion([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=100)),
        ('structural', StructuralFeatures(detector_wrapper))
    ])),
    ('scaler', StandardScaler(with_mean=False)),
    ('clf', LogisticRegression(max_iter=1000, class_weight='balanced', C=0.5))
])
pipeline_pu.fit(X_pu, y_pu)

# Estimate c
probs_pos = pipeline_pu.predict_proba(all_positive_texts)[:, 1]
c_pu = np.mean(probs_pos)
print(f"PU Model Trained. Estimated c: {c_pu:.3f}")


# %% [markdown] id="P10qf1pxpFEL"
# ## Compare & Find

# %% colab={"base_uri": "https://localhost:8080/", "height": 800} id="lquGn0ROpIDG" outputId="594c7af2-e5b1-4538-da75-012a18fd28f4"
# Cell 7: Full Analysis & Candidate Scanning

# 1. Verification on Test Cases
print("\n--- Verification ---")
friar_speech = """Hold thy desperate hand:
Art thou a man? thy form cries out thou art:
Thy tears are womanish; thy wild acts denote
The unreasonable fury of a beast:
Unseemly woman in a seeming man!
Or ill-beseeming beast in seeming both!
Thou hast amazed me: by my holy order,
I thought thy disposition better temper'd.
Hast thou slain Tybalt? wilt thou slay thyself?
And stay thy lady too that lives in thee,
By doing damned hate upon thyself?
Why rail'st thou on thy birth, the heaven, and earth?
Since birth, and heaven, and earth, all three do meet
In thee at once; which thou at once wouldst lose."""

prologue = """Two households, both alike in dignity,
In fair Verona, where we lay our scene,
From ancient grudge break to new mutiny,
Where civil blood makes civil hands unclean.
From forth the fatal loins of these two foes
A pair of star-cross'd lovers take their life;
Whose misadventured piteous overthrows
Do with their death bury their parents' strife.
The fearful passage of their death-mark'd love,
And the continuance of their parents' rage,
Which, but their children's end, nought could remove,
Is now the two hours' traffic of our stage;
The which if you with patient ears attend,
What here shall miss, our toil shall strive to mend."""

s_probs = pipeline_sup.predict_proba([friar_speech, prologue])[:, 1]
p_probs = pipeline_pu.predict_proba([friar_speech, prologue])[:, 1] / c_pu

print(f"Supervised: Friar={s_probs[0]:.4f}, Prologue={s_probs[1]:.4f}")
print(f"PU:         Friar={p_probs[0]:.4f}, Prologue={p_probs[1]:.4f}")

# 2. Scan Plays for Candidates
print("\n--- Scanning ALL Play Windows for Top Candidates ---")
# Use the full set of play windows
X_scan = play_windows
meta_scan = play_meta

print(f"Scanning {len(X_scan)} windows...")
sup_scores = pipeline_sup.predict_proba(X_scan)[:, 1]
pu_scores = pipeline_pu.predict_proba(X_scan)[:, 1] / c_pu

results = []
for i in range(len(X_scan)):
    results.append({
        "play": meta_scan[i][0],
        "line": meta_scan[i][1],
        "text": meta_scan[i][2],
        "sup": sup_scores[i],
        "pu": pu_scores[i],
        "full_text": X_scan[i]
    })

df = pd.DataFrame(results)

# 3. Export Top 2% Candidates (Text & CSV)
print("\n--- Exporting Top 2% Candidates ---")
try:
    from google.colab import files
    IS_COLAB = True
except ImportError:
    IS_COLAB = False
    import os

# Calculate Thresholds (Top 2% = 98th Percentile)
sup_thresh = np.percentile(df['sup'], 98)
pu_thresh = np.percentile(df['pu'], 98)
print(f"Supervised Top 2% Threshold: {sup_thresh:.4f}")
print(f"PU Top 2% Threshold: {pu_thresh:.4f}")

# Filter Dataframes
top_sup_df = df[df['sup'] >= sup_thresh].sort_values(['play', 'line'])
top_pu_df = df[df['pu'] >= pu_thresh].sort_values(['play', 'line'])

# Find Intersection (Indices present in both)
intersection_indices = list(set(top_sup_df.index) & set(top_pu_df.index))
intersection_df = df.loc[intersection_indices].sort_values(['play', 'line'])

print(f"Count Supervised: {len(top_sup_df)}")
print(f"Count PU: {len(top_pu_df)}")
print(f"Count Intersection: {len(intersection_df)}")

# --- A. Text File Export ---
def save_candidates(dataframe, filename, title):
    # Write to local file first
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"{title}\n")
        f.write("=" * len(title) + "\n\n")

        current_play = None
        for _, row in dataframe.iterrows():
            if row['play'] != current_play:
                f.write(f"\n--- {row['play']} ---\n")
                current_play = row['play']

            f.write(f"Line {row['line']} | Score: {row['sup']:.4f} (Sup) / {row['pu']:.4f} (PU)\n")
            # Write FULL TEXT
            f.write(f"{row['full_text']}\n")
            f.write("-" * 20 + "\n")

    print(f"Saved: {filename}")

    # Trigger download if in Colab
    if IS_COLAB:
        files.download(filename)
    else:
        try:
            downloads_path = os.path.expanduser("~/Downloads")
            full_path = os.path.join(downloads_path, filename)
            os.rename(filename, full_path)
            print(f"Moved to: {full_path}")
        except Exception:
            print(f"File remains in current directory: {filename}")

save_candidates(top_sup_df, "sonnet_candidates_supervised.txt", "Top 2% Candidates - Supervised Classifier")
save_candidates(top_pu_df, "sonnet_candidates_pu.txt", "Top 2% Candidates - PU Classifier")
save_candidates(intersection_df, "sonnet_candidates_intersection.txt", "Top 2% Candidates - INTERSECTION (Both Methods)")

# --- B. Structured CSV Export ---
print("\n--- Generating Structured CSV ---")
# Create a copy for export
export_df = df.copy()

# Add Line Ranges (assuming 14 lines for now, based on window size)
export_df['start_line'] = export_df['line']
export_df['end_line'] = export_df['line'] + 14

# Add Flags
export_df['is_top_sup'] = export_df['sup'] >= sup_thresh
export_df['is_top_pu'] = export_df['pu'] >= pu_thresh
export_df['is_intersection'] = export_df['is_top_sup'] & export_df['is_top_pu']

# Filter: Keep rows that are in Top 2% of EITHER model
sonnet_candidates_df = export_df[export_df['is_top_sup'] | export_df['is_top_pu']].copy()

# Select and Rename Columns
sonnet_candidates_df = sonnet_candidates_df[[
    'play', 'start_line', 'end_line',
    'sup', 'pu',
    'is_top_sup', 'is_top_pu', 'is_intersection'
]]
sonnet_candidates_df.columns = [
    'Play', 'Start_Line', 'End_Line',
    'Supervised_Score', 'PU_Score',
    'Is_Top_2_Pct_Sup', 'Is_Top_2_Pct_PU', 'Is_Intersection'
]

# Persist to global variable for later use
print(f"\nSUCCESS: Top 2% candidates stored in DataFrame 'sonnet_candidates_df' ({len(sonnet_candidates_df)} rows).")
print("Columns:", list(sonnet_candidates_df.columns))

csv_filename = "sonnet_candidates_structured.csv"
sonnet_candidates_df.to_csv(csv_filename, index=False)
print(f"Saved CSV: {csv_filename}")

if IS_COLAB:
    files.download(csv_filename)
else:
    try:
        downloads_path = os.path.expanduser("~/Downloads")
        full_path = os.path.join(downloads_path, csv_filename)
        os.rename(csv_filename, full_path)
        print(f"Moved CSV to: {full_path}")
    except Exception:
        pass

# 4. Graphical Output
print("\n--- Generating Visualizations ---")
plt.figure(figsize=(15, 6))

# A. Score Distribution
plt.subplot(1, 2, 1)
sns.histplot(sup_scores, bins=50, color='skyblue', label='Plays (Supervised)', kde=True, stat="density")
# Overlay Sonnet Scores
sonnet_scores = pipeline_sup.predict_proba(all_positive_texts)[:, 1]
sns.histplot(sonnet_scores, bins=20, color='orange', label='Canonical Sonnets', kde=True, stat="density", alpha=0.6)
plt.axvline(sup_thresh, color='red', linestyle='--', label='Top 2% Threshold')
plt.title('Score Distribution: Plays vs Sonnets (Supervised)')
plt.xlabel('Sonnet Probability')
plt.legend()

# B. Top Candidates Scatter
plt.subplot(1, 2, 2)
# Plot all points as small grey dots
plt.scatter(range(len(df)), df['sup'], alpha=0.1, s=1, color='lightgrey', label='All Windows')
# Highlight Intersection
plt.scatter(intersection_df.index, intersection_df['sup'], color='red', s=10, label='Intersection Candidates')
plt.title('Candidate Scores across Corpus')
plt.xlabel('Window Index')
plt.ylabel('Sonnet Score')
plt.legend()

plt.tight_layout()
plt.show()


# %% [markdown] id="Bg7O0p8Hn5CA"
# # Part 3 - Sonnet Hunter

# %% [markdown] id="V9rEbpzNfZ_g"
# ## Define Sonnet Finder

# %% id="iQ4vFglqKbki"
from functools import lru_cache

# %% id="5o3Sgi-N161X"
# Cell 8: Detecting Sonnets

'''
The final score (0.0 to 1.0) is a weighted sum of four distinct components.
A score of 1.0 represents a mathematically perfect Shakespearean sonnet in this schema.

1. Meter Score (25%)
Measures how well the rhythm matches iambic pentameter.

For each of the 14 lines:
We compare the line's stress pattern to the ideal 0-1-0-1-0-1...
Score:
Generate Notebook
Note: Handles "feminine endings" (11 syllables) by ignoring the extra unstressed beat.
Calculation: Average of all 14 line scores * 0.25

2. Syllable Count Score (15%)
Measures how close each line is to the required 10 syllables.

For each of the 14 lines:
Target: 10 Syllables (or 11 if feminine ending).
Penalty: -0.1 for every syllable over or under the target.
Score: 1.0 - (Distance * 0.1)
Calculation: Average of all 14 line scores * 0.15

3. Rhyme Scheme Score (35%)
Measures the rhyme patterns of the first 12 lines (three quatrains).

For each 4-line group (Quatrain):
ABAB (Cross Rhyme) = 1.0 (Perfect)
ABBA (Enclosed Rhyme) = 0.8
AABB (Couplets) = 0.6
AAAA (Monorhyme) = 0.5
Partial (Half the rhymes work) = 0.2
Calculation: Average of the 3 quatrain scores * 0.35

4. Couplet Score (25%)
Measures if the final two lines rhyme.

For the final 2 lines:
If they rhyme: Score = 1.0
If they don't: Score = 0.0
Calculation: (0 or 1) * 0.25


5. Final Calculation:
Total Score = (Meter * 0.25) + (Syllables * 0.15) + (Rhyme * 0.35) + (Couplet * 0.25)

'''


class SonnetDetector:
    def __init__(self):
        # Known Historical/Shakespearean Rhymes that fail phonetic checks
        self.historical_rhymes = {
            'love': ['move', 'prove'],
            'move': ['love'],
            'prove': ['love'],
            'gone': ['alone', 'moan'],
            'alone': ['gone'],
            'moan': ['gone'],
            'wind': ['mind', 'kind', 'find'],
            'mind': ['wind'],
            'kind': ['wind'],
            'find': ['wind'],
            'come': ['doom', 'tomb'],
            'doom': ['come'],
            'tomb': ['come'],
            'heaven': ['even'],
            'even': ['heaven'],
            'great': ['defeat', 'seat'],
            'defeat': ['great'],
            'seat': ['great']
        }

    def check_rhyme(self, word1, word2):
        w1_lower = word1.lower()
        w2_lower = word2.lower()

        # 0. Historical / Hardcoded Dictionary
        if w1_lower in self.historical_rhymes and w2_lower in self.historical_rhymes[w1_lower]:
            return True

        # 1. Exact/CMU check (Base)
        r1 = get_rhyme_part(word1)
        r2 = get_rhyme_part(word2)
        if r1 and r2 and r1 == r2:
            return True

        # 2. Slant Rhyme / Approximate Matching
        # Check last 2 characters (visual rhyme heuristic)
        if len(word1) >= 2 and len(word2) >= 2 and word1[-2:] == word2[-2:]:
            return True

        # Check shared vowels (Assonance)
        # Simplified: check if the last vowel character matches
        w1_vowels = [c for c in word1 if c in 'aeiouy']
        w2_vowels = [c for c in word2 if c in 'aeiouy']
        if w1_vowels and w2_vowels and w1_vowels[-1] == w2_vowels[-1]:
            return True

        return False

    def score_window(self, lines):
        # Flexible scoring for variable length passages (e.g. 10, 11, 14, 15, 18 lines)
        # Weights: Meter(15%), Syllable(15%), Rhyme(40%), Couplet(30%)

        if len(lines) < 8:
            return 0.0

        # 1. Meter & Syllable Score (15% + 15%)
        # Average over all lines provided
        meter_scores = []
        syllable_scores = []

        for line in lines:
            pat = get_stress_pattern(line)

            # Check for feminine ending (11 syllables, last unstressed)
            if len(pat) == 11 and pat[-1] == 0:
                meter_scores.append(is_iambic(pat[:-1]))
                syllable_scores.append(1.0) # Treat 11-fem as perfect length
            else:
                meter_scores.append(is_iambic(pat))
                # Syllable Count Score: 1.0 if 10, else penalty
                dist = abs(len(pat) - 10)
                syllable_scores.append(max(0, 1.0 - (dist * 0.1)))

        avg_meter = np.mean(meter_scores) if meter_scores else 0.0
        avg_syllable = np.mean(syllable_scores) if syllable_scores else 0.0

        # 2. Rhyme Scheme Score (40%)
        # We analyze the "body" of the poem (everything except the final couplet)
        last_words = []
        for line in lines:
            words = re.findall(r"\w+", line)
            if words:
                last_words.append(words[-1])
            else:
                last_words.append("")

        # Assume last 2 lines are the couplet, everything else is quatrains
        body_lines = last_words[:-2] if len(lines) >= 2 else []
        num_quatrains = len(body_lines) // 4
        remainder = len(body_lines) % 4

        quatrain_scores = []
        for i in range(0, num_quatrains * 4, 4):
            w = body_lines[i:i+4]
            # Check patterns
            is_cross = self.check_rhyme(w[0], w[2]) and self.check_rhyme(w[1], w[3])
            is_enclosed = self.check_rhyme(w[0], w[3]) and self.check_rhyme(w[1], w[2])
            is_couplets = self.check_rhyme(w[0], w[1]) and self.check_rhyme(w[2], w[3])
            is_mono = self.check_rhyme(w[0], w[1]) and self.check_rhyme(w[1], w[2]) and self.check_rhyme(w[2], w[3])

            if is_cross: val = 1.0
            elif is_enclosed: val = 0.8
            elif is_couplets: val = 0.6
            elif is_mono: val = 0.5
            elif self.check_rhyme(w[0], w[2]) or self.check_rhyme(w[1], w[3]): val = 0.2
            else: val = 0.0
            quatrain_scores.append(val)

        # Average of however many quatrains we found
        raw_rhyme_score = np.mean(quatrain_scores) if quatrain_scores else 0.0

        # Apply penalty for non-standard structure (Ideal = 3 quatrains, 0 leftover lines)
        # 3 quatrains + 0 remainder = 100%
        # Anything else (2 or 4 quatrains, or extra lines like 11/15) = 75%
        if num_quatrains == 3 and remainder == 0:
            rhyme_multiplier = 1.0
        else:
            rhyme_multiplier = 0.75

        rhyme_score = raw_rhyme_score * rhyme_multiplier

        # 3. Couplet Score (30%)
        # Always the last 2 lines
        if len(lines) >= 2:
            couplet_score = 1.0 if self.check_rhyme(last_words[-2], last_words[-1]) else 0.0
        else:
            couplet_score = 0.0

        # Weighted Sum
        # Meter: 0.15, Syllable: 0.15, Rhyme: 0.40, Couplet: 0.30
        total_score = (avg_meter * 0.15) + (avg_syllable * 0.15) + (rhyme_score * 0.40) + (couplet_score * 0.30)
        return total_score

detector = SonnetDetector()

# %% [markdown] id="Fu0mylRVfoae"
# ## Analyze Sonnets Specifically

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} id="jkKCvdp9w0Tk" outputId="7031bcf4-f525-4296-973c-e45922df9ea1"
# Cell 9: Analysis and Visualization

# --- 0. Configuration & Setup ---
WORK_TO_ANALYZE = "THE SONNETS"

# Abbreviation Map for Play Names
PLAY_ABBREVIATIONS = {
    "ALL’S WELL THAT ENDS WELL": "AllsWell",
    "THE TRAGEDY OF ANTONY AND CLEOPATRA": "Antony",
    "AS YOU LIKE IT": "AsYouLike",
    "THE COMEDY OF ERRORS": "Errors",
    "THE TRAGEDY OF CORIOLANUS": "Coriolanus",
    "CYMBELINE": "Cymbeline",
    "THE TRAGEDY OF HAMLET, PRINCE OF DENMARK": "Hamlet",
    "THE FIRST PART OF KING HENRY THE FOURTH": "Henry4-1",
    "THE SECOND PART OF KING HENRY THE FOURTH": "Henry4-2",
    "THE LIFE OF KING HENRY THE FIFTH": "Henry5",
    "THE FIRST PART OF HENRY THE SIXTH": "Henry6-1",
    "THE SECOND PART OF KING HENRY THE SIXTH": "Henry6-2",
    "THE THIRD PART OF KING HENRY THE SIXTH": "Henry6-3",
    "KING HENRY THE EIGHTH": "Henry8",
    "THE LIFE AND DEATH OF KING JOHN": "John",
    "THE TRAGEDY OF JULIUS CAESAR": "Caesar",
    "THE TRAGEDY OF KING LEAR": "Lear",
    "LOVE’S LABOUR’S LOST": "LLL",
    "THE TRAGEDY OF MACBETH": "Macbeth",
    "MEASURE FOR MEASURE": "Measure",
    "THE MERCHANT OF VENICE": "Merchant",
    "THE MERRY WIVES OF WINDSOR": "Wives",
    "A MIDSUMMER NIGHT’S DREAM": "Midsummer",
    "MUCH ADO ABOUT NOTHING": "MuchAdo",
    "THE TRAGEDY OF OTHELLO, THE MOOR OF VENICE": "Othello",
    "PERICLES, PRINCE OF TYRE": "Pericles",
    "KING RICHARD THE SECOND": "Richard2",
    "KING RICHARD THE THIRD": "Richard3",
    "THE TRAGEDY OF ROMEO AND JULIET": "Romeo",
    "THE TAMING OF THE SHREW": "Shrew",
    "THE TEMPEST": "Tempest",
    "THE LIFE OF TIMON OF ATHENS": "Timon",
    "THE TRAGEDY OF TITUS ANDRONICUS": "Titus",
    "TROILUS AND CRESSIDA": "Troilus",
    "TWELFTH NIGHT; OR, WHAT YOU WILL": "Twelfth",
    "THE TWO GENTLEMEN OF VERONA": "TwoGent",
    "THE TWO NOBLE KINSMEN": "Kinsmen",
    "THE WINTER’S TALE": "Winter",
    "A LOVER’S COMPLAINT": "Lover",
    "THE PASSIONATE PILGRIM": "Pilgrim",
    "THE PHOENIX AND THE TURTLE": "Phoenix",
    "THE RAPE OF LUCRECE": "Lucrece",
    "VENUS AND ADONIS": "Venus"
}

# --- 1. Date Categorization ---
def get_sonnet_date(num):
    if 1 <= num <= 60: return 1596
    elif 61 <= num <= 77: return 1594.5
    elif 78 <= num <= 86: return 1599
    elif 87 <= num <= 103: return 1594.5
    elif 104 <= num <= 126: return 1602
    elif 127 <= num <= 144: return 1593
    elif num == 145: return 1582
    elif 146 <= num <= 154: return 1593
    return None




def analyze_work(title, lines):
    print(f"Analyzing {title}...")
    scores = []
    # We now scan with multiple window sizes to find the best fit
    window_sizes = [10, 11, 14, 15, 18]
    max_window = max(window_sizes)

    clean_data = [] # List of tuples: (text, speaker)
    boundaries = [] # (index, label, type)

    current_clean_idx = 0
    current_speaker = "UNKNOWN"

    for line in lines:
        s_line = line.strip()
        if not s_line:
            continue

        # Check boundaries
        if s_line.upper().startswith("ACT "):
            label = s_line.split('.')[0]
            boundaries.append((current_clean_idx, label, 'act'))
            continue
        if s_line.upper().startswith("SCENE "):
            label = s_line.split('.')[0]
            boundaries.append((current_clean_idx, label, 'scene'))
            continue

        # Check Speaker
        if (
            s_line.isupper() and s_line.endswith(".") and len(s_line) < 20 and
            "SCENE" not in s_line and "ACT" not in s_line
        ):
            current_speaker = s_line.strip(".")
            continue

        if len(s_line) > 10:
            clean_data.append((s_line, current_speaker))
            current_clean_idx += 1

    # Extract just text for scoring
    clean_lines_text = [x[0] for x in clean_data]

    # Scan each position
    for i in range(len(clean_lines_text)):
        best_score_for_pos = 0.0

        for w_size in window_sizes:
            if i + w_size <= len(clean_lines_text):
                window = clean_lines_text[i : i + w_size]
                # Use updated detector.score_window (now flexible)
                score = detector.score_window(window)
                if score > best_score_for_pos:
                    best_score_for_pos = score

        scores.append(best_score_for_pos)

    return scores, clean_data, boundaries

# Analyze
lines = works[WORK_TO_ANALYZE]
scores, clean_data, boundaries = analyze_work(WORK_TO_ANALYZE, lines)

# Plot detection signal
plt.figure(figsize=(14, 8))
plt.plot(scores, label=WORK_TO_ANALYZE, linewidth=1)
plt.title(f"Sonnet Detection Signal in {WORK_TO_ANALYZE}")
plt.xlabel("Line Number (approx)")
plt.ylabel("Sonnet Score")

# 1. Add Boundaries
for idx, label, b_type in boundaries:
    if b_type == 'act':
        plt.axvline(x=idx, color='black', linewidth=0.3)
        plt.text(idx, 0.05, label, rotation=90, fontsize=6, verticalalignment='bottom')
    elif b_type == 'scene':
        plt.axvline(x=idx, color='black', linewidth=0.05)
        plt.text(idx, 0.15, label, rotation=90, fontsize=6, verticalalignment='bottom')

# 2. Top 10 Markers via NMS
scores_array = np.array(scores)
sorted_indices = scores_array.argsort()[::-1]

nms_indices = []
taken_mask = np.zeros(len(scores), dtype=bool)
window_size = 14

for idx in sorted_indices:
    if len(nms_indices) >= 10:
        break
    if taken_mask[idx]:
        continue

    nms_indices.append(idx)

    # Mark neighbors as taken (use max window size 18 to be safe)
    start = max(0, idx - 18 + 1)
    end = min(len(scores), idx + 18)
    taken_mask[start:end] = True

top_indices = nms_indices

# Plot markers for these top NMS hits
for idx in top_indices:
    score = scores[idx]
    plt.plot(idx, score, 'k.', markersize=5) # Black dot
    plt.text(idx, score, f"{idx}", fontsize=6, rotation=45, ha='left', va='bottom')

plt.legend()
plt.ylim(0, 1.0) # Make room for labels
plt.show()

# Top 10 Segments Text Output (NMS Applied)
print(f"\nTop 10 Scoring Segments in {WORK_TO_ANALYZE} (Distinct Regions):\n" + "="*40)
for i, idx in enumerate(top_indices):
    score = scores[idx]
    print(f"\nRank {i+1} | Score: {score:.4f} | Start Line: {idx}")
    print("-" * 20)

    last_speaker = None
    for j in range(idx, idx+14):
        line_text, speaker = clean_data[j]

        prefix = ""
        if speaker != last_speaker:
            prefix = f"[{speaker}] "
            last_speaker = speaker

        print(f"{prefix}{line_text}")

# --- Special Analysis: Scatter Plot with Stems (Sonnets + Play Sonnets) ---
# REMOVED IF CONDITION: Always run this
print("\nGenerating Sonnet Scatter Plot (Including Play Sonnets)...")

# Parse individual sonnets (Quick re-parse for this visualization)
sonnets_map = {}
current_sonnet = []
current_num = 0
roman_re = re.compile(r'^[XVI]+(\.)?$')

for line in lines:
    s_line = line.strip()
    if not s_line:
        continue

    # Check for header (Roman numeral or digit)
    is_header = False
    if s_line.isdigit():
        is_header = True
    elif roman_re.match(s_line):
        is_header = True

    if is_header:
        if current_sonnet and len(current_sonnet) >= 10:
            sonnets_map[current_num] = current_sonnet
        current_sonnet = []
        try:
            current_num = int(s_line.strip('.'))
        except:
            current_num += 1
    else:
        current_sonnet.append(s_line)
if current_sonnet:
    sonnets_map[current_num] = current_sonnet

# Score each sonnet
sonnet_scores = []
sonnet_labels = []
sonnet_colors = []

# 1. Canonical Sonnets
for num in sorted(sonnets_map.keys()):
    s_lines = sonnets_map[num]
    # Allow shorter sonnets (e.g. Sonnet 126 is 12 lines)
    if len(s_lines) < 10:
        continue

    # Use updated detector.score_window
    score = detector.score_window(s_lines)

    sonnet_scores.append(score)
    sonnet_labels.append(str(num))
    sonnet_colors.append(score)

# 2. Play Sonnets (Manually Defined)
# Note: works["PLAY SONNETS"] is built in Cell 3. We use it here.
if "PLAY SONNETS" not in works:
    print("WARNING: 'PLAY SONNETS' not found in works. Please run Cell 3.")
    works["PLAY SONNETS"] = []

print("Scoring Manual Play Sonnets...")
play_sonnet_lines = works["PLAY SONNETS"]
current_play_sonnet = []
current_label = ""

for line in play_sonnet_lines:
    if line.startswith("=== PLAY SONNET"):
        # Process previous
        if current_play_sonnet:
            # Use updated detector.score_window
            score = detector.score_window(current_play_sonnet)
            sonnet_scores.append(score)

            # Create short label
            parts = current_label.split(',')
            if len(parts) > 0:
                play_full = parts[0].strip()
                abbr = PLAY_ABBREVIATIONS.get(play_full, play_full[:8])
                # Handle "3.1.140–9" -> "3.1"
                rest = parts[1].strip() if len(parts) > 1 else ""
                # Extract just the Act.Scene part (e.g., "3.1")
                # Sometimes it's "scene 2.322–3" -> "scene 2"
                # Let's just take the first chunk before any dash or extra dot
                match_nums = re.search(r"(\d+\.\d+)", rest)
                if match_nums:
                    nums = match_nums.group(1)
                    short_label = f"{abbr} {nums}"
                else:
                    short_label = f"{abbr} {rest.split('–')[0]}"
                sonnet_labels.append(short_label)
            else:
                sonnet_labels.append(current_label[:10])

            sonnet_colors.append(score)

        # Start new
        current_play_sonnet = []
        # Extract label from header: "=== PLAY SONNET 1: Label (N lines) ==="
        match = re.search(r": (.*) \(", line)
        if match:
            current_label = match.group(1)
        else:
            current_label = "Unknown"
    elif line.strip():
        current_play_sonnet.append(line)

# Process last one
if current_play_sonnet:
    score = detector.score_window(current_play_sonnet)
    sonnet_scores.append(score)
    parts = current_label.split(',')
    if len(parts) > 0:
        play_full = parts[0].strip()
        abbr = PLAY_ABBREVIATIONS.get(play_full, play_full[:8])
        rest = parts[1].strip() if len(parts) > 1 else ""
        match_nums = re.search(r"(\d+\.\d+)", rest)
        if match_nums:
            nums = match_nums.group(1)
            short_label = f"{abbr} {nums}"
        else:
            short_label = f"{abbr} {rest.split('–')[0]}"
        sonnet_labels.append(short_label)
    else:
        sonnet_labels.append(current_label[:10])
    sonnet_colors.append(score)

# Plot
plt.figure(figsize=(18, 6)) # Wider for more labels
x_indices = range(len(sonnet_scores))

sc = plt.scatter(x_indices, sonnet_scores, c=sonnet_colors, cmap='jet', s=15, zorder=3)
plt.vlines(x_indices, 0, sonnet_scores, colors='gray', linestyles=':', linewidth=0.5, alpha=0.5, zorder=1)

plt.colorbar(sc, label="Sonnet Score")
plt.xlabel("Sonnet / Play Segment")
plt.ylabel("Sonnet Score")
plt.title("Shakespeare's Sonnets & Top Play Candidates: Score Comparison")

plt.xticks(x_indices, sonnet_labels, rotation=45, fontsize=6, ha='right')

plt.ylim(0, 1.1)
plt.xlim(-1, len(sonnet_scores))
plt.grid(axis='y', linestyle='--', alpha=0.3)
plt.tight_layout()
plt.show()

# --- Special Analysis: Detailed Rhyme Scheme Analysis & Examples ---
# REMOVED IF CONDITION: Always run this
print("\nGenerating Detailed Sonnet Rhyme Scheme Analysis...")

# 1. Helper to classify rhyme scheme (STRICT MODE) & Capture Examples
def classify_rhyme_scheme_with_example(lines):
    if len(lines) < 12:
        return "Unknown", None

    last_words = []
    for line in lines:
        words = re.findall(r"\w+", line)
        if words:
            last_words.append(words[-1])
        else:
            last_words.append("")

    quatrain_types = []
    deviant_examples = {"Enclosed": None, "Couplets": None, "Irregular": None}

    for i in range(0, 12, 4):
        w = last_words[i:i+4]
        current_quatrain = lines[i:i+4]

        # Check patterns
        is_cross = detector.check_rhyme(w[0], w[2]) and detector.check_rhyme(w[1], w[3])
        is_enclosed = detector.check_rhyme(w[0], w[3]) and detector.check_rhyme(w[1], w[2])
        is_couplets = detector.check_rhyme(w[0], w[1]) and detector.check_rhyme(w[2], w[3])

        q_type = "Irregular"
        if is_cross:
            q_type = "Cross"
        elif is_enclosed:
            q_type = "Enclosed"
        elif is_couplets:
            q_type = "Couplets"

        quatrain_types.append(q_type)

        if q_type != "Cross" and deviant_examples.get(q_type) is None:
            deviant_examples[q_type] = (current_quatrain, q_type)

    final_label = "Irregular / Other"
    selected_example = None

    if all(t == "Cross" for t in quatrain_types):
        final_label = "Pure Cross (ABAB)"
    elif "Enclosed" in quatrain_types:
        final_label = "Contains Enclosed (ABBA)"
        selected_example = deviant_examples["Enclosed"]
    elif "Couplets" in quatrain_types:
        final_label = "Contains Couplets (AABB)"
        selected_example = deviant_examples["Couplets"]
    else:
        final_label = "Irregular / Other"
        selected_example = deviant_examples["Irregular"]

    return final_label, selected_example

# 3. Score, Classify, and Collect Examples
data = [] # (num, score, scheme)
examples = {
    "Contains Enclosed (ABBA)": None,
    "Contains Couplets (AABB)": None,
    "Irregular / Other": None
}

for num, s_lines in sonnets_map.items():
    if len(s_lines) < 14:
        continue
    score = detector.score_window(s_lines[:14])
    scheme, deviant_info = classify_rhyme_scheme_with_example(s_lines[:14])
    data.append((num, score, scheme))

    if scheme in examples and examples[scheme] is None and deviant_info:
        examples[scheme] = (num, deviant_info[0], deviant_info[1])

data.sort(key=lambda x: x[1])

sorted_nums = [str(x[0]) for x in data]
sorted_scores = [x[1] for x in data]
sorted_schemes = [x[2] for x in data]

color_map = {
    "Pure Cross (ABAB)": "tab:blue",
    "Contains Enclosed (ABBA)": "tab:red",
    "Contains Couplets (AABB)": "tab:green",
    "Irregular / Other": "gray"
}

plt.figure(figsize=(14, 6))

for label in color_map:
    indices = [i for i, s in enumerate(sorted_schemes) if s == label]
    if indices:
        plt.scatter(
            [sorted_nums[i] for i in indices],
            [sorted_scores[i] for i in indices],
            c=color_map[label],
            label=label,
            s=30,
            edgecolor='white',
            linewidth=0.5,
            zorder=3
        )

plt.vlines(sorted_nums, 0, sorted_scores, colors='gray', linestyles=':', linewidth=0.5, alpha=0.5, zorder=1)

plt.xlabel("Sonnet Number (Sorted by Score)")
plt.ylabel("Sonnet Score")
plt.title("Shakespeare's Sonnets: Ranked by Score & Rhyme Deviations")

plt.xticks(rotation=90, fontsize=6)
plt.ylim(0, 1.1)
plt.xlim(-1, len(data))
plt.grid(axis='y', linestyle='--', alpha=0.3)
plt.legend(title="Rhyme Structure")
plt.tight_layout()
plt.show()

print("\n" + "-"*40)
print("EXAMPLES OF DEVIANT RHYME SCHEMES")
print("-"*40)

for category, info in examples.items():
    if info:
        num, lines_ex, q_type = info
        print(f"\n> {category}")
        print(f"  (From Sonnet {num}, Quatrain Type: {q_type})")
        for line in lines_ex:
            print(f"    {line}")
    else:
        print(f"\n> {category}")
        print("  (No examples found)")

# --- Refined Vocabulary Analysis ---
def analyze_vocabulary_refined(sonnets_map):
    print("\n" + "="*60)
    print("VOCABULARY ANALYSIS: Counts & Histogram")
    print("="*60)

    all_text = ""
    for lines in sonnets_map.values():
        all_text += " ".join(lines) + " "

    words = re.findall(r"\w+", all_text.lower())
    counts = Counter(words)

    words_1x = sorted([w for w, c in counts.items() if c == 1])
    words_2x = sorted([w for w, c in counts.items() if c == 2])
    words_3x = sorted([w for w, c in counts.items() if c == 3])

    print(f"\n--- Words appearing EXACTLY 1 time ({len(words_1x)} words) ---")
    print(", ".join(words_1x) if words_1x else "None")

    print(f"\n--- Words appearing EXACTLY 2 times ({len(words_2x)} words) ---")
    print(", ".join(words_2x) if words_2x else "None")

    print(f"\n--- Words appearing EXACTLY 3 times ({len(words_3x)} words) ---")
    print(", ".join(words_3x) if words_3x else "None")

    print("\nGenerating Vocabulary Histogram...")

    bins = list(range(1, 21))
    bin_counts = [0] * 20

    for w, c in counts.items():
        if c >= 20:
            bin_counts[19] += 1
        else:
            bin_counts[c-1] += 1

    plt.figure(figsize=(12, 6))
    bars = plt.bar(bins, bin_counts, color='skyblue', edgecolor='black')

    plt.xlabel("Number of Times Word is Used")
    plt.ylabel("Count of Unique Words")
    plt.title("Vocabulary Usage Distribution in Sonnets")
    plt.xticks(bins, [str(b) if b < 20 else "20+" for b in bins])
    plt.grid(axis='y', linestyle='--', alpha=0.3)

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height}',
                 ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.show()


# --- Rare-word feature construction: words appearing in 2–12 sonnets ---
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial import ConvexHull
from matplotlib.lines import Line2D

def build_rare_word_matrix(sonnets_map, min_df=2, max_df=12):
    sonnet_nums = sorted(sonnets_map.keys())

    sonnet_word_sets = {}
    df_counts = Counter()

    for num in sonnet_nums:
        text = " ".join(sonnets_map[num])
        words = re.findall(r"\w+", text.lower())
        wset = set(words)
        sonnet_word_sets[num] = wset
        for w in wset:
            df_counts[w] += 1

    rare_words = sorted(
        [w for w, c in df_counts.items() if min_df <= c <= max_df]
    )

    X = np.zeros((len(sonnet_nums), len(rare_words)), dtype=np.float32)
    for i, num in enumerate(sonnet_nums):
        wset = sonnet_word_sets[num]
        for j, w in enumerate(rare_words):
            if w in wset:
                X[i, j] = 1.0

    return sonnet_nums, rare_words, X


# --- Helper: compute word lists associated with each cluster ---
def get_cluster_word_lists(rare_words, X, cluster_labels):
    cluster_labels = np.asarray(cluster_labels)
    unique_clusters = sorted(set(cluster_labels))
    cluster_word_lists = {}

    for cid in unique_clusters:
        indices = np.where(cluster_labels == cid)[0]
        if len(indices) == 0:
            cluster_word_lists[cid] = []
            continue
        used_mask = (X[indices, :].sum(axis=0) > 0)
        words_in_cluster = [w for w, used in zip(rare_words, used_mask) if used]
        cluster_word_lists[cid] = sorted(words_in_cluster)

    return cluster_word_lists


# --- Helper: nicely print cluster word lists in columns (LIMITED TO 5 ROWS) ---
def print_cluster_word_lists(cluster_word_lists, method_name):
    print("\n" + "-"*60)
    print(f"WORD LISTS PER CLUSTER ({method_name})")
    print("-"*60)

    cluster_ids = sorted(cluster_word_lists.keys())
    lists = [cluster_word_lists[cid] for cid in cluster_ids]

    # LIMIT TO 5 ROWS
    max_len = min(5, max(len(lst) for lst in lists) if lists else 0)

    header = " | ".join([f"Cluster {cid}".ljust(20) for cid in cluster_ids])
    print(header)
    print("-" * len(header))

    for i in range(max_len):
        row_elems = []
        for lst in lists:
            if i < len(lst):
                row_elems.append(lst[i].ljust(20))
            else:
                row_elems.append("".ljust(20))
        print(" | ".join(row_elems))

    if any(len(lst) > 5 for lst in lists):
        print("... (truncated to 5 rows) ...")


# --- Helper: filtered cluster word lists (remove words in >=3 clusters) ---
def print_filtered_cluster_word_lists(cluster_word_lists, method_name, min_cluster_count=3):
    print("\n" + "-"*60)
    print(f"FILTERED WORD LISTS PER CLUSTER ({method_name})")
    print(f"(words appearing in >= {min_cluster_count} clusters removed)")
    print("-"*60)

    cluster_ids = sorted(cluster_word_lists.keys())
    word_cluster_counts = Counter()
    for cid in cluster_ids:
        for w in set(cluster_word_lists[cid]):
            word_cluster_counts[w] += 1

    filtered_lists = {}
    for cid in cluster_ids:
        filtered_lists[cid] = [
            w for w in cluster_word_lists[cid]
            if word_cluster_counts[w] < min_cluster_count
        ]

    lists = [filtered_lists[cid] for cid in cluster_ids]

    # LIMIT TO 5 ROWS
    max_len = min(5, max(len(lst) for lst in lists) if lists else 0)

    header = " | ".join([f"Cluster {cid}".ljust(20) for cid in cluster_ids])
    print(header)
    print("-" * len(header))

    for i in range(max_len):
        row_elems = []
        for lst in lists:
            if i < len(lst):
                row_elems.append(lst[i].ljust(20))
            else:
                row_elems.append("".ljust(20))
        print(" | ".join(row_elems))

    if any(len(lst) > 5 for lst in lists):
        print("... (truncated to 5 rows) ...")


# --- Helper: side-panel with up to 50 words per cluster ---
def add_cluster_word_sidepanel(ax_text, cluster_word_lists, max_words_per_cluster=50, fontsize=7):
    ax_text.axis('off')

    cluster_ids = sorted(cluster_word_lists.keys())
    n_clusters = len(cluster_ids)
    if n_clusters == 0:
        return

    x0 = 0.02
    x1 = 0.98
    dx = (x1 - x0) / max(n_clusters, 1)
    y_top = 0.98

    for i, cid in enumerate(cluster_ids):
        words = cluster_word_lists[cid][:max_words_per_cluster]
        text = f"Cluster {cid}\n" + "\n".join(words)
        x = x0 + i * dx
        ax_text.text(
            x, y_top,
            text,
            fontsize=fontsize,
            va='top',
            ha='left',
            family='monospace',
            transform=ax_text.transAxes
        )


# --- Helper: dendrogram for hierarchical clustering ---
def plot_dendrogram(X, sonnet_nums, rare_words, method_name):
    """
    Ward-linkage dendrogram with:
    - Leaves colored by date (viridis).
    - Short colored 'terminal branches' at the base for each sonnet.
    - 1–2 'link words' under each sonnet, indicating words shared with its nearest neighbor
      in rare-word Jaccard space.
    """
    # Build linkage
    linked = linkage(X, method='ward')

    fig, ax = plt.subplots(figsize=(14, 6))
    dd = dendrogram(
        linked,
        labels=[str(n) for n in sonnet_nums],
        leaf_rotation=90,
        leaf_font_size=6,
        ax=ax
    )
    ax.set_title(f"Dendrogram of Sonnets ({method_name})")
    ax.set_xlabel("Sonnet Number")
    ax.set_ylabel("Distance (Ward linkage)")

    # == Viridis coloring by date ==
    dates = [get_sonnet_date(n) for n in sonnet_nums]
    df_s = pd.DataFrame({"Num": sonnet_nums, "Date": dates})
    df_s = df_s.dropna().reset_index(drop=True)
    df_s['DateLabel'] = df_s['Date'].astype(str)
    unique_labels = sorted(df_s['DateLabel'].unique(), key=lambda x: float(x))
    palette_colors = sns.color_palette("viridis", n_colors=len(unique_labels))
    palette_dict = {label: color for label, color in zip(unique_labels, palette_colors)}
    date_for_num = dict(zip(df_s['Num'], df_s['DateLabel']))

    # Build rare-word sets per sonnet from X and rare_words
    num_to_idx = {n: i for i, n in enumerate(sonnet_nums)}
    sonnet_rare_sets = {}
    for n in sonnet_nums:
        i = num_to_idx[n]
        mask = X[i, :] > 0
        sonnet_rare_sets[n] = set(np.array(rare_words)[mask])

    # Helper for Jaccard similarity
    def jacc(s1, s2):
        if not s1 and not s2:
            return 1.0
        inter = len(s1 & s2)
        union = len(s1 | s2)
        return inter / union if union > 0 else 0.0

    # For each sonnet, find nearest neighbor (highest Jaccard over rare-word sets)
    nearest_info = {}
    nums = sonnet_nums
    for i, n in enumerate(nums):
        s1 = sonnet_rare_sets[n]
        best_n = None
        best_score = -1.0
        for m in nums:
            if m == n:
                continue
            s2 = sonnet_rare_sets[m]
            score = jacc(s1, s2)
            if score > best_score:
                best_score = score
                best_n = m
        if best_n is None:
            nearest_info[n] = (None, [])
        else:
            shared = sorted(list(sonnet_rare_sets[n] & sonnet_rare_sets[best_n]))
            nearest_info[n] = (best_n, shared)

    # Color tick labels and draw colored terminal branches + link-words
    ymax = ax.get_ylim()[1]
    for text in ax.get_xticklabels():
        lab = text.get_text()
        if not lab.strip():
            continue
        try:
            num = int(lab)
        except ValueError:
            continue

        dlabel = date_for_num.get(num, None)
        if dlabel is not None:
            color = palette_dict[dlabel]
            # Color the tick label
            text.set_color(color)

            # x-position in data coords
            x = text.get_position()[0]
            # Short colored vertical line at base as terminal branch
            ax.vlines(x, 0, 0.05 * ymax, color=color, linewidth=1.0)

            # Link words under label (shared with nearest neighbor)
            _, shared_words = nearest_info.get(num, (None, []))
            if shared_words:
                phrase = "/".join(shared_words[:2])
                ax.text(
                    x,
                    -0.08,  # a bit below the tick labels
                    phrase,
                    fontsize=5,
                    rotation=90,
                    ha='center',
                    va='top',
                    transform=ax.get_xaxis_transform()
                )

    fig.text(
        0.5, 0.01,
        (
            "Ward dendrogram: each leaf is a sonnet. Higher merges indicate joining more dissimilar clusters "
            "in the rare-word feature space, minimizing within-cluster variance at each step. "
            "Leaf colors (and short base branches) reflect hypothesized composition date (viridis scale). "
            "The tiny labels under each sonnet show 1–2 rare words it shares with its nearest neighbor, "
            "giving a lexical hint about why those sonnets cluster together."
        ),
        ha='center', va='bottom', fontsize=8, wrap=True
    )

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.show()


def plot_clusters_with_dates(X, sonnet_nums, cluster_labels, method_name, cluster_word_lists):
    """
    PCA(2D) visualization:
    - Color = date (viridis)
    - Marker shape = cluster
    - Each point labeled with sonnet number
    - Faint convex hull per cluster (alpha=0.15, black outline, lw=0.15)
    - Side panel (right subplot) listing up to 50 words per cluster
    """
    dates = [get_sonnet_date(num) for num in sonnet_nums]
    df_s = pd.DataFrame({"Num": sonnet_nums, "Date": dates})
    df_s = df_s.dropna().reset_index(drop=True)

    valid_indices = df_s.index.to_list()
    X_plot = X[valid_indices, :]
    cluster_labels = np.array(cluster_labels)[valid_indices]

    df_s['DateLabel'] = df_s['Date'].astype(str)
    unique_labels = sorted(df_s['DateLabel'].unique(), key=lambda x: float(x))

    palette_colors = sns.color_palette("viridis", n_colors=len(unique_labels))
    palette_dict = {label: color for label, color in zip(unique_labels, palette_colors)}

    pca_2d = PCA(n_components=2, random_state=42)
    X_2d = pca_2d.fit_transform(X_plot)

    markers = ['o', 's', '^', 'D', 'P', 'X', '*', 'v']

    fig, (ax, ax_text) = plt.subplots(
        1, 2,
        gridspec_kw={'width_ratios': [3, 2]},
        figsize=(18, 8)
    )

    # Convex hull per cluster
    unique_clusters = sorted(set(cluster_labels))
    for cid in unique_clusters:
        idxs = np.where(cluster_labels == cid)[0]
        if len(idxs) < 3:
            continue
        pts = X_2d[idxs, :]
        try:
            hull = ConvexHull(pts)
            hull_pts = pts[hull.vertices]
            ax.fill(
                hull_pts[:, 0],
                hull_pts[:, 1],
                facecolor='lightgray',
                edgecolor='black',
                alpha=0.15,
                linewidth=0.15
            )
        except Exception:
            pass

    # Scatter with labels
    for idx, row in df_s.iterrows():
        x, y = X_2d[idx, 0], X_2d[idx, 1]
        date_label = row['DateLabel']
        color = palette_dict[date_label]
        c_id = int(cluster_labels[idx])
        marker = markers[c_id % len(markers)]

        ax.scatter(
            x, y,
            c=[color],
            marker=marker,
            edgecolors='black',
            s=60,
            alpha=0.85
        )
        ax.text(
            x + 0.02,
            y + 0.02,
            str(row['Num']),
            fontsize=6
        )

    # Legends inside left axis
    date_handles = [
        Line2D([0], [0],
               marker='o', color='w',
               label=lab,
               markerfacecolor=palette_dict[lab],
               markersize=8)
        for lab in unique_labels
    ]

    cluster_ids = sorted(set(cluster_labels))
    cluster_handles = [
        Line2D([0], [0],
               marker=markers[c % len(markers)],
               color='k',
               linestyle='None',
               label=f"Cluster {c}")
        for c in cluster_ids
    ]

    leg1 = ax.legend(
        handles=date_handles,
        title="Hypothesized Date",
        loc='upper left'
    )
    ax.add_artist(leg1)
    ax.legend(
        handles=cluster_handles,
        title="Cluster",
        loc='lower left'
    )

    ax.set_title(f"Sonnets clustered by rare-word usage ({method_name})")
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")

    add_cluster_word_sidepanel(ax_text, cluster_word_lists, max_words_per_cluster=50, fontsize=7)

    plt.tight_layout()
    plt.show()


# --- Helper: compare clusterings across methods by labels ---
def compare_clusterings(sonnet_nums, labels_dict):
    methods = list(labels_dict.keys())
    print("\n" + "="*60)
    print("CLUSTER AGREEMENT ACROSS METHODS (Adjusted Rand Index)")
    print("="*60)

    for i in range(len(methods)):
        for j in range(i+1, len(methods)):
            m1, m2 = methods[i], methods[j]
            labels1 = np.asarray(labels_dict[m1])
            labels2 = np.asarray(labels_dict[m2])
            ari = adjusted_rand_score(labels1, labels2)
            print(f"{m1} vs {m2}: ARI = {ari:.3f}")

    if not methods:
        return

    ref = methods[0]
    for other in methods[1:]:
        labels_ref = labels_dict[ref]
        labels_other = labels_dict[other]
        cm = pd.crosstab(
            pd.Series(labels_ref, name=f"{ref} cluster"),
            pd.Series(labels_other, name=f"{other} cluster")
        )
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"Cluster overlap (by sonnets): {ref} vs {other}")
        plt.tight_layout()
        plt.show()


# --- Helper: compare clusters across methods by WORD SETS (Jaccard) ---
def compare_cluster_word_structures(word_lists_dict):
    """
    For each pair of methods:
      - Compute Jaccard similarity between every pair of clusters
        (using the sets of rare-words defining each cluster).
      - Report best matches and flag identical word sets.
      - Plot a heatmap of Jaccard similarities.
    """
    methods = list(word_lists_dict.keys())
    print("\n" + "="*60)
    print("CLUSTER SIMILARITY ACROSS METHODS (Jaccard of WORD SETS)")
    print("="*60)

    def jacc(s1, s2):
        if not s1 and not s2:
            return 1.0
        inter = len(s1 & s2)
        union = len(s1 | s2)
        return inter / union if union > 0 else 0.0

    for i in range(len(methods)):
        for j in range(i+1, len(methods)):
            m1, m2 = methods[i], methods[j]
            wdict1 = word_lists_dict[m1]
            wdict2 = word_lists_dict[m2]
            cids1 = sorted(wdict1.keys())
            cids2 = sorted(wdict2.keys())

            # Build Jaccard matrix
            mat = np.zeros((len(cids1), len(cids2)))
            for a_idx, cid1 in enumerate(cids1):
                s1 = set(wdict1[cid1])
                for b_idx, cid2 in enumerate(cids2):
                    s2 = set(wdict2[cid2])
                    mat[a_idx, b_idx] = jacc(s1, s2)

            print(f"\n{m1} vs {m2}: Jaccard similarity of cluster WORD sets")
            for a_idx, cid1 in enumerate(cids1):
                row = mat[a_idx, :]
                best_b = np.argmax(row)
                best_score = row[best_b]
                cid2 = cids2[best_b]
                identical = " (IDENTICAL WORD SETS)" if np.isclose(best_score, 1.0) else ""
                print(
                    f"  {m1} cluster {cid1} best matches {m2} cluster {cid2} "
                    f"with Jaccard = {best_score:.3f}{identical}"
                )

            # Heatmap
            df_jac = pd.DataFrame(
                mat,
                index=[f"{m1} c{cid}" for cid in cids1],
                columns=[f"{m2} c{cid}" for cid in cids2]
            )
            plt.figure(figsize=(6, 4))
            sns.heatmap(df_jac, annot=True, fmt=".2f", cmap="Greens", vmin=0, vmax=1)
            plt.title(f"Jaccard of cluster word sets: {m1} vs {m2}")
            plt.tight_layout()
            plt.show()


def cluster_sonnets_by_rare_words(sonnets_map, n_clusters=6):
    """
    Run three clustering methods on rare-word usage:
    1) K-Means
    2) Agglomerative (Ward)
    3) Spectral
    Plot each result and print word lists per cluster,
    plus filtered lists (removing words that appear in >=3 clusters).
    Also compare:
      - Label agreement (ARI, sonnet overlap heatmaps)
      - Word-set agreement (Jaccard, best matches, word-set heatmaps)
    """
    print("\n" + "="*60)
    print("RARE-WORD CLUSTERING OF SONNETS (2–12 SONNET DF)")
    print("="*60)

    sonnet_nums, rare_words, X = build_rare_word_matrix(sonnets_map, min_df=2, max_df=12)
    print(f"Using {len(sonnet_nums)} sonnets and {len(rare_words)} rare-ish words as features.")

    labels_dict = {}
    word_lists_dict = {}

    # 1) K-Means
    print("\nRunning K-Means clustering...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    labels_kmeans = kmeans.fit_predict(X)
    labels_dict["K-Means"] = labels_kmeans
    cluster_words_kmeans = get_cluster_word_lists(rare_words, X, labels_kmeans)
    word_lists_dict["K-Means"] = cluster_words_kmeans
    print_cluster_word_lists(cluster_words_kmeans, "K-Means")
    plot_clusters_with_dates(X, sonnet_nums, labels_kmeans, "K-Means", cluster_words_kmeans)

    # 2) Agglomerative Clustering (Ward)
    print("Running Agglomerative (Ward) clustering...")
    agg = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    labels_agg = agg.fit_predict(X)
    labels_dict["Agglomerative"] = labels_agg
    cluster_words_agg = get_cluster_word_lists(rare_words, X, labels_agg)
    word_lists_dict["Agglomerative"] = cluster_words_agg
    print_cluster_word_lists(cluster_words_agg, "Agglomerative (Ward)")
    plot_clusters_with_dates(X, sonnet_nums, labels_agg, "Agglomerative (Ward)", cluster_words_agg)

    # Dendrogram for Agglomerative, with viridis leaves and link-words
    plot_dendrogram(X, sonnet_nums, rare_words, "Agglomerative (Ward)")

    # 3) Spectral Clustering
    print("Running Spectral Clustering...")
    spec = SpectralClustering(
        n_clusters=n_clusters,
        affinity='nearest_neighbors',
        random_state=42
    )
    labels_spec = spec.fit_predict(X)
    labels_dict["Spectral"] = labels_spec
    cluster_words_spec = get_cluster_word_lists(rare_words, X, labels_spec)
    word_lists_dict["Spectral"] = cluster_words_spec
    print_cluster_word_lists(cluster_words_spec, "Spectral Clustering")
    plot_clusters_with_dates(X, sonnet_nums, labels_spec, "Spectral Clustering", cluster_words_spec)

    # Compare labelings across methods (by sonnet membership)
    compare_clusterings(sonnet_nums, labels_dict)

    # Compare cluster word-structures across methods (Jaccard of word sets)
    compare_cluster_word_structures(word_lists_dict)

    # Reprint filtered word lists (remove words appearing in >=3 clusters, per method)
    print_filtered_cluster_word_lists(cluster_words_kmeans, "K-Means", min_cluster_count=3)
    print_filtered_cluster_word_lists(cluster_words_agg, "Agglomerative (Ward)", min_cluster_count=3)


# --- EXECUTE ---
if 'sonnets_map' in locals():
    analyze_vocabulary_refined(sonnets_map)
    cluster_sonnets_by_rare_words(sonnets_map, n_clusters=6)
else:
    print("Error: 'sonnets_map' not found. Please run previous cells first.")


# --- FINAL COMBINED PCA: temporal metrics + LSA themes + rare-word Ward clusters ---

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.cluster import AgglomerativeClustering
from matplotlib.lines import Line2D
from scipy.spatial import ConvexHull

def combined_pca_temporal_and_rare_clusters(df_temporal, sonnets_map, n_clusters=6):
    """
    Build a combined feature space using:
      - Temporal metrics: Score, RhymeDiversity, NounRatio, VerbRatio, AdjRatio
      - LSA theme components: Theme_* columns from df_temporal
      - Rare-word structure:
          * TruncatedSVD components from the rare-word (2–12 DF) matrix
          * Agglomerative (Ward) clusters on rare-word usage (encoded as one-hot)

    Then run a PCA and plot PC1 vs PC2:
      - Color by composition date (viridis)
      - Marker shape by rare-word Ward cluster
      - Convex hull per cluster

    Finally, print a textual characterization of each cluster.
    """
    # --- 1. Rare-word matrix & Ward clustering ---
    sonnet_nums_rw, rare_words_rw, X_rw = build_rare_word_matrix(sonnets_map, min_df=2, max_df=12)

    # Agglomerative (Ward) on rare-word usage
    agg = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    labels_agg = agg.fit_predict(X_rw)

    # Some low-rank representation of the rare-word matrix
    if X_rw.shape[1] > 1:
        n_rw_comps = min(5, X_rw.shape[1] - 1, X_rw.shape[0] - 1)
        if n_rw_comps < 1:
            n_rw_comps = 1
    else:
        n_rw_comps = 1

    svd_rw = TruncatedSVD(n_components=n_rw_comps, random_state=42)
    X_rw_svd = svd_rw.fit_transform(X_rw)  # shape: (n_sonnets, n_rw_comps)

    # Build a DF of rare-word features keyed by sonnet number
    df_rw = pd.DataFrame({
        "Num": sonnet_nums_rw,
        "RareCluster": labels_agg
    })
    for i in range(n_rw_comps):
        df_rw[f"RareLSA_{i+1}"] = X_rw_svd[:, i]

    # --- 2. Merge with temporal / theme features from df_temporal ---
    # df_temporal has columns: Num, Date, Score, RhymeDiversity, NounRatio, VerbRatio, AdjRatio, Theme_...
    df_combined = pd.merge(df_temporal, df_rw, on="Num", how="inner")

    # Identify theme columns
    theme_cols = [c for c in df_combined.columns if c.startswith("Theme_")]

    # Base metrics
    base_features = ['Score', 'RhymeDiversity', 'NounRatio', 'VerbRatio', 'AdjRatio']

    # Rare-word SVD features
    rare_lsa_cols = [f"RareLSA_{i+1}" for i in range(n_rw_comps)]

    # One-hot encode cluster labels
    rare_cluster_dummies = pd.get_dummies(df_combined['RareCluster'], prefix='RareC')

    # Build final feature matrix
    feature_cols = base_features + theme_cols + rare_lsa_cols
    X_base = df_combined[feature_cols].values
    X_full = np.hstack([X_base, rare_cluster_dummies.values])

    # --- 3. Standardize & PCA ---
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_full)

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    df_combined['PC_comb1'] = X_pca[:, 0]
    df_combined['PC_comb2'] = X_pca[:, 1]

    # --- 4. Plot: PC1 vs PC2 with viridis colors by date & shapes by Ward cluster ---
    df_combined['DateLabel'] = df_combined['Date'].astype(str)
    unique_labels = sorted(df_combined['DateLabel'].unique(), key=lambda x: float(x))

    palette_colors = sns.color_palette("viridis", n_colors=len(unique_labels))
    palette_dict = {label: color for label, color in zip(unique_labels, palette_colors)}

    markers = ['o', 's', '^', 'D', 'P', 'X', '*', 'v']

    plt.figure(figsize=(12, 8))
    ax = plt.gca()

    # Convex hull per Ward cluster (faint, black outline)
    for cid in sorted(df_combined['RareCluster'].unique()):
        pts = df_combined[df_combined['RareCluster'] == cid][['PC_comb1', 'PC_comb2']].values
        if len(pts) < 3:
            continue
        try:
            hull = ConvexHull(pts)
            hull_pts = pts[hull.vertices]
            ax.fill(
                hull_pts[:, 0],
                hull_pts[:, 1],
                facecolor='lightgray',
                edgecolor='black',
                alpha=0.15,
                linewidth=0.15
            )
        except Exception:
            # Degenerate cases: skip hull
            pass

    # Plot points
    for _, row in df_combined.iterrows():
        x, y = row['PC_comb1'], row['PC_comb2']
        date_label = row['DateLabel']
        color = palette_dict[date_label]
        c_id = int(row['RareCluster'])
        marker = markers[c_id % len(markers)]

        ax.scatter(
            x, y,
            c=[color],
            marker=marker,
            edgecolors='black',
            s=70,
            alpha=0.9
        )
        ax.text(
            x + 0.02,
            y + 0.02,
            str(int(row['Num'])),
            fontsize=7
        )

    # Legends: date (color) and Ward cluster (shape)
    date_handles = [
        Line2D(
            [0], [0],
            marker='o', color='w',
            label=lab,
            markerfacecolor=palette_dict[lab],
            markersize=8
        )
        for lab in unique_labels
    ]

    cluster_ids = sorted(df_combined['RareCluster'].unique())
    cluster_handles = [
        Line2D(
            [0], [0],
            marker=markers[c % len(markers)],
            color='k',
            linestyle='None',
            label=f"RareC {c}"
        )
        for c in cluster_ids
    ]

    leg1 = ax.legend(
        handles=date_handles,
        title="Hypothesized Date",
        loc='upper left'
    )
    ax.add_artist(leg1)

    ax.legend(
        handles=cluster_handles,
        title="Ward Rare-word Cluster",
        loc='lower left'
    )

    ax.set_title(
        "Combined PCA: temporal metrics + LSA themes + rare-word Ward clusters"
    )
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%} var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%} var)")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # --- 5. Textual characterization of each cluster ---
    print("\n" + "="*70)
    print("CLUSTER CHARACTERIZATION (Agglomerative Ward on rare words)")
    print("="*70)

    # Map from sonnet number -> row index in X_rw, so we can align clusters with X_rw
    num_to_idx_rw = {num: idx for idx, num in enumerate(sonnet_nums_rw)}

    for cid in cluster_ids:
        df_c = df_combined[df_combined['RareCluster'] == cid]
        n_c = len(df_c)

        print(f"\n--- Cluster {cid} ---")
        print(f"Number of sonnets: {n_c}")

        if n_c == 0:
            print("  (Empty cluster)")
            continue

        # 5a) Average temporal metrics
        avg_metrics = {feat: df_c[feat].mean() for feat in base_features}
        metrics_str = ", ".join(f"{k}={v:.3f}" for k, v in avg_metrics.items())
        print(f"  Avg metrics: {metrics_str}")

        # 5b) Dominant themes (by absolute mean loading)
        if theme_cols:
            theme_means = df_c[theme_cols].mean(axis=0)
            theme_rank = theme_means.abs().sort_values(ascending=False)
            top_themes = theme_rank.index[:3].tolist()
            print("  Dominant themes:", ", ".join(top_themes))
        else:
            print("  Dominant themes: (none; no Theme_ columns found)")

        # 5c) Top rare words defining this cluster
        # Find indices in the rare-word matrix for sonnets in this cluster
        idxs_rw = [
            num_to_idx_rw[num]
            for num in df_c['Num'].tolist()
            if num in num_to_idx_rw
        ]
        if idxs_rw:
            subX = X_rw[idxs_rw, :]
            word_counts = np.asarray(subX.sum(axis=0)).ravel()
            order = np.argsort(word_counts)[::-1]
            top_n = 20
            top_words = [
                rare_words_rw[i]
                for i in order[:top_n]
                if word_counts[i] > 0
            ]
            print("  Top rare words:", ", ".join(top_words))
        else:
            print("  Top rare words: (no alignment with rare-word matrix)")

    print("\n(End of cluster characterization.)")


# --- EXECUTE FINAL COMBINED PCA ---
# Requires:
#   - df_temporal from your temporal/LSA cell
#   - sonnets_map (already built in this notebook)
if 'df_temporal' in locals() and 'sonnets_map' in locals():
    combined_pca_temporal_and_rare_clusters(df_temporal, sonnets_map, n_clusters=6)
else:
    print("Note: df_temporal or sonnets_map not found. "
          "Please run the temporal feature cell (extract_sonnet_features + analyze_and_visualize) first.")


# %% [markdown] id="dn1gOmx2fd7V"
# ## Search for Hidden Sonnets, 1 Play

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} id="iMk3tIa1196X" outputId="d478d359-7d5a-426a-f982-ced383fcb035"
# Cell 10: Specific Play Analysis with Enhanced Visualization


'''
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
        "THE RAPE OF LUCRECE", "VENUS AND ADONIS"
'''

# Configuration
WORK_TO_ANALYZE = "A MIDSUMMER NIGHT’S DREAM"

def analyze_work(title, lines):
    print(f"Analyzing {title}...")
    scores = []
    # Multi-window scanning for variable length sonnets
    window_sizes = [10, 11, 14, 15, 18]

    clean_data = [] # List of tuples: (text, speaker)
    boundaries = [] # (index, label, type)

    current_clean_idx = 0
    current_speaker = "UNKNOWN"

    for line in lines:
        s_line = line.strip()
        if not s_line: continue

        # Check boundaries
        if s_line.upper().startswith("ACT "):
            label = s_line.split('.')[0]
            boundaries.append((current_clean_idx, label, 'act'))
            continue
        if s_line.upper().startswith("SCENE "):
            label = s_line.split('.')[0]
            boundaries.append((current_clean_idx, label, 'scene'))
            continue

        # Check Speaker
        if s_line.isupper() and s_line.endswith(".") and len(s_line) < 20 and "SCENE" not in s_line and "ACT" not in s_line:
            current_speaker = s_line.strip(".")
            continue

        if len(s_line) > 10:
            clean_data.append((s_line, current_speaker))
            current_clean_idx += 1

    # Extract just text for scoring
    clean_lines_text = [x[0] for x in clean_data]

    # Scan each position with multiple window sizes
    for i in range(len(clean_lines_text)):
        best_score_for_pos = 0.0

        for w_size in window_sizes:
            if i + w_size <= len(clean_lines_text):
                window = clean_lines_text[i : i + w_size]
                score = detector.score_window(window)
                if score > best_score_for_pos:
                    best_score_for_pos = score

        scores.append(best_score_for_pos)

    return scores, clean_data, boundaries

# Analyze
lines = works[WORK_TO_ANALYZE]
scores, clean_data, boundaries = analyze_work(WORK_TO_ANALYZE, lines)

# --- Prepare Classifier Top 2% Signals ---
try:
    # Filter for current play (Column is 'Play')
    play_candidates = sonnet_candidates_df[sonnet_candidates_df['Play'] == WORK_TO_ANALYZE]

    # Get thresholds (approximate from global distribution if possible, or just top 2% of this play?)
    sup_threshold = np.percentile(sonnet_candidates_df['Supervised_Score'], 98)
    pu_threshold = np.percentile(sonnet_candidates_df['PU_Score'], 98)

    # Find indices where these occur
    sup_indices = play_candidates[play_candidates['Supervised_Score'] >= sup_threshold]['Start_Line'].values
    pu_indices = play_candidates[play_candidates['PU_Score'] >= pu_threshold]['Start_Line'].values

    overlap_indices = np.intersect1d(sup_indices, pu_indices)
    # Remove overlap from individual lists for cleaner plotting
    sup_only = np.setdiff1d(sup_indices, overlap_indices)
    pu_only = np.setdiff1d(pu_indices, overlap_indices)

except NameError:
    print("Warning: 'sonnet_candidates_df' not found. Skipping classifier signal overlay.")
    sup_only, pu_only, overlap_indices = [], [], []
except KeyError:
    print("Warning: 'Play' column not found in dataframe. Skipping classifier signal overlay.")
    sup_only, pu_only, overlap_indices = [], [], []


# Plot 1: Sonnet Detection Signal
plt.figure(figsize=(14, 8))
plt.plot(scores, label=WORK_TO_ANALYZE, linewidth=.5, color='gray', alpha=0.7)
plt.title(f"Sonnet Detection Signal in {WORK_TO_ANALYZE}")
plt.xlabel("Line Number (approx)")
plt.ylabel("Sonnet Score")

# Add Classifier Signal Lines at Top
# We'll plot short horizontal lines at y=1.18, 1.16, 1.14 to show regions (moved higher)
for idx in sup_only:
    plt.hlines(y=0.99, xmin=idx, xmax=idx+14, colors='lightsalmon', linewidth=2, label='Supervised (2%)' if 'Supervised (2%)' not in plt.gca().get_legend_handles_labels()[1] else "")
for idx in pu_only:
    plt.hlines(y=0.98, xmin=idx, xmax=idx+14, colors='skyblue', linewidth=2, label='PU (2%)' if 'PU (2%)' not in plt.gca().get_legend_handles_labels()[1] else "")
for idx in overlap_indices:
    plt.hlines(y=0.97, xmin=idx, xmax=idx+14, colors='palegreen', linewidth=2, label='Overlap (Top 2%)' if 'Overlap' not in plt.gca().get_legend_handles_labels()[1] else "")

# 1. Add Boundaries (Staggered Labels)
for idx, label, b_type in boundaries:
    if b_type == 'act':
        plt.axvline(x=idx, color='black', linewidth=0.4)
        plt.text(idx, 0.02, label, rotation=90, fontsize=6, verticalalignment='bottom') # Higher for Act
    elif b_type == 'scene':
        plt.axvline(x=idx, color='black', linewidth=0.2)
        plt.text(idx, 0.08, label, rotation=90, fontsize=6, verticalalignment='bottom') # Lower for Scene

# 2. Top 10 Markers (With NMS)
scores_array = np.array(scores)
sorted_indices = scores_array.argsort()[::-1]

# NMS Logic (Window 18)
nms_indices = []
taken_mask = np.zeros(len(scores), dtype=bool)
nms_window = 18

for idx in sorted_indices:
    if len(nms_indices) >= 20:
        break
    if taken_mask[idx]:
        continue

    nms_indices.append(idx)
    start = max(0, idx - nms_window + 1)
    end = min(len(scores), idx + nms_window)
    taken_mask[start:end] = True

top_indices = nms_indices

# Plot markers
for idx in top_indices:
    score = scores[idx]
    plt.plot(idx, score, 'k.', markersize=5)
    plt.text(idx, score, f"{idx}", fontsize=6, rotation=45, ha='left', va='bottom')

plt.legend(loc='upper right')
plt.ylim(0, 1.05)
plt.show()

# Top 10 Segments Output
print(f"\nTop 10 Scoring Segments in {WORK_TO_ANALYZE} (Distinct Regions):\n" + "="*40)
for i, idx in enumerate(top_indices):
    score = scores[idx]
    print(f"\nRank {i+1} | Score: {score:.4f} | Start Line: {idx}")
    print("-" * 20)
    # Print 14 lines context (or up to 18?) - Let's print 14 as standard view
    last_speaker = None
    for j in range(idx, min(idx+14, len(clean_data))):
        line_text, speaker = clean_data[j]
        prefix = ""
        if speaker != last_speaker:
            prefix = f"[{speaker}] "
            last_speaker = speaker
        print(f"{prefix}{line_text}")

# Special Analysis for THE SONNETS: Ranked Graph
if WORK_TO_ANALYZE == "THE SONNETS":
    print("\nGenerating Ranked Sonnet Graph...")

    # Parse individual sonnets (Quick re-parse for this visualization)
    sonnets_map = {}
    current_sonnet = []
    current_num = 0
    roman_re = re.compile(r'^[XVI]+(\.)?$')

    for line in lines:
        s_line = line.strip()
        if not s_line: continue

        # Check for header (Roman numeral or digit)
        is_header = False
        if s_line.isdigit(): is_header = True
        elif roman_re.match(s_line): is_header = True

        if is_header:
            if current_sonnet and len(current_sonnet) >= 10:
                sonnets_map[current_num] = current_sonnet
            current_sonnet = []
            try: current_num = int(s_line.strip('.'))
            except: current_num += 1
        else:
            current_sonnet.append(s_line)
    if current_sonnet: sonnets_map[current_num] = current_sonnet

    # Score each sonnet
    sonnet_scores = []
    for num, s_lines in sonnets_map.items():
        if len(s_lines) < 8: continue
        score = detector.score_window(s_lines)
        sonnet_scores.append((num, score))

    # Sort by score (Ascending for barh to put highest at top)
    sonnet_scores.sort(key=lambda x: x[1])

    # Prepare Plot Data
    names = [f"Sonnet {x[0]}" for x in sonnet_scores]
    values = [x[1] for x in sonnet_scores]

    # Plot
    plt.figure(figsize=(8, 30)) # Tall and narrow
    plt.barh(names, values, color='skyblue', edgecolor='black', linewidth=0.05)
    plt.xlabel("Sonnet Score")
    plt.title("All Sonnets Ranked by Score")
    plt.tick_params(axis='y', labelsize=5) # Tiny font
    plt.xlim(0, 1.1)
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

# --- Wavelet Analysis ---
sig = np.array(scores)
sig = sig - np.mean(sig)
t = np.arange(len(sig))

# 2. CWT with COMPLEX Morlet Wavelet
scales = np.linspace(10, 600, 400)
cwtmatr, freqs = pywt.cwt(sig, scales, 'cmor1.5-1.0')
power = (np.abs(cwtmatr)) ** 2
periods = 1.0 / freqs
coi = np.sqrt(2) * scales

# Plot 1: Power Spectrum
plt.figure(figsize=(14, 8))
levels = np.linspace(0, np.max(power), 20)
CS = plt.contourf(t, periods, power, levels=levels, cmap='jet', extend='both')
plt.colorbar(CS, label="Wavelet Power")

d_to_edge = np.minimum(t, (len(t) - 1) - t).astype(float)
s_boundary = d_to_edge / np.sqrt(2)
s_boundary_clipped = np.clip(s_boundary, scales.min(), scales.max())
P_boundary = np.interp(s_boundary_clipped, scales, periods)

plt.fill_between(t, P_boundary, periods.max(), color='lightgray', alpha=0.5, label='Outside COI', zorder=2)
plt.plot(t, P_boundary, color='k', linewidth=3, zorder=3, label='COI boundary')

plt.title(f"Complex Morlet Wavelet Power Spectrum: {WORK_TO_ANALYZE}")
plt.xlabel("Line Number")
plt.ylabel("Period (Lines)")
plt.ylim(periods.max(), periods.min())
plt.yscale('log')
plt.yticks([20, 50, 100, 200, 300, 400, 500, 600], ['20', '50', '100', '200', '300', '400', '500', '600'])
plt.grid(True, alpha=0.3, which='both')
plt.legend(loc='upper right')
plt.show()

# Plot 2: Wavelet Significance
plt.figure(figsize=(14, 8))
power_smooth = gaussian_filter(power, sigma=(2, 2))

# Significance Threshold (White Noise Model)
variance = np.var(sig)
significance_threshold = variance * (stats.chi2.ppf(0.95, 2) / 2)

levels = np.linspace(0, np.max(power), 20)
CS = plt.contourf(t, periods, power_smooth, levels=levels, cmap='jet', extend='both')
cbar = plt.colorbar(CS, label="Wavelet Power")

plt.contour(t, periods, power_smooth, levels=[significance_threshold], colors='k', linewidths=1.5)

plt.fill_between(t, P_boundary, periods.max(), color='white', alpha=0.5, hatch='x', zorder=2)
plt.plot(t, P_boundary, color='k', linewidth=2, zorder=3)

plt.title(f"Wavelet Power Spectrum (Complex Morlet) with 95% Significance: {WORK_TO_ANALYZE}")
plt.xlabel("Line Number")
plt.ylabel("Period (Lines)")
plt.yscale('log')
plt.yticks([20, 50, 100, 200, 300, 400, 500, 600], ['20', '50', '100', '200', '300', '400', '500', '600'])
plt.ylim(periods.max(), periods.min())
plt.grid(True, alpha=0.3, which='both')
plt.show()

# --- Rare Word Analysis ---
def tokenize(text):
    return re.findall(r"\w+", text.lower())

all_text = " ".join([line_text for (line_text, speaker) in clean_data])
all_tokens = tokenize(all_text)
word_counts = Counter(all_tokens)

# Histogram
bins = list(range(1, 21))
bin_counts = [0] * 20
for w, c in word_counts.items():
    if c >= 20: bin_counts[19] += 1
    else: bin_counts[c-1] += 1

plt.figure(figsize=(12, 6))
bars = plt.bar(bins, bin_counts, color='skyblue', edgecolor='black')
plt.xlabel("Number of Times Word is Used in the Play")
plt.ylabel("Count of Unique Words")
plt.title(f"Vocabulary Usage Distribution in {WORK_TO_ANALYZE}")
plt.xticks(bins, [str(b) if b < 20 else "20+" for b in bins])
plt.grid(axis='y', linestyle='--', alpha=0.3)
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height, f'{height}', ha='center', va='bottom', fontsize=8)
plt.tight_layout()
plt.show()

print(f"\nTotal unique word types in {WORK_TO_ANALYZE}: {len(word_counts)}")

# Test Groups
n_lines = len(clean_data)
poetic_indices = set()
for idx in top_indices:
    for j in range(idx, min(idx + 14, n_lines)):
        poetic_indices.add(j)

excluded_for_normal = set()
for idx in top_indices:
    start = max(0, idx - 10)
    end = min(n_lines, idx + 14 + 10)
    for j in range(start, end):
        excluded_for_normal.add(j)

normal_indices = [i for i in range(n_lines) if i not in excluded_for_normal]
poetic_indices_sorted = sorted(poetic_indices)

print(f"\nLines in TEST_GROUP_POETIC: {len(poetic_indices_sorted)}")
print(f"Lines in TEST_GROUP_NORMAL: {len(normal_indices)}")

def collect_tokens_for_indices(indices):
    tokens = []
    for i in indices:
        line_text, speaker = clean_data[i]
        tokens.extend(tokenize(line_text))
    return tokens

poetic_tokens = collect_tokens_for_indices(poetic_indices_sorted)
normal_tokens = collect_tokens_for_indices(normal_indices)

def compute_rare_percentages(tokens, word_counts):
    total = len(tokens)
    if total == 0: return 0.0, 0.0
    rare_1_2 = sum(1 for w in tokens if word_counts[w] in (1, 2))
    rare_1_2_3 = sum(1 for w in tokens if word_counts[w] in (1, 2, 3))
    return rare_1_2 / total, rare_1_2_3 / total

p_r12, p_r123 = compute_rare_percentages(poetic_tokens, word_counts)
n_r12, n_r123 = compute_rare_percentages(normal_tokens, word_counts)

print("\n=== RARE-WORD PERCENTAGES (TOKEN-BASED) ===")
print(f"TEST_GROUP_POETIC:  total tokens = {len(poetic_tokens)}")
print(f"  % of tokens with frequency 1–2:  {p_r12*100:.2f}%")
print(f"  % of tokens with frequency 1–3:  {p_r123*100:.2f}%")
print(f"\nTEST_GROUP_NORMAL:  total tokens = {len(normal_tokens)}")
print(f"  % of tokens with frequency 1–2:  {n_r12*100:.2f}%")
print(f"  % of tokens with frequency 1–3:  {n_r123*100:.2f}%")
print("\nComparison (Poetic minus Normal):")
print(f"  Δ% (1–2):  {(p_r12 - n_r12)*100:.2f} percentage points")
print(f"  Δ% (1–3):  {(p_r123 - n_r123)*100:.2f} percentage points")

# T-Tests
def rare_counts_per_line(indices, freq_set):
    counts_per_line = []
    for i in indices:
        line_text, speaker = clean_data[i]
        tokens = tokenize(line_text)
        c = sum(1 for w in tokens if word_counts[w] in freq_set)
        counts_per_line.append(c)
    return counts_per_line

freq_bins = {"1-use words": {1}, "1–2-use words": {1, 2}, "1–3-use words": {1, 2, 3}}
print("\n=== T-TESTS: RARE WORD COUNTS PER LINE (POETIC vs NORMAL) ===")
for label, fset in freq_bins.items():
    poetic_counts = rare_counts_per_line(poetic_indices_sorted, fset)
    normal_counts = rare_counts_per_line(normal_indices, fset)
    t_stat, p_val = stats.ttest_ind(poetic_counts, normal_counts, equal_var=False)
    print(f"\n{label}:")
    print(f"  Poetic lines:  mean={np.mean(poetic_counts):.3f}, std={np.std(poetic_counts, ddof=1):.3f}, n={len(poetic_counts)}")
    print(f"  Normal lines:  mean={np.mean(normal_counts):.3f}, std={np.std(normal_counts, ddof=1):.3f}, n={len(normal_counts)}")
    print(f"  Welch t-test: t = {t_stat:.3f}, p = {p_val:.4g}")

# --- NEW: Score Distribution Histogram (Play vs All Sonnets) ---
print("\nGenerating Score Distribution Comparison...")

# 1. Collect Scores for All Canonical Sonnets + Play Sonnets
all_sonnet_scores = []

# Canonical
sonnets_text = works.get("THE SONNETS", [])
s_map = {}
curr_s = []
curr_n = 0
for l in sonnets_text:
    if l.strip().isdigit() or re.match(r'^[XVI]+(\.)?$', l.strip()):
        if curr_s and len(curr_s)>=10: s_map[curr_n] = curr_s
        curr_s = []
        try: curr_n = int(l.strip('.'))
        except: curr_n += 1
    else:
        curr_s.append(l)
if curr_s: s_map[curr_n] = curr_s

for s_lines in s_map.values():
    if len(s_lines) >= 8:
        all_sonnet_scores.append(detector.score_window(s_lines))

# Play Sonnets
if "PLAY SONNETS" in works:
    ps_lines = works["PLAY SONNETS"]
    curr_ps = []
    for l in ps_lines:
        if l.startswith("=== PLAY SONNET"):
            if curr_ps: all_sonnet_scores.append(detector.score_window(curr_ps))
            curr_ps = []
        elif l.strip():
            curr_ps.append(l)
    if curr_ps: all_sonnet_scores.append(detector.score_window(curr_ps))

# 2. Plot Overlapping KDEs (Density)
plt.figure(figsize=(12, 6))

# KDE for THIS PLAY's window scores - TIGHTER BW
sns.kdeplot(scores, color='gray', label=f'Windows in {WORK_TO_ANALYZE}', fill=True, alpha=0.3, bw_adjust=0.2)

# KDE for ALL SONNETS (Canonical + Play) - TIGHTER BW
sns.kdeplot(all_sonnet_scores, color='red', label='All Sonnets (Canonical + Play)', fill=True, alpha=0.3, bw_adjust=0.2)

plt.title(f"Score Distribution (KDE): {WORK_TO_ANALYZE} vs. All Sonnets")
plt.xlabel("Sonnet Score")
plt.ylabel("Density")
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.xlim(0, 1.1)
plt.show()

# --- NEW: Grouped Bar Chart (Word Usage Fraction) ---
print("\nGenerating Word Usage Comparison Chart...")

def grouped_bar_chart_dualaxis(values_list, labels, title, ylabel_left, ylabel_right, works_legend):
    """
    values_list: list of lists, one per work (each length = number of bins)
    labels: x-axis bin labels
    """
    n_bins = len(labels)
    x = np.arange(n_bins)
    n_works = len(values_list)
    width = 0.15

    # color map
    cmap = plt.cm.get_cmap("viridis", n_works)
    colors = [cmap(i) for i in range(n_works)]

    # offsets for grouped bars
    offsets = (np.arange(n_works) - (n_works - 1) / 2) * width

    # Figure and twin axis
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()

    # Plot bins 1–4 on ax1
    for i, (vals, work) in enumerate(zip(values_list, works_legend)):
        ax1.bar(
            x[:-1] + offsets[i],     # first n_bins-1 bins
            vals[:-1],
            width,
            label=work,
            color=colors[i],
            zorder=3
        )

    # Plot last bin (5+) on ax2
    for i, (vals, work) in enumerate(zip(values_list, works_legend)):
        ax2.bar(
            x[-1] + offsets[i],      # last bin only
            vals[-1],
            width,
            color=colors[i],
            alpha=0.8,
            zorder=2
        )

    # Vertical divider
    divider_x = (x[-2] + x[-1]) / 2
    ax1.axvline(divider_x, color="black", linewidth=1.5)

    # Axes labels and ticks
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel(ylabel_left)
    ax2.set_ylabel(ylabel_right)
    ax1.set_title(title)
    ax1.legend(loc="upper center")

    plt.tight_layout()
    plt.show()

# 1. Prepare Data for 3 Groups: Sonnets, Play, Top 20 Candidates

# Helper to get bin fractions [1, 2, 3, 4, 5+]
def get_bin_fractions(tokens, reference_counts=None):
    """
    Calculates the fraction of UNIQUE TYPES in 'tokens' that have frequency 1, 2, 3, 4, 5+
    in the 'reference_counts'.
    If reference_counts is None, it uses the counts within 'tokens' itself (self-referential).
    """
    if not tokens: return [0]*5

    # Identify unique types in the target text
    unique_types = set(tokens)
    if not unique_types: return [0]*5

    # Determine counts to use
    if reference_counts is None:
        # Self-referential counts
        counts = Counter(tokens)
    else:
        # Use external reference counts
        counts = reference_counts

    # Bins: 1, 2, 3, 4, 5+
    bin_counts = [0]*5
    for w in unique_types:
        c = counts.get(w, 0) # Get count from reference
        if c == 1: bin_counts[0] += 1
        elif c == 2: bin_counts[1] += 1
        elif c == 3: bin_counts[2] += 1
        elif c == 4: bin_counts[3] += 1
        else: bin_counts[4] += 1

    total_types = len(unique_types)
    return [b / total_types for b in bin_counts]

# A. Sonnets (Self-referential: how rare are words in the Sonnets corpus?)
sonnets_text_list = works.get("THE SONNETS", [])
sonnets_tokens = tokenize(" ".join(sonnets_text_list))
# For Sonnets, we use Sonnet counts as reference (standard practice)
frac_sonnets = get_bin_fractions(sonnets_tokens, reference_counts=None)

# B. Current Play (Self-referential: how rare are words in this Play?)
play_tokens = all_tokens # Already computed above
# For Play, we use Play counts (word_counts computed earlier)
frac_play = get_bin_fractions(play_tokens, reference_counts=word_counts)

# C. Top 20 Candidates (Subset of Play)
# CRITICAL FIX: Use GLOBAL PLAY COUNTS as reference
top_20_tokens = collect_tokens_for_indices(top_indices)
frac_top20 = get_bin_fractions(top_20_tokens, reference_counts=word_counts)

# Plot
bin_labels = ["1", "2", "3", "4", "5+"]
works_legend = ["Sonnets", WORK_TO_ANALYZE]

grouped_bar_chart_dualaxis(
    [frac_sonnets, frac_play],
    bin_labels,
    "Fraction of Word Usage (Unique Types) by Frequency",
    "Fraction (Freq 1–4)",
    "Fraction (Freq 5+)",
    works_legend
)


# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} id="3oJHOS43dai6" outputId="b6e1c3fb-418b-4f2a-bd3b-8114a532afb3"
#from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
#from sklearn.decomposition import TruncatedSVD, PCA
#from sklearn.preprocessing import StandardScaler
#from sklearn.metrics.pairwise import cosine_similarity
#from collections import defaultdict

# Ensure NLTK resources are available
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('universal_tagset', quiet=True)
# Some newer NLTK installs use this auxiliary resource; ignore errors if it doesn't exist
try:
    nltk.download('punkt_tab', quiet=True)
except:
    pass

# --- 1. Date Categorization ---
def get_sonnet_date(num):
    if 1 <= num <= 60: return 1596
    elif 61 <= num <= 77: return 1594.5
    elif 78 <= num <= 86: return 1599
    elif 87 <= num <= 103: return 1594.5
    elif 104 <= num <= 126: return 1602
    elif 127 <= num <= 144: return 1593
    elif num == 145: return 1582
    elif 146 <= num <= 154: return 1593
    return None

# --- 2. Feature Extraction ---
def extract_sonnet_features(sonnets_map, detector):
    data = []
    corpus = []

    print("Extracting features for Temporal Analysis...")

    for num in sorted(sonnets_map.keys()):
        lines = sonnets_map[num]
        if len(lines) < 14:
            continue

        # A. Basic Info
        date = get_sonnet_date(num)
        if date is None:
            continue

        # B. Score
        score = detector.score_window(lines[:14])

        # C. Rhyme Scheme Diversity
        last_words = []
        for line in lines[:14]:
            words = re.findall(r"\w+", line)
            last_words.append(words[-1] if words else "")

        q_types = []
        for i in range(0, 12, 4):
            w = last_words[i:i+4]
            if detector.check_rhyme(w[0], w[2]) and detector.check_rhyme(w[1], w[3]):
                q_types.append("ABAB")
            elif detector.check_rhyme(w[0], w[3]) and detector.check_rhyme(w[1], w[2]):
                q_types.append("ABBA")
            elif detector.check_rhyme(w[0], w[1]) and detector.check_rhyme(w[2], w[3]):
                q_types.append("AABB")
            else:
                q_types.append("Irregular")

        rhyme_diversity = len(set(q_types))

        # D. Grammatical Style (POS Tags)
        full_text = " ".join(lines)
        tokens = nltk.word_tokenize(full_text)
        tags = nltk.pos_tag(tokens, tagset='universal')

        counts = defaultdict(int)
        for word, tag in tags:
            counts[tag] += 1

        total_words = len(tokens)
        noun_ratio = counts['NOUN'] / total_words if total_words else 0
        verb_ratio = counts['VERB'] / total_words if total_words else 0
        adj_ratio = counts['ADJ'] / total_words if total_words else 0

        row = {
            'Num': num,
            'Date': date,
            'Score': score,
            'RhymeDiversity': rhyme_diversity,
            'NounRatio': noun_ratio,
            'VerbRatio': verb_ratio,
            'AdjRatio': adj_ratio,
            'Text': full_text
        }
        data.append(row)
        corpus.append(full_text)

    df = pd.DataFrame(data)
    return df, corpus

BANNED_THEME_WORDS = {
    'thy', 'thou', 'doth', 'love', 'shall', 'eyes',
    'heart', 'beauty', 'self', 'thine', 'time', 'did',
    'dost', 'day', 'days', 'art', 'st', 'thee', 'eye',
    'hath', 'sweet'
}

def make_theme_name(pos_words, neg_words, max_tokens=5, n_pos_first=3):
    """
    Build a compact label like:
    pos_words = ['love', 'heart', 'time']
    neg_words = ['death', 'decay']
    -> 'Love[+]Heart[+]Time[+]Death[-]Decay[-]'

    max_tokens: total #word-sign pairs to include
    n_pos_first: how many positives to include before negatives
    """
    # Filter out banned words (in case they slipped through)
    pos_words = [w for w in pos_words if w not in BANNED_THEME_WORDS]
    neg_words = [w for w in neg_words if w not in BANNED_THEME_WORDS]

    # Take a few positive and a few negative words
    pos_words = pos_words[:n_pos_first]
    remaining = max_tokens - len(pos_words)
    neg_words = neg_words[:max(0, remaining)]

    tokens = []
    for w in pos_words:
        base = re.sub(r'[^A-Za-z0-9]+', '', w)
        if base:
            tokens.append(f"{base.capitalize()}[+]")
    for w in neg_words:
        base = re.sub(r'[^A-Za-z0-9]+', '', w)
        if base:
            tokens.append(f"{base.capitalize()}[-]")

    if not tokens:
        return "Unnamed"
    return "".join(tokens)  # no separators; keeps column name compact


# --- 3. Group Analysis & PCA ---
def analyze_and_visualize(df, corpus):
    # A. Vocabulary & Thematic Analysis (TF-IDF + LSA)
    print("Running Vocabulary & Thematic Analysis...")
    vectorizer = CountVectorizer(stop_words='english', min_df=2)
    X_counts = vectorizer.fit_transform(corpus)
    tfidf = TfidfTransformer()
    X_tfidf = tfidf.fit_transform(X_counts)

    # LSA for "Thematic" components (5 dims)
    n_themes = 5
    lsa = TruncatedSVD(n_components=n_themes, random_state=42)
    X_lsa = lsa.fit_transform(X_tfidf)

    # ---- derive human-readable theme names from pos/neg words ----
    terms = vectorizer.get_feature_names_out()
    theme_col_names = []

    print("\nIdentified LSA themes (top 15 positive & negative words):")
    for comp_idx, comp in enumerate(lsa.components_):
        # full sorted indices for this component
        full_pos_sorted = np.argsort(comp)[::-1]
        full_neg_sorted = np.argsort(comp)

        # For printing: top 15 positive and negative (unfiltered)
        pos_idx = full_pos_sorted[:15]
        neg_idx = full_neg_sorted[:15]
        top_pos_words = [terms[i] for i in pos_idx]
        top_neg_words = [terms[i] for i in neg_idx]

        # For naming: walk down the sorted lists, skipping banned words
        name_pos_words = []
        for idx in full_pos_sorted:
            w = terms[idx]
            if w in BANNED_THEME_WORDS:
                continue
            name_pos_words.append(w)
            if len(name_pos_words) >= 15:  # enough candidates for naming
                break

        name_neg_words = []
        for idx in full_neg_sorted:
            w = terms[idx]
            if w in BANNED_THEME_WORDS:
                continue
            name_neg_words.append(w)
            if len(name_neg_words) >= 15:
                break

        # If filtering somehow removed everything, fall back to original lists
        if not name_pos_words and not name_neg_words:
            name_pos_words = top_pos_words
            name_neg_words = top_neg_words

        # Make a name from 5 words (3 positive, 2 negative by default)
        theme_name = make_theme_name(
            pos_words=name_pos_words,
            neg_words=name_neg_words,
            max_tokens=5,
            n_pos_first=3
        )
        col_name = f"Theme_{theme_name}"
        theme_col_names.append(col_name)

        print(f"\n  Theme {comp_idx+1}: {col_name}")
        print("     Top positive words:",
              ", ".join(f"{w}[+]" for w in top_pos_words))
        print("     Top negative words:",
              ", ".join(f"{w}[-]" for w in top_neg_words))

        # add this component as a column
        df[col_name] = X_lsa[:, comp_idx]

    # --- The rest of analyze_and_visualize is unchanged ---
    # B. Group Comparisons
    print("\n" + "="*60)
    print("TEMPORAL GROUP ANALYSIS")
    print("="*60)

    groups = df.groupby('Date')
    results = []

    # Precompute the global centroid once
    X_tfidf_dense = X_tfidf.toarray()
    global_centroid = np.mean(X_tfidf_dense, axis=0)

    for date, group in groups:
        # 1. Score & Rhyme
        avg_score = group['Score'].mean()
        avg_rhyme_div = group['RhymeDiversity'].mean()

        # 2. Vocabulary Distinctness
        group_indices = group.index
        group_vectors = X_tfidf_dense[group_indices, :]
        group_centroid = np.mean(group_vectors, axis=0)

        vocab_sim = cosine_similarity(
            group_centroid.reshape(1, -1),
            global_centroid.reshape(1, -1)
        )[0][0]
        vocab_distinctness = 1.0 - vocab_sim

        # 3. Grammar
        avg_noun = group['NounRatio'].mean()
        avg_verb = group['VerbRatio'].mean()

        results.append({
            'Date': date,
            'Count': len(group),
            'Avg Score': avg_score,
            'Avg Rhyme Div': avg_rhyme_div,
            'Vocab Distinctness': vocab_distinctness,
            'Avg Noun Ratio': avg_noun,
            'Avg Verb Ratio': avg_verb
        })

    results_df = pd.DataFrame(results).sort_values('Date')
    print(results_df.to_string(index=False))

    # C. PCA Visualization
    print("\nGenerating PCA...")

    base_features = ['Score', 'RhymeDiversity', 'NounRatio', 'VerbRatio', 'AdjRatio']
    feature_cols = base_features + theme_col_names
    X = df[feature_cols].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=4, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    df['PC1'] = X_pca[:, 0]
    df['PC2'] = X_pca[:, 1]
    df['PC3'] = X_pca[:, 2]
    df['PC4'] = X_pca[:, 3]

    loadings_df = pd.DataFrame(
        pca.components_.T,
        index=feature_cols,
        columns=['PC1', 'PC2', 'PC3', 'PC4']
    )

    df['DateLabel'] = df['Date'].astype(str)
    unique_labels = sorted(df['DateLabel'].unique(), key=lambda x: float(x))

    palette_colors = sns.color_palette("viridis", n_colors=len(unique_labels))
    palette_dict = {label: color for label, color in zip(unique_labels, palette_colors)}

    # === 1) PC1 vs PC2 scatter (all points labeled) ===
    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        data=df,
        x='PC1', y='PC2',
        hue='DateLabel',
        hue_order=unique_labels,
        palette=palette_dict,
        s=100,
        edgecolor='black',
        alpha=0.8
    )

    for i in range(len(df)):
        plt.text(
            df.PC1.iloc[i] + 0.02,
            df.PC2.iloc[i],
            str(df.Num.iloc[i]),
            fontsize=8
        )

    plt.title("PCA of Shakespeare's Sonnets (PC1 vs PC2, Colored by Date)")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%} var)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%} var)")
    plt.legend(title="Composition Date")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # === 2) Loadings for PC1 & PC2 ===
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    loadings_df['PC1'].sort_values().plot(kind='barh', ax=axes[0])
    axes[0].set_title("PC1 Loadings")
    axes[0].set_xlabel("Loading")
    axes[0].grid(axis='x', alpha=0.3)

    loadings_df['PC2'].sort_values().plot(kind='barh', ax=axes[1])
    axes[1].set_title("PC2 Loadings")
    axes[1].set_xlabel("Loading")
    axes[1].grid(axis='x', alpha=0.3)

    plt.suptitle("Feature Loadings for PC1 and PC2", y=1.02)
    plt.tight_layout()
    plt.show()

    # === 3) PC3 vs PC4 scatter (also labeled) ===
    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        data=df,
        x='PC3', y='PC4',
        hue='DateLabel',
        hue_order=unique_labels,
        palette=palette_dict,
        s=100,
        edgecolor='black',
        alpha=0.8
    )

    for i in range(len(df)):
        plt.text(
            df.PC3.iloc[i] + 0.02,
            df.PC4.iloc[i],
            str(df.Num.iloc[i]),
            fontsize=8
        )

    plt.title("PCA of Shakespeare's Sonnets (PC3 vs PC4, Colored by Date)")
    plt.xlabel(f"PC3 ({pca.explained_variance_ratio_[2]:.2%} var)")
    plt.ylabel(f"PC4 ({pca.explained_variance_ratio_[3]:.2%} var)")
    plt.legend(title="Composition Date")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # === 4) Loadings for PC3 & PC4 ===
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    loadings_df['PC3'].sort_values().plot(kind='barh', ax=axes[0])
    axes[0].set_title("PC3 Loadings")
    axes[0].set_xlabel("Loading")
    axes[0].grid(axis='x', alpha=0.3)

    loadings_df['PC4'].sort_values().plot(kind='barh', ax=axes[1])
    axes[1].set_title("PC4 Loadings")
    axes[1].set_xlabel("Loading")
    axes[1].grid(axis='x', alpha=0.3)

    plt.suptitle("Feature Loadings for PC3 and PC4", y=1.02)
    plt.tight_layout()
    plt.show()

# --- EXECUTE ---
# Requires 'sonnets_map' and 'detector' from previous cells
if 'sonnets_map' in locals() and 'detector' in locals():
    df_temporal, corpus_temporal = extract_sonnet_features(sonnets_map, detector)
    analyze_and_visualize(df_temporal, corpus_temporal)
else:
    print("Error: 'sonnets_map' or 'detector' not found. Please run previous cells first.")

# %% [markdown] id="0kwpmkRfyvFj"
# ## Analyze All Plays

# %% id="tq6RVQV1VvWh"
import zipfile

# %% colab={"base_uri": "https://localhost:8080/", "height": 892, "referenced_widgets": ["548d38b22edd4c61ae304493adfd74c3", "13e94e9e75fd43a3b3571bb99d5d0891", "8d81aaf26c564eb6b85a6bcc09b2a789", "c65d7bd77cd9422887aba84b28403dd2", "4ed59885563249dabc23148fc6841dc4", "3ad5f72f9b174fd7adaf5d372cb4c6d6", "fb987e019b554b84a8df047a5346b8c9", "c6fc0a0fee7f4805acb034854cf62152", "ccf330574f7541bfb9b74eef5c93e99a", "298f082b33ed43d48a17bc20efcc3aa1", "5b9c75126269410686c3372d6a3b59ad", "45a02d9d3fd448f0844c1a6f7edce4a1", "78881fc95b984508b21f7fb9551cbfec", "98d298644cb8420bbf9fbae6e5173088", "a7b7c8bbdd4249fbb068fc0cb7a5e367", "af3a9272ccc14235a56b4555b19057e0", "6bdc9377b741418f9a0aafc947c66f14", "d6870e2466b04488babf170b87176343", "cc451db9f6414f1e97c78290303e8fe1", "2fa31e5f43544dd9ab7d124109ce5bb3", "acf5ff2de2a748b8b3bd3b3d29b00421", "7f747f89edb9430a9f9b99d1149ebac6", "247858c05bd6462fb66990be0631ddea", "558aaf1445f241c1b9372d4ce7041c0f", "7b54ec4b6e924bcdb085591fb13fdd33", "7ab652f227324aa1a1df600ebabe4554", "db866683267143d4a98274f158a8e45c", "97343816151f4943822c8dda0740e0d4", "7d752afc722d4805a034d7ed030a1e0b", "01f092080d86453bb3d2f6c2ed406977", "6fa698384e0849c190461086fbc2a923", "c79cfa09f5e44e31b9984f07474160db", "c16e6cbf75c24299b4e65eddea7415aa", "d88dad0dc0d242348664ac3913b64f50", "dd9088b590494c01aa386bbaf4cb0e4f", "52666e972a9a4b799d6ce28c53c4ba9e", "1ff05acdedf149d3a5588fe01762907e", "d06a7973542a4d18b1888901b046eedc", "5e4821b408fe40de99698d57699b4316", "a1e277f1db824a97bcf055b4a83b7ab5", "1c49466efeea4756bdb81be9075920d6", "96ab959df0ee410cb45b13112f850816", "669a70e0f1aa41e4b2223f37c9ae7e00", "787266ccb61148e8b57ed8ba7f2dcd36", "19ea66543c3b44d8b555716b88d5bf22", "2ab66d378cac4dca9b2857abd93e04ef", "92b81c9fed484b51aae5bbd1820791d5", "f32616ed08114c3aa41ef92779e54028", "7f172128a131445788be676bd8882bf0", "94492e9a344c4964b790650ddccf6b1b", "7486b0fa2f8a4471ac3aa50583e438c1", "d65e8ed3c12c4057bc4f0ca01a3bc7ee", "97cf99c1f5a5415683a34974879cce25", "e9aacb0d71994738a327d4f1e47918e5", "9f1f3a82c0104c4a8a4f427ecc7fab1f", "dde1b0bbced4424eb50bbaf5aefd444b", "884abf834bed43aea213708027bd436a", "806766c692354018b08a8d37a13b20e5", "ea85d379ad3b47f2b390cc444f0a8e5c", "655ec304fd5149efbe2397054f288467", "aecc06bf10e045909b97362d0eea2852", "04c22f5da6c44c229dc6c224e1d86e35", "78ec9facd34145308bbd5d21ceb193f2", "0e52a061e1ee46e29e30c421f1bd92fa", "c223eacdd1bf4d79b073de0e596bd7fa", "987eb1a70f7c4b9ea419c53c304dd4f4"]} id="b5GbdqcWAjmn" outputId="495b4878-7267-4ebb-d9c5-dd7a3d763f95"
# Cell 11 - Analyze All Plays

# Configuration
OUTPUT_DIR = "batch_output"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

print("Starting Batch Analysis...")

# --- 1. Define Analysis Function (Multi-Window) ---
def analyze_work_batch(title, lines):
    # print(f"  Analyzing {title}...")
    scores = []
    window_sizes = [10, 11, 14, 15, 18]

    clean_data = []
    boundaries = []
    current_clean_idx = 0
    current_speaker = "UNKNOWN"

    for line in lines:
        s_line = line.strip()
        if not s_line: continue

        if s_line.upper().startswith("ACT "):
            label = s_line.split('.')[0]
            boundaries.append((current_clean_idx, label, 'act'))
            # Do not continue; include line to maintain index alignment with classifier

        if s_line.upper().startswith("SCENE "):
            label = s_line.split('.')[0]
            boundaries.append((current_clean_idx, label, 'scene'))
            # Do not continue

        if s_line.isupper() and s_line.endswith(".") and len(s_line) < 20 and "SCENE" not in s_line and "ACT" not in s_line:
            current_speaker = s_line.strip(".")
            # Do not continue

        # MATCH CLASSIFIER FILTERING: Just non-empty lines
        # (Previously was len > 10, which caused index mismatch with classifier results)
        clean_data.append((s_line, current_speaker))
        current_clean_idx += 1

    clean_lines_text = [x[0] for x in clean_data]

    for i in range(len(clean_lines_text)):
        best_score_for_pos = 0.0
        for w_size in window_sizes:
            if i + w_size <= len(clean_lines_text):
                window = clean_lines_text[i : i + w_size]
                score = detector.score_window(window)
                if score > best_score_for_pos:
                    best_score_for_pos = score
        scores.append(best_score_for_pos)

    return scores, clean_data, boundaries

# --- 2. Helper for Downloads ---
def trigger_download(filepath):
    try:
        from google.colab import files
        files.download(filepath)
    except ImportError:
        print(f"  (Saved locally: {filepath})")

# --- 3. Main Batch Loop ---
results_text_buffer = []

# Sort works: Sonnets first, then alphabetical
work_titles = sorted(works.keys())
if "THE SONNETS" in work_titles:
    work_titles.remove("THE SONNETS")
    work_titles.insert(0, "THE SONNETS")

    work_titles.insert(0, "THE SONNETS")

# Batch Configuration
# Batch Configuration
NUM_BATCHES = 6
batch_size = int(np.ceil(len(work_titles) / NUM_BATCHES))
batches = [work_titles[i:i + batch_size] for i in range(0, len(work_titles), batch_size)]

print(f"Split {len(work_titles)} works into {len(batches)} batches.")

for batch_idx, batch_titles in enumerate(batches):
    print(f"\n=== Starting Batch {batch_idx + 1} of {len(batches)} ===")

    batch_signal_data = []
    batch_wavelet_data = []

    pbar = tqdm(batch_titles, desc=f"Batch {batch_idx + 1}", unit="play")
    for i, title in enumerate(pbar):
        pbar.set_description(f"Batch {batch_idx + 1}/{len(batches)} | Play {i + 1}/{len(batch_titles)}: {title}")

        lines = works[title]
        scores, clean_data, boundaries = analyze_work_batch(title, lines)

        if not scores:
            print(f"  No valid lines found in {title}")
            continue

        # --- A. Prepare Classifier Overlays ---
        sup_only, pu_only, overlap_indices = [], [], []
        try:
            # Filter for current play
            if 'sonnet_candidates_df' in globals():
                play_candidates = sonnet_candidates_df[sonnet_candidates_df['Play'] == title]

                # Global Thresholds (98th percentile)
                sup_threshold = np.percentile(sonnet_candidates_df['Supervised_Score'], 98)
                pu_threshold = np.percentile(sonnet_candidates_df['PU_Score'], 98)

                sup_indices = play_candidates[play_candidates['Supervised_Score'] >= sup_threshold]['Start_Line'].values
                pu_indices = play_candidates[play_candidates['PU_Score'] >= pu_threshold]['Start_Line'].values

                overlap_indices = np.intersect1d(sup_indices, pu_indices)
                sup_only = np.setdiff1d(sup_indices, overlap_indices)
                pu_only = np.setdiff1d(pu_indices, overlap_indices)
        except Exception as e:
            pass

        # --- B. Identify Top 20 Hits (NMS) ---
        scores_array = np.array(scores)
        sorted_indices = scores_array.argsort()[::-1]
        nms_indices = []
        taken_mask = np.zeros(len(scores), dtype=bool)
        nms_window = 18

        for idx in sorted_indices:
            if len(nms_indices) >= 20: break
            if taken_mask[idx]: continue
            nms_indices.append(idx)
            start = max(0, idx - nms_window + 1)
            end = min(len(scores), idx + nms_window)
            taken_mask[start:end] = True

        top_indices = nms_indices

        # --- C. Generate Individual Signal Plot ---
        plt.figure(figsize=(14, 8))
        plt.plot(scores, label=title, linewidth=0.5, color='gray', alpha=0.7)
        plt.title(f"Sonnet Detection Signal in {title}")
        plt.xlabel("Line Number (approx)")
        plt.ylabel("Sonnet Score")

        # Classifier Bars (Requested Colors & Positions)
        for idx in sup_only:
            plt.hlines(y=0.99, xmin=idx, xmax=idx+14, colors='lightsalmon', linewidth=2, label='Supervised (2%)' if 'Supervised (2%)' not in plt.gca().get_legend_handles_labels()[1] else "")
        for idx in pu_only:
            plt.hlines(y=0.98, xmin=idx, xmax=idx+14, colors='skyblue', linewidth=2, label='PU (2%)' if 'PU (2%)' not in plt.gca().get_legend_handles_labels()[1] else "")
        for idx in overlap_indices:
            plt.hlines(y=0.97, xmin=idx, xmax=idx+14, colors='palegreen', linewidth=2, label='Overlap (Top 2%)' if 'Overlap' not in plt.gca().get_legend_handles_labels()[1] else "")

        # Boundaries (Staggered Labels at Bottom)
        for idx, label, b_type in boundaries:
            if b_type == 'act':
                plt.axvline(x=idx, color='black', linewidth=0.4)
                plt.text(idx, 0.02, label, rotation=90, fontsize=6, verticalalignment='bottom') # Higher for Act
            elif b_type == 'scene':
                plt.axvline(x=idx, color='black', linewidth=0.2)
                plt.text(idx, 0.08, label, rotation=90, fontsize=6, verticalalignment='bottom') # Lower for Scene

        # Top 20 Markers
        for idx in top_indices:
            score = scores[idx]
            plt.plot(idx, score, 'k.', markersize=5)
            plt.text(idx, score, f"{idx}", fontsize=6, rotation=45, ha='left', va='bottom')

        plt.legend(loc='upper right')
        plt.ylim(0, 1.05)

        # Save Individual Signal Plot
        safe_title = "".join(x for x in title if x.isalnum() or x in " -_").strip().replace(" ", "_")
        sig_filename = f"{OUTPUT_DIR}/{safe_title}_signal.png"
        plt.savefig(sig_filename, dpi=150, bbox_inches='tight')
        plt.savefig(sig_filename.replace(".png", ".svg"), dpi=150, bbox_inches='tight')
        plt.close()
        # trigger_download(sig_filename) # Optional: Download every single one? Might be too many files.

        # Store data for combined plots
        batch_signal_data.append({
            'title': title,
            'scores': scores,
            'boundaries': boundaries,
            'sup': sup_only, 'pu': pu_only, 'overlap': overlap_indices,
            'top_indices': top_indices
        })

        # --- D. Generate Individual Wavelet Plot (Smoothed + Sig) ---
        sig = np.array(scores)
        sig = sig - np.mean(sig)
        t = np.arange(len(sig))
        scales = np.linspace(10, 600, 400)
        cwtmatr, freqs = pywt.cwt(sig, scales, 'cmor1.5-1.0')
        power = (np.abs(cwtmatr)) ** 2
        periods = 1.0 / freqs

        power_smooth = gaussian_filter(power, sigma=(2, 2))
        variance = np.var(sig)
        significance_threshold = variance * (stats.chi2.ppf(0.95, 2) / 2)

        # COI
        d_to_edge = np.minimum(t, (len(t) - 1) - t).astype(float)
        s_boundary = d_to_edge / np.sqrt(2)
        s_boundary_clipped = np.clip(s_boundary, scales.min(), scales.max())
        P_boundary = np.interp(s_boundary_clipped, scales, periods)

        plt.figure(figsize=(14, 8))
        levels = np.linspace(0, np.max(power), 20)
        CS = plt.contourf(t, periods, power_smooth, levels=levels, cmap='jet', extend='both')
        plt.colorbar(CS, label="Wavelet Power")
        plt.contour(t, periods, power_smooth, levels=[significance_threshold], colors='k', linewidths=1.5)
        plt.fill_between(t, P_boundary, periods.max(), color='white', alpha=0.5, hatch='x', zorder=2)
        plt.plot(t, P_boundary, color='k', linewidth=2, zorder=3)

        plt.title(f"Wavelet Power (Smoothed, Sig): {title}")
        plt.xlabel("Line Number")
        plt.ylabel("Period")
        plt.yscale('log')
        plt.yticks([20, 50, 100, 200, 300, 400, 500, 600], ['20', '50', '100', '200', '300', '400', '500', '600'])
        plt.ylim(periods.max(), periods.min())

        wav_filename = f"{OUTPUT_DIR}/{safe_title}_wavelet.png"
        plt.savefig(wav_filename, dpi=150, bbox_inches='tight')
        plt.savefig(wav_filename.replace(".png", ".svg"), dpi=150, bbox_inches='tight')
        plt.close()

        # Store for combined
        batch_wavelet_data.append({
            'title': title,
            'power': power_smooth,
            'periods': periods,
            't': t,
            'sig_thresh': significance_threshold,
            'P_boundary': P_boundary
        })

        # --- E. Collect Text Results ---
        results_text_buffer.append(f"********************************")
        results_text_buffer.append(f"{title}")
        results_text_buffer.append(f"********************************\n")

        for i, idx in enumerate(top_indices):
            score = scores[idx]
            results_text_buffer.append(f"Rank {i+1} | Score: {score:.4f} | Start Line: {idx}")
            results_text_buffer.append("-" * 30)

            last_speaker = None
            # Print 14 lines context
            for j in range(idx, min(idx+14, len(clean_data))):
                line_text, speaker = clean_data[j]
                prefix = ""
                if speaker != last_speaker:
                    prefix = f"[{speaker}] "
                    last_speaker = speaker
                results_text_buffer.append(f"{prefix}{line_text}")
            results_text_buffer.append("\n")
        results_text_buffer.append("\n" + "="*50 + "\n")

    # --- 4. Generate Combined Plots (PER BATCH) ---
    print(f"Generating Combined Plots for Batch {batch_idx + 1}...")
    num_works = len(batch_signal_data)
    if num_works == 0: continue

    # A. Vertical Stacked Signal
    fig, axes = plt.subplots(num_works, 1, figsize=(12, 2 * num_works), sharex=False)
    if num_works == 1: axes = [axes]

    for i, data in enumerate(batch_signal_data):
        ax = axes[i]
        ax.plot(data['scores'], linewidth=0.5, color='gray')
        ax.set_title(data['title'], fontsize=10, loc='left')
        ax.set_ylim(0, 1.25)
        ax.set_ylabel("Score", fontsize=8)

        # Add classifiers (Updated Colors/Pos)
        for idx in data['sup']: ax.hlines(y=0.99, xmin=idx, xmax=idx+14, colors='lightsalmon', linewidth=2)
        for idx in data['pu']: ax.hlines(y=0.98, xmin=idx, xmax=idx+14, colors='skyblue', linewidth=2)
        for idx in data['overlap']: ax.hlines(y=0.97, xmin=idx, xmax=idx+14, colors='palegreen', linewidth=2)

        # Add Boundaries (FIXED: Now included in Combined Plots)
        for idx, label, b_type in data['boundaries']:
            if b_type == 'act':
                ax.axvline(x=idx, color='black', linewidth=0.4)
                ax.text(idx, 0.02, label, rotation=90, fontsize=4, verticalalignment='bottom')
            elif b_type == 'scene':
                ax.axvline(x=idx, color='black', linewidth=0.2)
                ax.text(idx, 0.08, label, rotation=90, fontsize=4, verticalalignment='bottom')

        # Add top markers with labels
        for idx in data['top_indices']:
            score = data['scores'][idx]
            ax.plot(idx, score, 'k.', markersize=3)

    plt.tight_layout()
    vert_sig_filename = f"{OUTPUT_DIR}/batch_{batch_idx+1}_combined_signal_vertical.png"
    plt.savefig(vert_sig_filename, dpi=150, bbox_inches='tight')
    plt.savefig(vert_sig_filename.replace(".png", ".svg"), dpi=150, bbox_inches='tight')
    plt.close('all')
    # trigger_download(vert_sig_filename)
    # trigger_download(vert_sig_filename.replace(".png", ".svg"))

    # B. Horizontal Wide Signal (Side by Side)
    # This will be huge. Let's make it 1 row, N columns.
    fig, axes = plt.subplots(1, num_works, figsize=(10 * num_works, 6), sharey=True)
    if num_works == 1: axes = [axes]

    for i, data in enumerate(batch_signal_data):
        ax = axes[i]
        ax.plot(data['scores'], linewidth=0.5, color='gray')
        ax.set_title(data['title'], fontsize=12)
        ax.set_ylim(0, 1.05)

        for idx in data['sup']: ax.hlines(y=0.99, xmin=idx, xmax=idx+14, colors='lightsalmon', linewidth=2)
        for idx in data['pu']: ax.hlines(y=0.98, xmin=idx, xmax=idx+14, colors='skyblue', linewidth=2)
        for idx in data['overlap']: ax.hlines(y=0.97, xmin=idx, xmax=idx+14, colors='palegreen', linewidth=2)

        # Add Boundaries (FIXED)
        for idx, label, b_type in data['boundaries']:
            if b_type == 'act':
                ax.axvline(x=idx, color='black', linewidth=0.4)
                ax.text(idx, 0.02, label, rotation=90, fontsize=4, verticalalignment='bottom')
            elif b_type == 'scene':
                ax.axvline(x=idx, color='black', linewidth=0.2)
                ax.text(idx, 0.08, label, rotation=90, fontsize=4, verticalalignment='bottom')

        for idx in data['top_indices']:
            score = data['scores'][idx]
            ax.plot(idx, score, 'k.', markersize=3)
            ax.text(idx, score, f"{idx}", fontsize=6, rotation=45, ha='left', va='bottom')

    plt.tight_layout()
    horz_sig_filename = f"{OUTPUT_DIR}/batch_{batch_idx+1}_combined_signal_horizontal.png"
    plt.savefig(horz_sig_filename, dpi=150, bbox_inches='tight') # Keep 150 for wide plot to avoid massive size
    plt.savefig(horz_sig_filename.replace(".png", ".svg"), dpi=150, bbox_inches='tight')
    plt.savefig(horz_sig_filename.replace(".png", ".svg"), dpi=150, bbox_inches='tight')
    plt.close('all')
    # trigger_download(horz_sig_filename)
    # trigger_download(horz_sig_filename.replace(".png", ".svg"))


    # C. Vertical Stacked Wavelet
    fig, axes = plt.subplots(num_works, 1, figsize=(12, 3 * num_works), sharex=False)
    if num_works == 1: axes = [axes]

    for i, data in enumerate(batch_wavelet_data):
        ax = axes[i]
        levels = np.linspace(0, np.max(data['power']), 20)
        # Use contourf
        ax.contourf(data['t'], data['periods'], data['power'], levels=levels, cmap='jet', extend='both')
        # Sig lines
        ax.contour(data['t'], data['periods'], data['power'], levels=[data['sig_thresh']], colors='k', linewidths=1)
        # COI
        ax.fill_between(data['t'], data['P_boundary'], data['periods'].max(), color='white', alpha=0.5, hatch='x')

        ax.set_title(data['title'], fontsize=10, loc='left')
        ax.set_yscale('log')
        ax.set_ylim(data['periods'].max(), data['periods'].min())
        ax.set_yticks([20, 100, 400])
        ax.set_yticklabels(['20', '100', '400'])

    plt.tight_layout()
    vert_wav_filename = f"{OUTPUT_DIR}/batch_{batch_idx+1}_combined_wavelet_vertical.png"
    plt.savefig(vert_wav_filename, dpi=150, bbox_inches='tight')
    plt.savefig(vert_wav_filename.replace(".png", ".svg"), dpi=150, bbox_inches='tight')
    plt.close('all')
    # trigger_download(vert_wav_filename)
    # trigger_download(vert_wav_filename.replace(".png", ".svg"))

    # D. Horizontal Wide Wavelet
    fig, axes = plt.subplots(1, num_works, figsize=(10 * num_works, 6), sharey=True)
    if num_works == 1: axes = [axes]

    for i, data in enumerate(batch_wavelet_data):
        ax = axes[i]
        levels = np.linspace(0, np.max(data['power']), 20)
        ax.contourf(data['t'], data['periods'], data['power'], levels=levels, cmap='jet', extend='both')
        ax.contour(data['t'], data['periods'], data['power'], levels=[data['sig_thresh']], colors='k', linewidths=1)
        ax.fill_between(data['t'], data['P_boundary'], data['periods'].max(), color='white', alpha=0.5, hatch='x')

        ax.set_title(data['title'], fontsize=12)
        ax.set_yscale('log')
        ax.set_ylim(data['periods'].max(), data['periods'].min())

    plt.tight_layout()
    horz_wav_filename = f"{OUTPUT_DIR}/batch_{batch_idx+1}_combined_wavelet_horizontal.png"
    plt.savefig(horz_wav_filename, dpi=150, bbox_inches='tight')
    plt.savefig(horz_wav_filename.replace(".png", ".svg"), dpi=150, bbox_inches='tight')
    plt.savefig(horz_wav_filename.replace(".png", ".svg"), dpi=150, bbox_inches='tight')
    plt.close('all')
    # trigger_download(horz_wav_filename)
    # trigger_download(horz_wav_filename.replace(".png", ".svg"))

    # Save Text Report Incrementally (Append mode logic or overwrite with full buffer)
    # Since results_text_buffer is cumulative, we can just overwrite the file each time.
    text_report_filename = f"{OUTPUT_DIR}/all_plays_top_20_hits.txt"
    with open(text_report_filename, "w", encoding="utf-8") as f:
        f.write("\n".join(results_text_buffer))
    # trigger_download(text_report_filename)

    # --- Zip and Download Batch Results ---
    zip_filename = f"{OUTPUT_DIR}/batch_{batch_idx+1}_results.zip"
    with zipfile.ZipFile(zip_filename, 'w') as zipf:
        # Add plots
        zipf.write(vert_sig_filename, os.path.basename(vert_sig_filename))
        zipf.write(vert_sig_filename.replace(".png", ".svg"), os.path.basename(vert_sig_filename.replace(".png", ".svg")))
        zipf.write(horz_sig_filename, os.path.basename(horz_sig_filename))
        zipf.write(horz_sig_filename.replace(".png", ".svg"), os.path.basename(horz_sig_filename.replace(".png", ".svg")))
        zipf.write(vert_wav_filename, os.path.basename(vert_wav_filename))
        zipf.write(vert_wav_filename.replace(".png", ".svg"), os.path.basename(vert_wav_filename.replace(".png", ".svg")))
        zipf.write(horz_wav_filename, os.path.basename(horz_wav_filename))
        zipf.write(horz_wav_filename.replace(".png", ".svg"), os.path.basename(horz_wav_filename.replace(".png", ".svg")))
        # Add text report
        zipf.write(text_report_filename, os.path.basename(text_report_filename))

    print(f"Zipped batch {batch_idx+1} results to {zip_filename}")
    trigger_download(zip_filename)

print(f"Batch Analysis Complete. All outputs in {OUTPUT_DIR}/")

# Download Individual Files (Optional - uncomment if desired, but might be 80+ files)
# for f in os.listdir(OUTPUT_DIR):
#     if "_signal.png" in f or "_wavelet.png" in f:
#         trigger_download(os.path.join(OUTPUT_DIR, f))


# %% [markdown] id="TxFZqoxAy2Vm"
# ## Metrics of Popularity

# %% colab={"base_uri": "https://localhost:8080/"} id="S3Wmc0MTDVcg" outputId="bf867059-eb0d-43b3-bafb-17eab90f7f00"
# Cell 12

# Data gathered from external sources (Wikipedia, Open Syllabus, Goodreads)
# Note: These are approximate values based on available aggregated data.
# Syllabus Score is estimated based on ranking (100 = Top 1, 0 = >100)
external_metrics = {
    "THE TRAGEDY OF ROMEO AND JULIET": {"Wiki_Views": 5000000, "Syllabus_Score": 95, "GR_Rating": 3.74, "GR_Count": 2281937},
    "THE TRAGEDY OF HAMLET, PRINCE OF DENMARK": {"Wiki_Views": 4500000, "Syllabus_Score": 100, "GR_Rating": 4.03, "GR_Count": 817051},
    "THE TRAGEDY OF MACBETH": {"Wiki_Views": 4000000, "Syllabus_Score": 98, "GR_Rating": 3.89, "GR_Count": 759866},
    "A MIDSUMMER NIGHT’S DREAM": {"Wiki_Views": 3000000, "Syllabus_Score": 90, "GR_Rating": 3.94, "GR_Count": 483268},
    "THE TRAGEDY OF OTHELLO, THE MOOR OF VENICE": {"Wiki_Views": 3200000, "Syllabus_Score": 92, "GR_Rating": 3.89, "GR_Count": 345059},
    "MUCH ADO ABOUT NOTHING": {"Wiki_Views": 2500000, "Syllabus_Score": 85, "GR_Rating": 4.06, "GR_Count": 227703},
    "THE TRAGEDY OF KING LEAR": {"Wiki_Views": 2800000, "Syllabus_Score": 88, "GR_Rating": 3.91, "GR_Count": 196817},
    "THE TEMPEST": {"Wiki_Views": 2200000, "Syllabus_Score": 82, "GR_Rating": 3.78, "GR_Count": 187722},
    "THE TRAGEDY OF JULIUS CAESAR": {"Wiki_Views": 2400000, "Syllabus_Score": 91, "GR_Rating": 3.71, "GR_Count": 183809},
    "THE MERCHANT OF VENICE": {"Wiki_Views": 2100000, "Syllabus_Score": 80, "GR_Rating": 3.78, "GR_Count": 170795},
    "TWELFTH NIGHT; OR, WHAT YOU WILL": {"Wiki_Views": 2000000, "Syllabus_Score": 83, "GR_Rating": 3.96, "GR_Count": 166627},
    "THE TAMING OF THE SHREW": {"Wiki_Views": 1800000, "Syllabus_Score": 75, "GR_Rating": 3.80, "GR_Count": 160352},
    "AS YOU LIKE IT": {"Wiki_Views": 1500000, "Syllabus_Score": 70, "GR_Rating": 3.80, "GR_Count": 76640},
    "KING RICHARD THE THIRD": {"Wiki_Views": 1200000, "Syllabus_Score": 65, "GR_Rating": 3.90, "GR_Count": 48178},
    "THE LIFE OF KING HENRY THE FIFTH": {"Wiki_Views": 1100000, "Syllabus_Score": 60, "GR_Rating": 3.80, "GR_Count": 45795},
    "THE TRAGEDY OF ANTONY AND CLEOPATRA": {"Wiki_Views": 900000, "Syllabus_Score": 55, "GR_Rating": 3.70, "GR_Count": 34654},
    "THE WINTER’S TALE": {"Wiki_Views": 800000, "Syllabus_Score": 50, "GR_Rating": 3.70, "GR_Count": 29703},
    "THE COMEDY OF ERRORS": {"Wiki_Views": 700000, "Syllabus_Score": 45, "GR_Rating": 3.70, "GR_Count": 27544},
    "THE TRAGEDY OF TITUS ANDRONICUS": {"Wiki_Views": 750000, "Syllabus_Score": 40, "GR_Rating": 3.70, "GR_Count": 27331},
    "THE FIRST PART OF KING HENRY THE FOURTH": {"Wiki_Views": 850000, "Syllabus_Score": 68, "GR_Rating": 3.80, "GR_Count": 27197},
    "MEASURE FOR MEASURE": {"Wiki_Views": 600000, "Syllabus_Score": 48, "GR_Rating": 3.60, "GR_Count": 25446},
    "KING RICHARD THE SECOND": {"Wiki_Views": 650000, "Syllabus_Score": 52, "GR_Rating": 3.80, "GR_Count": 22074},
    "THE SECOND PART OF KING HENRY THE FOURTH": {"Wiki_Views": 500000, "Syllabus_Score": 42, "GR_Rating": 3.80, "GR_Count": 18339},
    "ALL’S WELL THAT ENDS WELL": {"Wiki_Views": 400000, "Syllabus_Score": 35, "GR_Rating": 3.50, "GR_Count": 16603},
    "THE TRAGEDY OF CORIOLANUS": {"Wiki_Views": 450000, "Syllabus_Score": 38, "GR_Rating": 3.70, "GR_Count": 15721},
    "THE SONNETS": {"Wiki_Views": 2000000, "Syllabus_Score": 85, "GR_Rating": 4.20, "GR_Count": 50000},
    # Estimates for less popular works
    "CYMBELINE": {"Wiki_Views": 300000, "Syllabus_Score": 30, "GR_Rating": 3.6, "GR_Count": 10000},
    "THE FIRST PART OF HENRY THE SIXTH": {"Wiki_Views": 200000, "Syllabus_Score": 20, "GR_Rating": 3.5, "GR_Count": 8000},
    "THE SECOND PART OF KING HENRY THE SIXTH": {"Wiki_Views": 200000, "Syllabus_Score": 20, "GR_Rating": 3.5, "GR_Count": 7500},
    "THE THIRD PART OF KING HENRY THE SIXTH": {"Wiki_Views": 200000, "Syllabus_Score": 20, "GR_Rating": 3.5, "GR_Count": 7000},
    "KING HENRY THE EIGHTH": {"Wiki_Views": 250000, "Syllabus_Score": 25, "GR_Rating": 3.4, "GR_Count": 9000},
    "THE LIFE AND DEATH OF KING JOHN": {"Wiki_Views": 280000, "Syllabus_Score": 28, "GR_Rating": 3.6, "GR_Count": 11000},
    "LOVE’S LABOUR’S LOST": {"Wiki_Views": 350000, "Syllabus_Score": 32, "GR_Rating": 3.6, "GR_Count": 14000},
    "THE MERRY WIVES OF WINDSOR": {"Wiki_Views": 400000, "Syllabus_Score": 35, "GR_Rating": 3.6, "GR_Count": 15000},
    "PERICLES, PRINCE OF TYRE": {"Wiki_Views": 200000, "Syllabus_Score": 15, "GR_Rating": 3.5, "GR_Count": 6000},
    "THE LIFE OF TIMON OF ATHENS": {"Wiki_Views": 180000, "Syllabus_Score": 15, "GR_Rating": 3.4, "GR_Count": 5000},
    "TROILUS AND CRESSIDA": {"Wiki_Views": 300000, "Syllabus_Score": 40, "GR_Rating": 3.5, "GR_Count": 12000},
    "THE TWO GENTLEMEN OF VERONA": {"Wiki_Views": 320000, "Syllabus_Score": 30, "GR_Rating": 3.6, "GR_Count": 13000},
    "THE TWO NOBLE KINSMEN": {"Wiki_Views": 100000, "Syllabus_Score": 10, "GR_Rating": 3.3, "GR_Count": 3000},
    "A LOVER’S COMPLAINT": {"Wiki_Views": 50000, "Syllabus_Score": 5, "GR_Rating": 3.5, "GR_Count": 1000},
    "THE PASSIONATE PILGRIM": {"Wiki_Views": 40000, "Syllabus_Score": 5, "GR_Rating": 3.4, "GR_Count": 800},
    "THE PHOENIX AND THE TURTLE": {"Wiki_Views": 60000, "Syllabus_Score": 10, "GR_Rating": 3.8, "GR_Count": 1500},
    "THE RAPE OF LUCRECE": {"Wiki_Views": 150000, "Syllabus_Score": 20, "GR_Rating": 3.7, "GR_Count": 5000},
    "VENUS AND ADONIS": {"Wiki_Views": 200000, "Syllabus_Score": 25, "GR_Rating": 3.6, "GR_Count": 8000}
}

# Create DataFrame
df_metrics = pd.DataFrame.from_dict(external_metrics, orient='index')
df_metrics.index.name = 'Work'
df_metrics.reset_index(inplace=True)

# Sort by Goodreads Count (Popularity)
df_metrics.sort_values(by='GR_Count', ascending=False, inplace=True)

# Display
print("External Metrics Table (Top 20 by Popularity):")
print(df_metrics.head(20).to_string(index=False))

# Optional: Save to CSV
df_metrics.to_csv("shakespeare_external_metrics.csv", index=False)
print("\nSaved metrics to shakespeare_external_metrics.csv")


# %% colab={"base_uri": "https://localhost:8080/", "height": 549} id="ko1JoW8lnhGW" outputId="7d00a5ed-0e46-4170-b3d6-e25d356537c6"
def grouped_bar_chart_5_dualaxis(values_list, labels, title, ylabel_left, ylabel_right):
    """
    values_list: list of 5 lists, one per work (each length = number of bins)
    labels: x-axis bin labels
    """

    works = [
        "Sonnets",
        "Midsummer Night's Dream",
        "Romeo and Juliet",
        "Julius Caesar",
        "Harry Potter"
    ]

    n_bins = len(labels)
    x = np.arange(n_bins)

    n_works = 5
    width = 0.15

    # color map
    cmap = plt.cm.get_cmap("viridis", n_works)
    colors = [cmap(i) for i in range(n_works)]

    # offsets for grouped bars
    offsets = (np.arange(n_works) - (n_works - 1) / 2) * width

    # -----------------------
    # Figure and twin axis
    # -----------------------
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()

    # -----------------------
    # Plot bins 1–4 on ax1
    # -----------------------
    for i, (vals, work) in enumerate(zip(values_list, works)):
        ax1.bar(
            x[:-1] + offsets[i],     # first n_bins-1 bins
            vals[:-1],
            width,
            label=work,
            color=colors[i],
            zorder=3
        )

    # -----------------------
    # Plot last bin (5+) on ax2
    # -----------------------
    for i, (vals, work) in enumerate(zip(values_list, works)):
        ax2.bar(
            x[-1] + offsets[i],      # last bin only
            vals[-1],
            width,
            color=colors[i],
            alpha=0.8,
            zorder=2
        )

    # -----------------------
    # Vertical divider
    # -----------------------
    # Between bin index 3 (label "4") and bin index 4 (label "5+")
    divider_x = (x[-2] + x[-1]) / 2
    ax1.axvline(divider_x, color="black", linewidth=1.5)

    # -----------------------
    # Axes labels and ticks
    # -----------------------
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel(ylabel_left)
    ax2.set_ylabel(ylabel_right)

    ax1.set_title(title)

    # Put legend on ax1
    ax1.legend(loc="upper center")

    plt.tight_layout()
    plt.show()

grouped_bar_chart_5_dualaxis(
    [frac_sonnets, frac_mnd, frac_rj, frac_jc, frac_hp],
    bin_labels,
    "Fraction of Word Usage (1–4 on left axis, 5+ on right axis)",
    "Fraction (bins 1–4)",
    "Fraction (bin 5+)"
)



# %% colab={"base_uri": "https://localhost:8080/", "height": 71} id="umSs67kJkwt7" outputId="d305a888-7a85-4c6f-9e3a-30ab00a3306a"
# Cell 8: Download Results (Colab Compatible)
from google.colab import files

def save_and_download(dataframe, filename, title):
    # Write to the current working directory (Colab root)
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"{title}\n")
        f.write("=" * len(title) + "\n\n")

        current_play = None
        for _, row in dataframe.iterrows():
            if row['play'] != current_play:
                f.write(f"\n--- {row['play']} ---\n")
                current_play = row['play']

            f.write(f"Line {row['line']} | Score: {row['sup']:.4f} (Sup) / {row['pu']:.4f} (PU)\n")
            f.write(f"{row['text']}\n")
            f.write("-" * 20 + "\n")

    print(f"Downloading: {filename}")
    files.download(filename)

# Trigger downloads for the dataframes already in memory
save_and_download(top_sup_df, "sonnet_candidates_supervised.txt", "Top 5% Candidates - Supervised Classifier")
save_and_download(top_pu_df, "sonnet_candidates_pu.txt", "Top 5% Candidates - PU Classifier")
save_and_download(intersection_df, "sonnet_candidates_intersection.txt", "Top 5% Candidates - INTERSECTION (Both Methods)")

# %% [markdown] id="pD437fdsy9gB"
# # Part 4 - Distinctness

# %% [markdown] id="TRXqhOdMzm_6"
# ## Sonnet Finder V2

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} id="9--ZjrqCw9jO" outputId="084bb1e7-b60c-45b4-cc70-2d59c1421dd0"
# Sonnet Finder V2

# Work to analyze in depth (Signal Plot + Text Output)
WORK_TO_ANALYZE = "THE TRAGEDY OF ROMEO AND JULIET"

# Works to compare in the KDE Plot
COMPARISON_WORKS = ["A MIDSUMMER NIGHT’S DREAM", "TROILUS AND CRESSIDA", "THE TRAGEDY OF ROMEO AND JULIET"]
# ---------------------

# Initialize Neural G2P
neural_g2p = G2p()
print("Neural G2P Model Loaded.")

def get_neural_stress_pattern(line):
    phonemes = neural_g2p(line)
    pattern = []
    for p in phonemes:
        if p[-1].isdigit():
            stress = int(p[-1])
            pattern.append(1 if stress > 0 else 0)
    return pattern

class AdvancedSonnetDetector(SonnetDetector):
    def score_window(self, lines):
        if len(lines) != 14: return 0.0

        # 1. Neural Meter Score
        meter_scores = []
        syllable_scores = []

        for line in lines:
            pat = get_neural_stress_pattern(line)
            if len(pat) == 11 and pat[-1] == 0:
                meter_scores.append(is_iambic(pat[:-1]))
                syllable_scores.append(1.0)
            else:
                meter_scores.append(is_iambic(pat))
                dist = abs(len(pat) - 10)
                syllable_scores.append(max(0, 1.0 - (dist * 0.1)))

        avg_meter = np.mean(meter_scores)
        avg_syllable = np.mean(syllable_scores)

        # 2. Rhyme (Reuse existing logic)
        last_words = [re.findall(r"\w+", l)[-1] if re.findall(r"\w+", l) else "" for l in lines]
        quatrain_scores = []
        for i in range(0, 12, 4):
            w = last_words[i:i+4]
            if self.check_rhyme(w[0], w[2]) and self.check_rhyme(w[1], w[3]): quatrain_scores.append(1.0)
            elif self.check_rhyme(w[0], w[3]) and self.check_rhyme(w[1], w[2]): quatrain_scores.append(0.8)
            elif self.check_rhyme(w[0], w[1]) and self.check_rhyme(w[2], w[3]): quatrain_scores.append(0.6)
            else: quatrain_scores.append(0.2 if (self.check_rhyme(w[0], w[2]) or self.check_rhyme(w[1], w[3])) else 0.0)
        rhyme_val = np.mean(quatrain_scores)
        couplet_val = 1.0 if self.check_rhyme(last_words[12], last_words[13]) else 0.0

        return (avg_meter * 0.25) + (avg_syllable * 0.15) + (rhyme_val * 0.35) + (couplet_val * 0.25)

adv_detector = AdvancedSonnetDetector()
print("Advanced Sonnet Detector Ready.")

# --- 1. Deep Dive Analysis (Signal Plot & Text) ---

def analyze_work_advanced(title, lines, detector):
    print(f"Analyzing {title} with Neural Detector...")
    scores = []
    window_size = 14
    clean_data = []
    boundaries = []
    current_clean_idx = 0
    current_speaker = "UNKNOWN"

    for line in lines:
        s_line = line.strip()
        if not s_line: continue
        if s_line.upper().startswith("ACT "):
            label = s_line.split('.')[0]
            boundaries.append((current_clean_idx, label, 'act'))
            continue
        if s_line.upper().startswith("SCENE "):
            label = s_line.split('.')[0]
            boundaries.append((current_clean_idx, label, 'scene'))
            continue
        if s_line.isupper() and s_line.endswith(".") and len(s_line) < 20 and "SCENE" not in s_line and "ACT" not in s_line:
            current_speaker = s_line.strip(".")
            continue
        if len(s_line) > 10:
            clean_data.append((s_line, current_speaker))
            current_clean_idx += 1

    clean_lines_text = [x[0] for x in clean_data]
    for i in range(len(clean_lines_text) - window_size):
        window = clean_lines_text[i : i + window_size]
        score = detector.score_window(window)
        scores.append(score)
    return scores, clean_data, boundaries

# Run Analysis on Target Work
lines = works[WORK_TO_ANALYZE]
scores, clean_data, boundaries = analyze_work_advanced(WORK_TO_ANALYZE, lines, adv_detector)

fig, ax = plt.subplots(figsize=(14, 8))

# Create a set of line segments so that we can color them individually
# Points are (x, y)
points = np.array([np.arange(len(scores)), scores]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)

# Create a continuous norm to map from data points to colors
norm = plt.Normalize(0, 1.0)
lc = LineCollection(segments, cmap='Greys', norm=norm)

# Set the values used for colormapping
lc.set_array(np.array(scores))
lc.set_linewidth(0.1)

line = ax.add_collection(lc)
ax.set_xlim(0, len(scores))
ax.set_ylim(0, 1.15)

ax.set_title(f"Neural Sonnet Detection Signal in {WORK_TO_ANALYZE}")
ax.set_xlabel("Line Number (approx)")
ax.set_ylabel("Sonnet Score")

for idx, label, b_type in boundaries:
    if b_type == 'act':
        ax.axvline(x=idx, color='black', linewidth=0.5)
        ax.text(idx, 1.02, label, rotation=90, fontsize=6, verticalalignment='bottom')
    elif b_type == 'scene':
        ax.axvline(x=idx, color='black', linewidth=0.1)
        ax.text(idx, 1.01, label, rotation=90, fontsize=6, verticalalignment='bottom')

# Top Markers (NMS)
scores_array = np.array(scores)
sorted_indices = scores_array.argsort()[::-1]
nms_indices = []
taken_mask = np.zeros(len(scores), dtype=bool)
for idx in sorted_indices:
    if len(nms_indices) >= 20: break
    if taken_mask[idx]: continue
    nms_indices.append(idx)
    taken_mask[max(0, idx-14):min(len(scores), idx+14)] = True

for idx in nms_indices:
    ax.plot(idx, scores[idx], 'k.', markersize=5)
    ax.text(idx, scores[idx], f"{idx}", fontsize=6, rotation=45, ha='left', va='bottom')

# Legend proxy (since LineCollection doesn't auto-legend well)
legend_elements = [Line2D([0], [0], color='black', lw=1, label=WORK_TO_ANALYZE)]
ax.legend(handles=legend_elements)

plt.show()

# Text Output
print(f"\nTop 10 Scoring Segments in {WORK_TO_ANALYZE} (Neural Detector):\n" + "="*40)
for i, idx in enumerate(nms_indices[:10]):
    print(f"\nRank {i+1} | Score: {scores[idx]:.4f} | Start Line: {idx}")
    print("-" * 20)
    last_speaker = None
    for j in range(idx, idx+14):
        line_text, speaker = clean_data[j]
        prefix = ""
        if speaker != last_speaker:
            prefix = f"[{speaker}] "
            last_speaker = speaker
        print(f"{prefix}{line_text}")

# --- 2. KDE Distribution Plot ---
print(f"\nComparing Score Distributions for: {COMPARISON_WORKS}")
plt.figure(figsize=(12, 6))

# Baseline: The Sonnets
sonnet_scores = []
current_sonnet = []
for line in works["THE SONNETS"]:
    s = line.strip()
    if not s: continue
    if s.isdigit() or re.match(r'^[XVI]+(\.)?$', s):
        if len(current_sonnet) >= 14:
            sonnet_scores.append(adv_detector.score_window(current_sonnet[:14]))
        current_sonnet = []
    else:
        current_sonnet.append(s)
if len(current_sonnet) >= 14: sonnet_scores.append(adv_detector.score_window(current_sonnet[:14]))

sns.kdeplot(sonnet_scores, label="THE SONNETS (Baseline)", fill=True, alpha=0.3, linewidth=2)

# Comparison Works
for title in COMPARISON_WORKS:
    # Use the advanced analyzer for consistency
    s, _, _ = analyze_work_advanced(title, works[title], adv_detector)
    if s:
        sns.kdeplot(s, label=title, fill=True, alpha=0.2)

plt.title("Distribution of Sonnet Scores (Neural Detector)")
plt.xlabel("Sonnet Score")
plt.ylabel("Density")
plt.legend()
plt.xlim(0, 1.1)
plt.grid(True, alpha=0.3)
plt.show()

# --- 3. Ranked Graph (If applicable) ---
if WORK_TO_ANALYZE == "THE SONNETS":
    print("\nGenerating Ranked Sonnet Graph...")
    sorted_s_scores = sorted(sonnet_scores)
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(sorted_s_scores)), sorted_s_scores, color='skyblue')
    plt.title("All Sonnets Ranked by Neural Score")
    plt.xlabel("Rank")
    plt.ylabel("Score")
    plt.show()

# %% [markdown] id="JitvqpG2zcKg"
# ## Semantic Coherence

# %% colab={"base_uri": "https://localhost:8080/", "height": 741, "referenced_widgets": ["55837779302e42369833b1e849992803", "6396bc35f260435ca33bb45460674f2d", "31b57a7d15574fa295315738abae0dbd", "66b35fcf363b4fd8b414fb7a6f4cb0cb", "4a31466fb5f4472999ae3864b08585d3", "7d70141aede84b7b8ca02a93582d985f", "3c5e533717da4937be9f61cc15cce8e2", "4c9946aa01944a4b86156636eb83a728", "9a508338c1b24869be2e9ea8afbb6b1a", "4e709d5bf8f94dd1afe03b28cf77465f", "98f21ccf42d44f3bb5dd0555921bf1ef", "a7239dd56f084e1b815f9009660637fe", "1cdd25cca3714211a3243dfbad500ee6", "e47c4a3467ea42f88f5f91f2ad7ab1bc", "d300ec992d1c4d1497caa54c803820dd", "543a324bb2fa4c1287e10f3e560b2179", "b367c4aa0b734aa3953b7b20eb8cdfe4", "23f6cd8070a5403b9bb97f5c3659e785", "e1e32a63444246c389161dc64d97f4c8", "2b5cb04bf3b142989ba5624dad49fea3", "8d54e074562b4a94a1ac2b186f5f46db", "265a8213cdd14fa4a4c357a7e107e0c8", "53828c2a65dc41dd8591b770f212ff06", "36aeb1ab3c9144b4ae6d2470a62872ab", "3ad8ac4d1bbf4abb86852787cf6f3d7c", "ce483632332443b586c2059c503610f8", "e37691833b134031b32fcd186c62ca02", "c14e41a37b384ceea91e865b57404d2b", "0f020b2d6a6e4956bf9a9bb36487641d", "a2c8f7e24cfb49ee8732382b85aed3e7", "d228e72148e14db28804e9ca3c712f36", "a3de63defe0947c0969e725f837a1115", "028d664ed20c4dc583a72d076c228c43", "b5a4746ed4f64dc1b7de6b52615015f7", "a2c30af2f4614430803cacc9740002e4", "9eb2134518d94c2ea366eb7046fb852e", "f8d40c7750114efaba6fd9903d023fcb", "e7ba9bb0036c4c49ac65bcb2773a2b0b", "5b449f420fe94c999e34435fe8f18fa1", "68e413e7c51d4a5d9f4d3966ebb4f0c0", "6f4099529f4545e08939363ac4c3d29f", "b6b0f9ba75394c01b5a056677d682379", "c7bdae967e954533aa84dfa23833bc1f", "a5f5c09859224111b1e4adf787aa4282", "31d145ac91c14fdca67a0672446e26f7", "3e7071a8ad264306808e343e5ae490e9", "652d28dfc08841549d1b2160a85457e5", "8d8dc4c8266f434786a5282f2c7840a1", "7f8de6a818e14a119c5871d893dbd22d", "0597afedb752493db2ff29a431df02c1", "ed36cf72904a4d77ae23879064e51d0b", "38f426f13b1540c584e5a580251d8cb6", "a6ff011c7664473787ab71da64cde0d4", "2d7841e854f14ae2a42450b1703ded06", "3da784ff27e248198d0e70ae053a0c56"]} id="7ottDuO1zMLt" outputId="4f245c99-b20f-4be9-97a4-53f9bd4b011f"
# Semantic Coherence

# --- CONFIGURATION ---
# Works to analyze for semantic coherence
COHERENCE_WORKS = ["A MIDSUMMER NIGHT’S DREAM", "TROILUS AND CRESSIDA"]
TOP_N_CANDIDATES = 25
# ---------------------

# 1. Load BERT
print("Loading BERT Model...")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
print("BERT Loaded.")

def get_line_embedding(line):
    inputs = tokenizer(line, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].numpy()

def score_semantic_coherence(lines):
    embeddings = [get_line_embedding(l) for l in lines if l.strip()]
    if len(embeddings) < 2: return 0.0
    embeddings = np.vstack(embeddings)
    sim_matrix = cosine_similarity(embeddings)
    upper_tri = sim_matrix[np.triu_indices(len(sim_matrix), k=1)]
    return np.mean(upper_tri)

# 2. Gather Data & Score
coherence_data = []

# A. The Sonnets (Baseline)
sonnets_map_bert = {}
curr = []
c_num = 0
for l in works["THE SONNETS"]:
    s = l.strip()
    if not s: continue
    if s.isdigit() or re.match(r'^[XVI]+(\.)?$', s):
        if curr: sonnets_map_bert[c_num] = curr
        curr = []
        try: c_num = int(s.strip('.'))
        except: c_num += 1
    else: curr.append(s)
if curr: sonnets_map_bert[c_num] = curr

print("Scoring 'THE SONNETS'...")
for i in range(1, 21): # Sample first 20 for speed/viz
    if i in sonnets_map_bert:
        coh = score_semantic_coherence(sonnets_map_bert[i][:14])
        coherence_data.append({"Work": "THE SONNETS", "Type": "Real Sonnet", "Coherence": coh})

# B. Comparison Works
for title in COHERENCE_WORKS:
    print(f"Scoring '{title}'...")
    scores, clean, _ = analyze_work(title, works[title])
    top_indices = np.argsort(scores)[-TOP_N_CANDIDATES:][::-1]

    for idx in top_indices:
        lines = [clean[j][0] for j in range(idx, idx+14)]
        coh = score_semantic_coherence(lines)
        coherence_data.append({"Work": title, "Type": "Candidate", "Coherence": coh})

# 3. Visualization
import pandas as pd
df_coh = pd.DataFrame(coherence_data)

plt.figure(figsize=(10, 6))
sns.boxplot(x="Work", y="Coherence", data=df_coh, palette="Set2")
sns.stripplot(x="Work", y="Coherence", data=df_coh, color=".25", alpha=0.5)
plt.title("Semantic Coherence: Real Sonnets vs. Play Candidates")
plt.ylabel("Cosine Similarity (Coherence)")
plt.grid(axis='y', alpha=0.3)
plt.show()

# %% [markdown] id="M50HJ5AXzVb0"
# ## Style & Authorship

# %% colab={"base_uri": "https://localhost:8080/", "height": 754} id="ZLauOoHizPKh" outputId="40ca670f-5a2b-42f7-bd79-32f8a9b6cbff"
# Cell 3: Stylometry & Authorship Attribution

# --- CONFIGURATION ---
STYLO_WORKS = ["A MIDSUMMER NIGHT’S DREAM", "TROILUS AND CRESSIDA"]
# ---------------------

# 1. Prepare Data
corpus = []
labels = []

# A. Real Sonnets
for num, lines in sonnets_map_bert.items():
    if len(lines) < 14: continue
    corpus.append(" ".join(lines[:14]))
    labels.append("The Sonnets")

# B. Comparison Works
for title in STYLO_WORKS:
    scores, clean, _ = analyze_work(title, works[title])
    # Take top 20 candidates
    top_indices = np.argsort(scores)[-20:][::-1]
    for idx in top_indices:
        lines = [clean[j][0] for j in range(idx, idx+14)]
        corpus.append(" ".join(lines))
        labels.append(title)

# 2. Vectorize & PCA
vectorizer = CountVectorizer(max_features=100, stop_words=None)
X = vectorizer.fit_transform(corpus).toarray()
pca = PCA(n_components=2)
coords = pca.fit_transform(X)

# 3. Visualization
plt.figure(figsize=(12, 8))
sns.scatterplot(x=coords[:,0], y=coords[:,1], hue=labels, style=labels, s=100, palette="deep", alpha=0.8)

# Draw centroids
for label in set(labels):
    indices = [i for i, x in enumerate(labels) if x == label]
    centroid = np.mean(coords[indices], axis=0)
    plt.plot(centroid[0], centroid[1], 'k+', markersize=15, markeredgewidth=2)
    plt.text(centroid[0], centroid[1], f" {label}", fontsize=12, fontweight='bold')

plt.title("Stylometric Analysis (PCA of Top 100 Words)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# %% [markdown] id="B28jpAmoz-sQ"
# ## Distinctness

# %% colab={"base_uri": "https://localhost:8080/"} id="v4XGMmPppKEp" outputId="9d4c27a7-5360-449d-a5f3-45a96c329ae2"
# !ls -lh ~/.cache/huggingface/hub
# !find ~/.cache/huggingface -maxdepth 3 -type f -name "model.safetensors" -ls


# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} id="L0brMAW90C3-" outputId="b38f86b4-c693-4d68-d9de-25ee94dde167"
# Distinctness

# --- CONFIGURATION ---
DISTINCTNESS_WORKS = [
    "THE TRAGEDY OF ROMEO AND JULIET",
    "A MIDSUMMER NIGHT’S DREAM",
    "THE TRAGEDY OF HAMLET, PRINCE OF DENMARK"
]
TOP_N_DISTINCT = 10
EMBED_DIM = 768  # bert-base-uncased hidden size
# ---------------------


# --- BERT / EMBEDDING SETUP ---
# Only load BERT if not already present (prevents double-loading if you ran another cell earlier)
if "bert_tokenizer" not in globals() or "bert_model" not in globals():
    print("Loading BERT model for distinctness analysis...")
    bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    bert_model = BertModel.from_pretrained("bert-base-uncased")
    bert_model.eval()
    print("BERT loaded.")

def get_line_embedding(line: str) -> np.ndarray:
    """
    Returns a (1, EMBED_DIM) numpy array for a single line of text,
    using the [CLS] token embedding from bert-base-uncased.
    """
    inputs = bert_tokenizer(
        line,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128
    )
    with torch.no_grad():
        outputs = bert_model(**inputs)
    # CLS token embedding
    cls_emb = outputs.last_hidden_state[:, 0, :]  # shape (1, EMBED_DIM)
    return cls_emb.numpy()


# --- DISTINCTNESS HELPERS ---
def get_text_centroid(lines):
    """
    Given a list of lines, compute the centroid of their BERT embeddings.
    Returns a (1, EMBED_DIM) numpy array.
    """
    if not lines:
        return np.zeros((1, EMBED_DIM))
    embeddings = [get_line_embedding(l) for l in lines if l.strip()]
    if not embeddings:
        return np.zeros((1, EMBED_DIM))
    return np.mean(np.vstack(embeddings), axis=0, keepdims=True)


def analyze_contextual_distinctness(work_title, top_n=5):
    """
    For a given work:
      - Uses 'analyze_work' to get candidate scores & clean lines
      - Picks top_n non-overlapping candidates (NMS-style)
      - Measures semantic and stylistic distance vs. surrounding context
    """
    print(f"\nAnalyzing Contextual Distinctness for: {work_title}")

    # 'works' and 'analyze_work' must be defined elsewhere in your notebook
    scores, clean_data, _ = analyze_work(work_title, works[work_title])
    scores_array = np.array(scores)

    # Non-Maximum Suppression (to avoid overlapping 14-line windows)
    top_indices = []
    taken_mask = np.zeros(len(scores), dtype=bool)
    sorted_indices = scores_array.argsort()[::-1]  # high to low

    for idx in sorted_indices:
        if len(top_indices) >= top_n:
            break
        if taken_mask[idx]:
            continue
        top_indices.append(idx)
        taken_mask[max(0, idx - 14):min(len(scores), idx + 14)] = True

    results = []
    for rank, idx in enumerate(top_indices):
        candidate_lines = [x[0] for x in clean_data[idx : idx + 14]]
        pre_context = [x[0] for x in clean_data[max(0, idx - 14) : idx]]
        post_context = [x[0] for x in clean_data[idx + 14 : min(len(clean_data), idx + 28)]]
        context_lines = pre_context + post_context

        if not context_lines:
            continue

        # Semantic Distance (BERT centroid cosine distance)
        cand_vec = get_text_centroid(candidate_lines)   # (1, EMBED_DIM)
        cont_vec = get_text_centroid(context_lines)     # (1, EMBED_DIM)
        sem_dist = 1.0 - cosine_similarity(cand_vec, cont_vec)[0][0]

        # Stylistic Distance (Jaccard distance over word sets)
        cand_words = set(" ".join(candidate_lines).lower().split())
        cont_words = set(" ".join(context_lines).lower().split())
        inter = len(cand_words.intersection(cont_words))
        union = len(cand_words.union(cont_words))
        style_dist = 1.0 - (inter / union if union > 0 else 0)

        results.append({
            "Rank":          rank + 1,
            "Score":         scores[idx],
            "Semantic_Dist": sem_dist,
            "Style_Dist":    style_dist,
            "Snippet":       candidate_lines[0]
        })

    return results


# --- RUN ANALYSIS ---
all_results = []
for title in DISTINCTNESS_WORKS:
    res = analyze_contextual_distinctness(title, top_n=TOP_N_DISTINCT)
    for r in res:
        r["Work"] = title
        all_results.append(r)

# --- VISUALIZATION ---
df_dist = pd.DataFrame(all_results)

plt.figure(figsize=(10, 8))
sns.scatterplot(
    data=df_dist,
    x="Semantic_Dist",
    y="Style_Dist",
    hue="Work",
    style="Work",
    s=150,
    palette="viridis"
)

# Add labels with rank numbers
for i in range(df_dist.shape[0]):
    plt.text(
        df_dist.Semantic_Dist[i] + 0.0005,
        df_dist.Style_Dist[i],
        f"#{df_dist.Rank[i]}",
        fontsize=9,
        alpha=0.7
    )

plt.title("Contextual Distinctness: Semantic vs. Stylistic Distance")
plt.xlabel("Semantic Distance (Meaning Shift)")
plt.ylabel("Stylistic Distance (Vocab Shift)")
plt.legend(loc="lower right")
#plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Table Output
print("\nDetailed Distinctness Data:")
print(df_dist[["Work", "Rank", "Score", "Semantic_Dist", "Style_Dist", "Snippet"]].to_string(index=False))


# %% colab={"base_uri": "https://localhost:8080/"} id="TdVCIWkPXWbq" outputId="85f28177-92ad-4cd2-b6fb-0005731f4887"
TOP_N_DISTINCT = 20

for title in DISTINCTNESS_WORKS:
    res = analyze_contextual_distinctness(title, top_n=TOP_N_DISTINCT)
    for r in res:
        r["Work"] = title
        all_results.append(r)

# --- VISUALIZATION ---
df_dist = pd.DataFrame(all_results)

plt.figure(figsize=(10, 8))
sns.scatterplot(
    data=df_dist,
    x="Semantic_Dist",
    y="Style_Dist",
    hue="Work",
    style="Work",
    s=90,
    palette="viridis"
)

# Add labels with rank numbers
for i in range(df_dist.shape[0]):
    plt.text(
        df_dist.Semantic_Dist[i] + 0.0005,
        df_dist.Style_Dist[i],
        f"#{df_dist.Rank[i]}",
        fontsize=7,
        alpha=0.7
    )

plt.title("Contextual Distinctness: Semantic vs. Stylistic Distance")
plt.xlabel("Semantic Distance (Meaning Shift)")
plt.ylabel("Stylistic Distance (Vocab Shift)")
plt.legend(loc="lower right")
#plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Table Output
print("\nDetailed Distinctness Data:")
print(df_dist[["Work", "Rank", "Score", "Semantic_Dist", "Style_Dist", "Snippet"]].to_string(index=False))
