# First full run (version 5)

Saved output from the Colab run of notebook version 5: key printed text and every figure.

## Cell 6: classifier training

```text
Extracted 154 positive sonnets.
106420 clean play windows.

--- Supervised Classifier: held-out ROC-AUC (5-fold CV) ---
  structural only: AUC = 0.976
  structural + TF-IDF: AUC = 0.978
  -> USE_TFIDF = False
Supervised Model Trained.

--- PU Classifier (Elkan & Noto) ---
  held-out AUC (labeled vs unlabeled) = 0.977
  Estimated c (held-out) = 0.411   [first version, in-sample + balanced weights: 0.981]
PU Model Trained.
Caveat: PU assumes labeled sonnets are a random sample of all sonnets. Play sonnets are
often 10-18 lines and split between speakers, so treat PU probabilities as a ranking.
```

## Cell 7: scan and exports

```text
--- Verification ---
Supervised: Friar=0.0594, Prologue=0.7807
PU:         Friar=0.0148, Prologue=0.4280

--- Scanning ALL Play Windows for Top Candidates ---
Scanning 106420 windows...

--- Exporting Top 2% Candidates ---
Supervised Top 2% Threshold: 0.8969
PU Top 2% Threshold: 0.7305
Count Supervised: 2129
Count PU: 2129
Count Intersection: 1807
Saved: sonnet_candidates_supervised.txt
Saved: sonnet_candidates_pu.txt
Saved: sonnet_candidates_intersection.txt

--- Generating Structured CSV ---

SUCCESS: Top 2% candidates stored in DataFrame 'sonnet_candidates_df' (2451 rows).
Columns: ['Play', 'Start_Line', 'End_Line', 'Supervised_Score', 'PU_Score', 'Is_Top_2_Pct_Sup', 'Is_Top_2_Pct_PU', 'Is_Intersection']
Saved CSV: sonnet_candidates_structured.csv

--- Generating Visualizations ---
/usr/local/lib/python3.13/dist-packages/IPython/core/pylabtools.py:151: UserWarning: Creating legend with loc="best" can be slow with large amounts of data.
  fig.canvas.print_figure(bytes_io, **kw)
```

**Figure 1.** Supervised scores for play windows vs held-out canonical sonnets (left); top-2% candidates across the corpus (right)

![Supervised scores for play windows vs held-out canonical sonnets (left); top-2% candidates across the corpus (right)](run-1-figures/01-classifier-scores.png)

## Cell 8b: evaluation on the known play sonnets

```text
Located 15 of 22 known play sonnets in the cleaned plays.
Not in the corpus (cannot be recalled): ['The Two Gentlemen of Verona, 3.1.140–9.', 'Edward III, scene 2.322–3.', 'Love’s Labour’s Lost, 4.3.24–39.', 'Romeo and Juliet, 1.5.92–109.', 'Much Ado About Nothing, 3.1.107–16.', 'All’s Well That Ends Well, 3.4.4–17.', 'Pericles, scene 10.1–14.']
                                          label                             play  line split
               The Comedy of Errors, 3.2.41–54.             THE COMEDY OF ERRORS   770  test
                Love’s Labour’s Lost, 1.1.80–9.             LOVE’S LABOUR’S LOST   101  tune
              Love’s Labour’s Lost, 1.1.159–74.             LOVE’S LABOUR’S LOST   181  tune
              Love’s Labour’s Lost, 4.2.106–19.             LOVE’S LABOUR’S LOST  1115  test
               Love’s Labour’s Lost, 4.3.57–70.             LOVE’S LABOUR’S LOST  1212  tune
              Love’s Labour’s Lost, 5.2.401–15.             LOVE’S LABOUR’S LOST  2064  test
                    Romeo and Juliet, Prologue.  THE TRAGEDY OF ROMEO AND JULIET    31  tune
Romeo and Juliet, 2.0. Introduction to Act Two.  THE TRAGEDY OF ROMEO AND JULIET   757  test
         A Midsummer Night’s Dream, 5.1.108–17.        A MIDSUMMER NIGHT’S DREAM  1812  tune
                             Henry V, Epilogue. THE LIFE OF KING HENRY THE FIFTH  3026  tune
                      As You Like It, 3.2.1–10.                   AS YOU LIKE IT  1078  test
              Troilus and Cressida, 1.2.278–91.             TROILUS AND CRESSIDA   421  test
        All’s Well That Ends Well, 1.1.212–225.        ALL’S WELL THAT ENDS WELL   202  test
                     Pericles, scene 21.225–35.         PERICLES, PRINCE OF TYRE  2235  test
                        Cymbeline, 5.5.187–204.                        CYMBELINE  2952  tune

Scanning 38 plays with the rule detector (cached after the first run)...
=== Baseline: how well does each detector find the known play sonnets? ===
      detector  n_known  median_rank  recall@10  recall@25  recall@50  recall@100  recall@250 split
rule (WEIGHTS)        7         40.0      0.286      0.429      0.714       0.714       0.714  tune
rule (WEIGHTS)        8         27.5      0.375      0.500      0.625       0.625       1.000  test
supervised clf        7          inf      0.000      0.143      0.143       0.286       0.286  tune
supervised clf        8        492.5      0.000      0.000      0.125       0.125       0.125  test
        PU clf        7        618.0      0.000      0.000      0.143       0.143       0.429  tune
        PU clf        8        548.5      0.000      0.000      0.000       0.000       0.125  test

Rank of each known sonnet (lower is better; inf = never in the hit list):
                                          label split  rule (WEIGHTS)  supervised clf  PU clf
               The Comedy of Errors, 3.2.41–54.  test             1.0             inf     inf
                Love’s Labour’s Lost, 1.1.80–9.  tune             2.0             inf     inf
              Love’s Labour’s Lost, 1.1.159–74.  tune            14.0            21.0   125.0
              Love’s Labour’s Lost, 4.2.106–19.  test            12.0           378.0   639.0
               Love’s Labour’s Lost, 4.3.57–70.  tune             inf             inf   129.0
              Love’s Labour’s Lost, 5.2.401–15.  test             4.0           528.0   437.0
                    Romeo and Juliet, Prologue.  tune             5.0           642.0   618.0
Romeo and Juliet, 2.0. Introduction to Act Two.  test             3.0            37.0   221.0
         A Midsummer Night’s Dream, 5.1.108–17.  tune            46.0             inf     inf
                             Henry V, Epilogue.  tune            40.0            83.0    46.0
                      As You Like It, 3.2.1–10.  test            43.0          1353.0  1264.0
              Troilus and Cressida, 1.2.278–91.  test           115.0           457.0   458.0
        All’s Well That Ends Well, 1.1.212–225.  test           218.0           434.0   399.0
                     Pericles, scene 21.225–35.  test           124.0             inf     inf
                        Cymbeline, 5.5.187–204.  tune             inf             inf     inf

Tuned weights (chosen on the tune half): {'meter': np.float64(0.4), 'syllable': np.float64(0.0), 'rhyme': np.float64(0.3), 'couplet': np.float64(0.3)}
  tune: recall@50 0.71 -> 0.71; median rank 40 -> 35
  test: recall@50 0.62 -> 0.62; median rank 28 -> 28
If the TEST numbers improve, copy the tuned weights into WEIGHTS in cell 4 and re-run.
```

## Cell 9: THE SONNETS analysis

**Figure 2.** Rule-detector signal along THE SONNETS

![Rule-detector signal along THE SONNETS](run-1-figures/02-sonnets-signal.png)

**Figure 3.** Scores of the 154 canonical sonnets and the 22 known play sonnets

![Scores of the 154 canonical sonnets and the 22 known play sonnets](run-1-figures/03-sonnets-and-play-sonnets-scores.png)

**Figure 4.** Sonnets ranked by score, marked by rhyme-scheme deviation

![Sonnets ranked by score, marked by rhyme-scheme deviation](run-1-figures/04-sonnets-ranked-rhyme-deviations.png)

**Figure 5.** How often each word is used in the Sonnets

![How often each word is used in the Sonnets](run-1-figures/05-sonnets-vocabulary-histogram.png)

**Figure 6.** Sonnets clustered by rare-word usage: K-Means

![Sonnets clustered by rare-word usage: K-Means](run-1-figures/06-clusters-kmeans.png)

**Figure 7.** Sonnets clustered by rare-word usage: Agglomerative (Ward)

![Sonnets clustered by rare-word usage: Agglomerative (Ward)](run-1-figures/07-clusters-ward.png)

**Figure 8.** Ward dendrogram of the sonnets

![Ward dendrogram of the sonnets](run-1-figures/08-dendrogram-ward.png)

**Figure 9.** Sonnets clustered by rare-word usage: Spectral

![Sonnets clustered by rare-word usage: Spectral](run-1-figures/09-clusters-spectral.png)

**Figure 10.** Cluster overlap by sonnets: K-Means vs Ward

![Cluster overlap by sonnets: K-Means vs Ward](run-1-figures/10-overlap-kmeans-vs-ward.png)

**Figure 11.** Cluster overlap by sonnets: K-Means vs Spectral

![Cluster overlap by sonnets: K-Means vs Spectral](run-1-figures/11-overlap-kmeans-vs-spectral.png)

**Figure 12.** Jaccard of cluster word sets: K-Means vs Ward

![Jaccard of cluster word sets: K-Means vs Ward](run-1-figures/12-jaccard-kmeans-vs-ward.png)

**Figure 13.** Jaccard of cluster word sets: K-Means vs Spectral

![Jaccard of cluster word sets: K-Means vs Spectral](run-1-figures/13-jaccard-kmeans-vs-spectral.png)

**Figure 14.** Jaccard of cluster word sets: Ward vs Spectral

![Jaccard of cluster word sets: Ward vs Spectral](run-1-figures/14-jaccard-ward-vs-spectral.png)

## Cell 10: Midsummer (top hits, statistics)

```text
Analyzing A MIDSUMMER NIGHT’S DREAM...

Top 10 Scoring Segments in A MIDSUMMER NIGHT’S DREAM (Distinct Regions):
========================================

Rank 1 | Score: 0.8877 | Start Line: 2051
--------------------
[PUCK] Now the wasted brands do glow,
Whilst the screech-owl, screeching loud,
Puts the wretch that lies in woe
In remembrance of a shroud.
Now it is the time of night
That the graves, all gaping wide,
Every one lets forth his sprite,
In the church-way paths to glide.
And we fairies, that do run
By the triple Hecate’s team
From the presence of the sun,
Following darkness like a dream,
Now are frolic; not a mouse
Shall disturb this hallow’d house.

Rank 2 | Score: 0.8820 | Start Line: 1844
--------------------
[THESEUS] And as she fled, her mantle she did fall;
Which Lion vile with bloody mouth did stain.
Anon comes Pyramus, sweet youth, and tall,
And finds his trusty Thisbe’s mantle slain;
Whereat with blade, with bloody blameful blade,
He bravely broach’d his boiling bloody breast;
And Thisbe, tarrying in mulberry shade,
His dagger drew, and died. For all the rest,
Let Lion, Moonshine, Wall, and lovers twain,
At large discourse while here they do remain.
I wonder if the lion be to speak.
[DEMETRIUS] No wonder, my lord. One lion may, when many asses do.
[WALL] In this same interlude it doth befall
That I, one Snout by name, present a wall:

Rank 3 | Score: 0.8357 | Start Line: 1058
--------------------
[DEMETRIUS] So sorrow’s heaviness doth heavier grow
For debt that bankrupt sleep doth sorrow owe;
Which now in some slight measure it will pay,
If for his tender here I make some stay.
[OBERON] What hast thou done? Thou hast mistaken quite,
And laid the love-juice on some true-love’s sight.
Of thy misprision must perforce ensue
Some true love turn’d, and not a false turn’d true.
[PUCK] Then fate o’er-rules, that, one man holding troth,
A million fail, confounding oath on oath.
[OBERON] About the wood go swifter than the wind,
And Helena of Athens look thou find.
All fancy-sick she is, and pale of cheer
With sighs of love, that costs the fresh blood dear.

Rank 4 | Score: 0.8357 | Start Line: 1151
--------------------
[HERMIA] Dark night, that from the eye his function takes,
The ear more quick of apprehension makes;
Wherein it doth impair the seeing sense,
It pays the hearing double recompense.
Thou art not by mine eye, Lysander, found;
Mine ear, I thank it, brought me to thy sound.
But why unkindly didst thou leave me so?
[LYSANDER] Why should he stay whom love doth press to go?
[HERMIA] What love could press Lysander from my side?
[LYSANDER] Lysander’s love, that would not let him bide,
Fair Helena, who more engilds the night
Than all yon fiery oes and eyes of light.
Why seek’st thou me? Could not this make thee know
The hate I bare thee made me leave thee so?

Rank 5 | Score: 0.8311 | Start Line: 241
--------------------
[HERMIA] Take comfort: he no more shall see my face;
Lysander and myself will fly this place.
Before the time I did Lysander see,
Seem’d Athens as a paradise to me.
O, then, what graces in my love do dwell,
That he hath turn’d a heaven into hell!
[LYSANDER] Helen, to you our minds we will unfold:
Tomorrow night, when Phoebe doth behold
Her silver visage in the watery glass,
Decking with liquid pearl the bladed grass
(A time that lovers’ flights doth still conceal),
Through Athens’ gates have we devis’d to steal.
[HERMIA] And in the wood where often you and I
Upon faint primrose beds were wont to lie,

Rank 6 | Score: 0.8265 | Start Line: 685
--------------------
[LYSANDER] Fair love, you faint with wand’ring in the wood.
And, to speak troth, I have forgot our way.
We’ll rest us, Hermia, if you think it good,
And tarry for the comfort of the day.
[HERMIA] Be it so, Lysander: find you out a bed,
For I upon this bank will rest my head.
[LYSANDER] One turf shall serve as pillow for us both;
One heart, one bed, two bosoms, and one troth.
[HERMIA] Nay, good Lysander; for my sake, my dear,
Lie further off yet, do not lie so near.
[LYSANDER] O take the sense, sweet, of my innocence!
Love takes the meaning in love’s conference.
I mean that my heart unto yours is knit,
So that but one heart we can make of it:

Rank 7 | Score: 0.8255 | Start Line: 1038
--------------------
[DEMETRIUS] I had rather give his carcass to my hounds.
[HERMIA] Out, dog! Out, cur! Thou driv’st me past the bounds
Of maiden’s patience. Hast thou slain him, then?
Henceforth be never number’d among men!
O once tell true; tell true, even for my sake!
Durst thou have look’d upon him, being awake,
And hast thou kill’d him sleeping? O brave touch!
Could not a worm, an adder, do so much?
An adder did it; for with doubler tongue
Than thine, thou serpent, never adder stung.
[DEMETRIUS] You spend your passion on a mispris’d mood:
I am not guilty of Lysander’s blood;
Nor is he dead, for aught that I can tell.
[HERMIA] I pray thee, tell me then that he is well.

Rank 8 | Score: 0.8180 | Start Line: 274
--------------------
[HELENA] Love looks not with the eyes, but with the mind;
And therefore is wing’d Cupid painted blind.
Nor hath love’s mind of any judgment taste.
Wings, and no eyes, figure unheedy haste.
And therefore is love said to be a child,
Because in choice he is so oft beguil’d.
As waggish boys in game themselves forswear,
So the boy Love is perjur’d everywhere.
For, ere Demetrius look’d on Hermia’s eyne,
He hail’d down oaths that he was only mine;
And when this hail some heat from Hermia felt,
So he dissolv’d, and showers of oaths did melt.
I will go tell him of fair Hermia’s flight.
Then to the wood will he tomorrow night

Rank 9 | Score: 0.8072 | Start Line: 1112
--------------------
[DEMETRIUS] Crystal is muddy. O how ripe in show
Thy lips, those kissing cherries, tempting grow!
That pure congealèd white, high Taurus’ snow,
Fann’d with the eastern wind, turns to a crow
When thou hold’st up thy hand. O, let me kiss
This princess of pure white, this seal of bliss!
[HELENA] O spite! O hell! I see you all are bent
To set against me for your merriment.
If you were civil, and knew courtesy,
You would not do me thus much injury.
Can you not hate me, as I know you do,
But you must join in souls to mock me too?
If you were men, as men you are in show,
You would not use a gentle lady so;

Rank 10 | Score: 0.8049 | Start Line: 1812
--------------------
[THESEUS] If we offend, it is with our good will.
That you should think, we come not to offend,
But with good will.  To show our simple skill,
That is the true beginning of our end.
Consider then, we come but in despite.
We do not come, as minding to content you,
Our true intent is.  All for your delight
We are not here.  That you should here repent you,
The actors are at hand, and, by their show,
You shall know all that you are like to know.
This fellow doth not stand upon points.
[LYSANDER] He hath rid his prologue like a rough colt; he knows not the stop. A
good moral, my lord: it is not enough to speak, but to speak true.
[HIPPOLYTA] Indeed he hath played on this prologue like a child on a recorder; a

Rank 11 | Score: 0.7966 | Start Line: 223
--------------------
[HELENA] More tuneable than lark to shepherd’s ear,
When wheat is green, when hawthorn buds appear.
Sickness is catching. O were favour so,
Yours would I catch, fair Hermia, ere I go.
My ear should catch your voice, my eye your eye,
My tongue should catch your tongue’s sweet melody.
Were the world mine, Demetrius being bated,
The rest I’d give to be to you translated.
O, teach me how you look, and with what art
You sway the motion of Demetrius’ heart!
[HERMIA] I frown upon him, yet he loves me still.
[HELENA] O that your frowns would teach my smiles such skill!
[HERMIA] I give him curses, yet he gives me love.
[HELENA] O that my prayers could such affection move!

Rank 12 | Score: 0.7950 | Start Line: 1351
--------------------
[OBERON] Till o’er their brows death-counterfeiting sleep
With leaden legs and batty wings doth creep.
Then crush this herb into Lysander’s eye,
Whose liquor hath this virtuous property,
To take from thence all error with his might
And make his eyeballs roll with wonted sight.
When they next wake, all this derision
Shall seem a dream and fruitless vision;
And back to Athens shall the lovers wend,
With league whose date till death shall never end.
Whiles I in this affair do thee employ,
I’ll to my queen, and beg her Indian boy;
And then I will her charmèd eye release
From monster’s view, and all things shall be peace.

Rank 13 | Score: 0.7821 | Start Line: 762
--------------------
[LYSANDER] Not Hermia, but Helena I love.
Who will not change a raven for a dove?
The will of man is by his reason sway’d,
And reason says you are the worthier maid.
Things growing are not ripe until their season;
So I, being young, till now ripe not to reason;
And touching now the point of human skill,
Reason becomes the marshal to my will,
And leads me to your eyes, where I o’erlook
Love’s stories, written in love’s richest book.
[HELENA] Wherefore was I to this keen mockery born?
When at your hands did I deserve this scorn?
Is’t not enough, is’t not enough, young man,
That I did never, no, nor never can

Rank 14 | Score: 0.7785 | Start Line: 993
--------------------
[PUCK] And forth my mimic comes. When they him spy,
As wild geese that the creeping fowler eye,
Or russet-pated choughs, many in sort,
Rising and cawing at the gun’s report,
Sever themselves and madly sweep the sky,
So at his sight away his fellows fly,
And at our stamp, here o’er and o’er one falls;
He murder cries, and help from Athens calls.
Their sense thus weak, lost with their fears, thus strong,
Made senseless things begin to do them wrong;
For briers and thorns at their apparel snatch;
Some sleeves, some hats, from yielders all things catch.
I led them on in this distracted fear,
And left sweet Pyramus translated there.

Rank 15 | Score: 0.7745 | Start Line: 926
--------------------
[TITANIA] Out of this wood do not desire to go.
Thou shalt remain here whether thou wilt or no.
I am a spirit of no common rate.
The summer still doth tend upon my state;
And I do love thee: therefore, go with me.
I’ll give thee fairies to attend on thee;
And they shall fetch thee jewels from the deep,
And sing, while thou on pressèd flowers dost sleep.
And I will purge thy mortal grossness so
That thou shalt like an airy spirit go.—
Peaseblossom! Cobweb! Moth! and Mustardseed!
[PEASEBLOSSOM] Ready.
[COBWEB] And I.
[MOTH] And I.

Rank 16 | Score: 0.7692 | Start Line: 1371
--------------------
[PUCK] Already to their wormy beds are gone;
For fear lest day should look their shames upon,
They wilfully themselves exile from light,
And must for aye consort with black-brow’d night.
[OBERON] But we are spirits of another sort:
I with the morning’s love have oft made sport;
And, like a forester, the groves may tread
Even till the eastern gate, all fiery-red,
Opening on Neptune with fair blessèd beams,
Turns into yellow gold his salt-green streams.
But, notwithstanding, haste, make no delay.
We may effect this business yet ere day.
[PUCK] Up and down, up and down,
I will lead them up and down.

Rank 17 | Score: 0.7658 | Start Line: 1088
--------------------
[PUCK] Shall we their fond pageant see?
Lord, what fools these mortals be!
[OBERON] Stand aside. The noise they make
Will cause Demetrius to awake.
[PUCK] Then will two at once woo one.
That must needs be sport alone;
And those things do best please me
That befall prepost’rously.
[LYSANDER] Why should you think that I should woo in scorn?
Scorn and derision never come in tears.
Look when I vow, I weep; and vows so born,
In their nativity all truth appears.
How can these things in me seem scorn to you,
Bearing the badge of faith, to prove them true?

Rank 18 | Score: 0.7426 | Start Line: 396
--------------------
[PUCK] For Oberon is passing fell and wrath,
Because that she, as her attendant, hath
A lovely boy, stol’n from an Indian king;
She never had so sweet a changeling.
And jealous Oberon would have the child
Knight of his train, to trace the forests wild:
But she perforce withholds the lovèd boy,
Crowns him with flowers, and makes him all her joy.
And now they never meet in grove or green,
By fountain clear, or spangled starlight sheen,
But they do square; that all their elves for fear
Creep into acorn cups, and hide them there.
Either I mistake your shape and making quite,
Or else you are that shrewd and knavish sprite

Rank 19 | Score: 0.7415 | Start Line: 1329
--------------------
[HELENA] Your hands than mine are quicker for a fray.
My legs are longer though, to run away.
[HERMIA] I am amaz’d, and know not what to say.
[OBERON] This is thy negligence: still thou mistak’st,
Or else commit’st thy knaveries willfully.
[PUCK] Believe me, king of shadows, I mistook.
Did not you tell me I should know the man
By the Athenian garments he had on?
And so far blameless proves my enterprise
That I have ’nointed an Athenian’s eyes:
And so far am I glad it so did sort,
As this their jangling I esteem a sport.
[OBERON] Thou seest these lovers seek a place to fight.
Hie therefore, Robin, overcast the night;

Rank 20 | Score: 0.7349 | Start Line: 738
--------------------
[HELENA] O, I am out of breath in this fond chase!
The more my prayer, the lesser is my grace.
Happy is Hermia, wheresoe’er she lies,
For she hath blessèd and attractive eyes.
How came her eyes so bright? Not with salt tears.
If so, my eyes are oftener wash’d than hers.
No, no, I am as ugly as a bear,
For beasts that meet me run away for fear:
Therefore no marvel though Demetrius
Do, as a monster, fly my presence thus.
What wicked and dissembling glass of mine
Made me compare with Hermia’s sphery eyne?
But who is here? Lysander, on the ground!
Dead or asleep? I see no blood, no wound.

Total unique word types in A MIDSUMMER NIGHT’S DREAM: 3002

Lines in TEST_GROUP_POETIC: 280
Lines in TEST_GROUP_NORMAL: 1518

=== RARE-WORD PERCENTAGES (TOKEN-BASED) ===
TEST_GROUP_POETIC:  total tokens = 2242
  % of tokens with frequency 1–2:  16.99%
  % of tokens with frequency 1–3:  21.86%

TEST_GROUP_NORMAL:  total tokens = 11755
  % of tokens with frequency 1–2:  15.93%
  % of tokens with frequency 1–3:  19.91%

Comparison (Poetic minus Normal):
  Δ% (1–2):  1.07 percentage points
  Δ% (1–3):  1.94 percentage points

=== T-TESTS: RARE WORD COUNTS PER LINE (POETIC vs NORMAL) ===

1-use words:
  Poetic lines:  mean=0.861, std=0.938, n=280
  Normal lines:  mean=0.809, std=0.934, n=1518
  Welch t-test: t = 0.849, p = 0.3965

1–2-use words:
  Poetic lines:  mean=1.361, std=1.162, n=280
  Normal lines:  mean=1.233, std=1.102, n=1518
  Welch t-test: t = 1.701, p = 0.08981

1–3-use words:
  Poetic lines:  mean=1.750, std=1.242, n=280
  Normal lines:  mean=1.542, std=1.196, n=1518
  Welch t-test: t = 2.587, p = 0.01004

Generating Score Distribution Comparison...

Generating Word Usage Comparison Chart...
```

**Figure 15.** Detection signal in A Midsummer Night's Dream, with classifier overlay and top hits

![Detection signal in A Midsummer Night's Dream, with classifier overlay and top hits](run-1-figures/15-midsummer-signal.png)

**Figure 16.** Wavelet power spectrum of the Midsummer signal

![Wavelet power spectrum of the Midsummer signal](run-1-figures/16-midsummer-wavelet-power.png)

**Figure 17.** Wavelet power with significance contours

![Wavelet power with significance contours](run-1-figures/17-midsummer-wavelet-significance.png)

**Figure 18.** How often each word is used in Midsummer

![How often each word is used in Midsummer](run-1-figures/18-midsummer-vocabulary-histogram.png)

**Figure 19.** Score distribution: Midsummer windows vs all sonnets

![Score distribution: Midsummer windows vs all sonnets](run-1-figures/19-midsummer-vs-sonnets-score-kde.png)

**Figure 20.** Fraction of word types by frequency: Sonnets vs Midsummer

![Fraction of word types by frequency: Sonnets vs Midsummer](run-1-figures/20-midsummer-word-usage.png)

## Temporal analysis

**Figure 21.** Sonnets on PC1 vs PC2, coloured by composition date

![Sonnets on PC1 vs PC2, coloured by composition date](run-1-figures/21-temporal-pca-pc1-pc2.png)

**Figure 22.** Feature loadings for PC1 and PC2

![Feature loadings for PC1 and PC2](run-1-figures/22-temporal-loadings-pc1-pc2.png)

**Figure 23.** Sonnets on PC3 vs PC4, coloured by composition date

![Sonnets on PC3 vs PC4, coloured by composition date](run-1-figures/23-temporal-pca-pc3-pc4.png)

**Figure 24.** Feature loadings for PC3 and PC4

![Feature loadings for PC3 and PC4](run-1-figures/24-temporal-loadings-pc3-pc4.png)

## Combined PCA

**Figure 25.** Combined PCA: temporal metrics, LSA themes and rare-word Ward clusters

![Combined PCA: temporal metrics, LSA themes and rare-word Ward clusters](run-1-figures/25-combined-pca.png)

## Word-usage chart

**Figure 26.** Fraction of word types by frequency: Sonnets and three plays

![Fraction of word types by frequency: Sonnets and three plays](run-1-figures/26-word-usage-four-works.png)

## Sonnet Finder V2 (neural mode)

```text
Advanced Sonnet Detector Ready.
Analyzing THE TRAGEDY OF ROMEO AND JULIET with Neural Detector...

Top 10 Scoring Segments in THE TRAGEDY OF ROMEO AND JULIET (Neural Detector):
========================================

Rank 1 | Score: 0.9566 | Start Line: 757
--------------------
[CHORUS] Now old desire doth in his deathbed lie,
And young affection gapes to be his heir;
That fair for which love groan’d for and would die,
With tender Juliet match’d, is now not fair.
Now Romeo is belov’d, and loves again,
Alike bewitched by the charm of looks;
But to his foe suppos’d he must complain,
And she steal love’s sweet bait from fearful hooks:
Being held a foe, he may not have access
To breathe such vows as lovers use to swear;
And she as much in love, her means much less
To meet her new beloved anywhere.
But passion lends them power, time means, to meet,
Tempering extremities with extreme sweet.

Rank 2 | Score: 0.9026 | Start Line: 31
--------------------
[CHORUS] Two households, both alike in dignity,
In fair Verona, where we lay our scene,
From ancient grudge break to new mutiny,
Where civil blood makes civil hands unclean.
From forth the fatal loins of these two foes
A pair of star-cross’d lovers take their life;
Whose misadventur’d piteous overthrows
Doth with their death bury their parents’ strife.
The fearful passage of their death-mark’d love,
And the continuance of their parents’ rage,
Which, but their children’s end, nought could remove,
Is now the two hours’ traffic of our stage;
The which, if you with patient ears attend,
What here shall miss, our toil shall strive to mend.

Rank 3 | Score: 0.8810 | Start Line: 704
--------------------
[JULIET] Good pilgrim, you do wrong your hand too much,
Which mannerly devotion shows in this;
For saints have hands that pilgrims’ hands do touch,
And palm to palm is holy palmers’ kiss.
[ROMEO] Have not saints lips, and holy palmers too?
[JULIET] Ay, pilgrim, lips that they must use in prayer.
[ROMEO] O, then, dear saint, let lips do what hands do:
They pray, grant thou, lest faith turn to despair.
[JULIET] Saints do not move, though grant for prayers’ sake.
[ROMEO] Then move not while my prayer’s effect I take.
Thus from my lips, by thine my sin is purg’d.
[JULIET] Then have my lips the sin that they have took.
[ROMEO] Sin from my lips? O trespass sweetly urg’d!
Give me my sin again.

Rank 4 | Score: 0.8282 | Start Line: 298
--------------------
[CAPULET] At my poor house look to behold this night
Earth-treading stars that make dark heaven light:
Such comfort as do lusty young men feel
When well apparell’d April on the heel
Of limping winter treads, even such delight
Among fresh female buds shall you this night
Inherit at my house. Hear all, all see,
And like her most whose merit most shall be:
Which, on more view of many, mine, being one,
May stand in number, though in reckoning none.
Come, go with me. Go, sirrah, trudge about
Through fair Verona; find those persons out
Whose names are written there,  and to them say,
My house and welcome on their pleasure stay.

Rank 5 | Score: 0.8230 | Start Line: 1012
--------------------
[ROMEO] Sleep dwell upon thine eyes, peace in thy breast.
Would I were sleep and peace, so sweet to rest.
Hence will I to my ghostly Sire’s cell,
His help to crave and my dear hap to tell.
[FRIAR LAWRENCE] The grey-ey’d morn smiles on the frowning night,
Chequering the eastern clouds with streaks of light;
And fleckled darkness like a drunkard reels
From forth day’s pathway, made by Titan’s fiery wheels
Now, ere the sun advance his burning eye,
The day to cheer, and night’s dank dew to dry,
I must upfill this osier cage of ours
With baleful weeds and precious-juiced flowers.
The earth that’s nature’s mother, is her tomb;
What is her burying grave, that is her womb:

Rank 6 | Score: 0.7820 | Start Line: 1085
--------------------
[FRIAR LAWRENCE] Jesu Maria, what a deal of brine
Hath wash’d thy sallow cheeks for Rosaline!
How much salt water thrown away in waste,
To sea
```

**Figure 27.** Neural-mode detection signal in Romeo and Juliet

![Neural-mode detection signal in Romeo and Juliet](run-1-figures/27-romeo-juliet-neural-signal.png)

**Figure 28.** Neural-mode score distributions: Sonnets vs three plays

![Neural-mode score distributions: Sonnets vs three plays](run-1-figures/28-neural-score-distributions.png)

## Semantic coherence

```text
Loading BERT Model...
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
WARNING:huggingface_hub.utils._http:Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Key                                        | Status     |  | 
-------------------------------------------+------------+--+-
cls.predictions.bias                       | UNEXPECTED |  | 
cls.predictions.transform.LayerNorm.weight | UNEXPECTED |  | 
cls.predictions.transform.dense.weight     | UNEXPECTED |  | 
cls.seq_relationship.weight                | UNEXPECTED |  | 
cls.predictions.transform.LayerNorm.bias   | UNEXPECTED |  | 
cls.seq_relationship.bias                  | UNEXPECTED |  | 
cls.predictions.transform.dense.bias       | UNEXPECTED |  | 

Notes:
- UNEXPECTED:	can be ignored when loading from different task/architecture; not ok if you expect identical arch.
BERT Loaded.
Scoring 'THE SONNETS'...
Scoring 'A MIDSUMMER NIGHT’S DREAM'...
Analyzing A MIDSUMMER NIGHT’S DREAM...
Scoring 'TROILUS AND CRESSIDA'...
Analyzing TROILUS AND CRESSIDA...
/tmp/ipykernel_3719/660258789.py:66: FutureWarning: 

Passing `palette` without assigning `hue` is deprecated and will be removed in v0.14.0. Assign the `x` variable to `hue` and set `legend=False` for the same effect.

  sns.boxplot(x="Work", y="Coherence", data=df_coh, palette="Set2")
```

**Figure 29.** Semantic coherence by group

![Semantic coherence by group](run-1-figures/29-coherence.png)

## Stylometry

**Figure 30.** Stylometry: PCA of top-100-word counts

![Stylometry: PCA of top-100-word counts](run-1-figures/30-stylometry-pca.png)

## Distinctness

```text
Loading BERT model for distinctness analysis...
Key                                        | Status     |  | 
-------------------------------------------+------------+--+-
cls.predictions.bias                       | UNEXPECTED |  | 
cls.predictions.transform.LayerNorm.weight | UNEXPECTED |  | 
cls.predictions.transform.dense.weight     | UNEXPECTED |  | 
cls.seq_relationship.weight                | UNEXPECTED |  | 
cls.predictions.transform.LayerNorm.bias   | UNEXPECTED |  | 
cls.seq_relationship.bias                  | UNEXPECTED |  | 
cls.predictions.transform.dense.bias       | UNEXPECTED |  | 

Notes:
- UNEXPECTED:	can be ignored when loading from different task/architecture; not ok if you expect identical arch.
BERT loaded.

Analyzing Contextual Distinctness for: THE TRAGEDY OF ROMEO AND JULIET
Analyzing THE TRAGEDY OF ROMEO AND JULIET...

Analyzing Contextual Distinctness for: A MIDSUMMER NIGHT’S DREAM
Analyzing A MIDSUMMER NIGHT’S DREAM...

Analyzing Contextual Distinctness for: THE TRAGEDY OF HAMLET, PRINCE OF DENMARK
Analyzing THE TRAGEDY OF HAMLET, PRINCE OF DENMARK...

Detailed Distinctness Data:
                                    Work  Rank    Score  Semantic_Dist  Style_Dist                                                   Snippet
         THE TRAGEDY OF ROMEO AND JULIET     1 0.953750       0.024828    0.904040                  Now old desire doth in his deathbed lie,
         THE TRAGEDY OF ROMEO AND JULIET     2 0.901667       0.042355    0.924242                    Two households, both alike in dignity,
         THE TRAGEDY OF ROMEO AND JULIET     3 0.879750       0.028890    0.886139            Good pilgrim, you do wrong your hand too much,
         THE TRAGEDY OF ROMEO AND JULIET     4 0.826071       0.010576    0.889401         Sleep dwell upon thine eyes, peace in thy breast.
         THE TRAGEDY OF ROMEO AND JULIET     5 0.825536       0.010581    0.868421                At my poor house look to behold this night
         THE TRAGEDY OF ROMEO AND JULIET     6 0.781364       0.011561    0.904545           This precious book of love, this unbound lover,
         THE TRAGEDY OF ROMEO AND JULIET     7 0.777250       0.013136    0.868687             O, she doth teach the torches to burn bright!
         THE TRAGEDY OF ROMEO AND JULIET     8 0.766000       0.015539    0.892857            For naught so vile that on the earth doth live
         THE TRAGEDY OF ROMEO AND JULIET     9 0.765375       0.015327    0.901408                      When the devout religion of mine eye
         THE TRAGEDY OF ROMEO AND JULIET    10 0.754318       0.012266    0.889952                           At thy good heart’s oppression.
               A MIDSUMMER NIGHT’S DREAM     1 0.887679       0.011008    0.900000                            Now the wasted brands do glow,
               A MIDSUMMER NIGHT’S DREAM     2 0.882000       0.011036    0.823232                 And as she fled, her mantle she did fall;
               A MIDSUMMER NIGHT’S DREAM     3 0.835714       0.017686    0.859903                   So sorrow’s heaviness doth heavier grow
               A MIDSUMMER NIGHT’S DREAM     4 0.835714       0.014092    0.906863         Dark night, that from the eye his function takes,
               A MIDSUMMER NIGHT’S DREAM     5 0.831071       0.015046    0.881517               Take comfort: he no more shall see my face;
               A MIDSUMMER NIGHT’S DREAM     6 0.826488       0.015025    0.849246          Fair love, you faint with wand’ring in the wood.
               A MIDSUMMER NIGHT’S DREAM     7 0.825536       0.010748    0.882883               I had rather give his carcass to my hounds.
               A MIDSUMMER NIGHT’S DREAM     8 0.818036       0.014267    0.882075          Love looks not with the eyes, but with the mind;
               A MIDSUMMER NIGHT’S DREAM     9 0.807202       0.007816    0.861386                      Crystal is muddy. O how ripe in show
               A MIDSUMMER NIGHT’S DREAM    10 0.804875       0.023373    0.863415                   If we offend, it is with our good will.
THE TRAGEDY OF HAMLET, PRINCE OF DENMARK     1 0.831429       0.018341    0.911765                    That our devices still are overthrown.
THE TRAGEDY OF HAMLET, PRINCE OF DENMARK     2 0.719643       0.017189    0.884058                  But, woe is me, you are so sick of late,
THE TRAGEDY OF HAMLET, PRINCE OF DENMARK     3 0.705000       0.023195    0.877551                   The instances that second marriage move
THE TRAGEDY OF HAMLET, PRINCE OF DENMARK     4 0.664250       0.011717    0.867647                  Or in th’incestuous pleasure of his bed,
THE TRAGEDY OF HAMLET, PRINCE OF DENMARK     5 0.645114       0.010310    0.859729         And from his mother’s closet hath he dragg’d him.
THE TRAGEDY OF HAMLET, PRINCE OF DENMARK     6 0.632743       0.012671    0.873239                     Of habits evil, is angel yet in this,
THE TRAGEDY OF HAMLET, PRINCE OF DENMARK     7 0.632216       0.013
```

**Figure 31.** Contextual distinctness of the top 10 candidates per play

![Contextual distinctness of the top 10 candidates per play](run-1-figures/31-distinctness-top10.png)

## Distinctness, top 20

**Figure 32.** Contextual distinctness of the top 20 candidates per play

![Contextual distinctness of the top 20 candidates per play](run-1-figures/32-distinctness-top20.png)
