# Second full run (version 6)

Saved output from the Colab run of notebook version 6: key printed text and every figure.

## Cell 6: classifier training

```text
Extracted 154 positive sonnets.
107177 clean play windows.

--- Supervised Classifier: held-out ROC-AUC (5-fold CV) ---
  structural only: AUC = 0.964
  structural + TF-IDF: AUC = 0.981
  -> USE_TFIDF = True
Supervised Model Trained.

--- PU Classifier (Elkan & Noto) ---
  held-out AUC (labeled vs unlabeled) = 0.986
  Estimated c (held-out) = 0.645   [first version, in-sample + balanced weights: 0.981]
PU Model Trained.
Caveat: PU assumes labeled sonnets are a random sample of all sonnets. Play sonnets are
often 10-18 lines and split between speakers, so treat PU probabilities as a ranking.
```

## Cell 7: scan and exports

```text
--- Verification ---
Supervised: Friar=0.6878, Prologue=0.9802
PU:         Friar=0.0057, Prologue=0.2083

--- Scanning ALL Play Windows for Top Candidates ---
Scanning 107177 windows...

--- Exporting Top 2% Candidates ---
Supervised Top 2% Threshold: 0.9479
PU Top 2% Threshold: 0.3991
Count Supervised: 2144
Count PU: 2144
Count Intersection: 972
Saved: sonnet_candidates_supervised.txt
Saved: sonnet_candidates_pu.txt
Saved: sonnet_candidates_intersection.txt

--- Generating Structured CSV ---

SUCCESS: Top 2% candidates stored in DataFrame 'sonnet_candidates_df' (3316 rows).
Columns: ['Play', 'Start_Line', 'End_Line', 'Supervised_Score', 'PU_Score', 'Is_Top_2_Pct_Sup', 'Is_Top_2_Pct_PU', 'Is_Intersection']
Saved CSV: sonnet_candidates_structured.csv

--- Generating Visualizations ---
```

**Figure 1.** Supervised scores for play windows vs held-out canonical sonnets (left); top-2% candidates across the corpus (right)

![Supervised scores for play windows vs held-out canonical sonnets (left); top-2% candidates across the corpus (right)](run-2-figures/01-classifier-scores.png)

## Cell 8b: evaluation on the known play sonnets

```text
Located 20 of 22 known play sonnets in the cleaned plays.
Not in the corpus (cannot be recalled): ['Edward III, scene 2.322–3.', 'Love’s Labour’s Lost, 4.3.24–39.']
                                          label                             play  line split
        The Two Gentlemen of Verona, 3.1.140–9.      THE TWO GENTLEMEN OF VERONA  1137  test
               The Comedy of Errors, 3.2.41–54.             THE COMEDY OF ERRORS   773  tune
                Love’s Labour’s Lost, 1.1.80–9.             LOVE’S LABOUR’S LOST   101  tune
              Love’s Labour’s Lost, 1.1.159–74.             LOVE’S LABOUR’S LOST   183  test
              Love’s Labour’s Lost, 4.2.106–19.             LOVE’S LABOUR’S LOST  1132  tune
               Love’s Labour’s Lost, 4.3.57–70.             LOVE’S LABOUR’S LOST  1214  test
              Love’s Labour’s Lost, 5.2.401–15.             LOVE’S LABOUR’S LOST  2073  tune
                    Romeo and Juliet, Prologue.  THE TRAGEDY OF ROMEO AND JULIET    31  test
                  Romeo and Juliet, 1.5.92–109.  THE TRAGEDY OF ROMEO AND JULIET   700  tune
Romeo and Juliet, 2.0. Introduction to Act Two.  THE TRAGEDY OF ROMEO AND JULIET   757  test
         A Midsummer Night’s Dream, 5.1.108–17.        A MIDSUMMER NIGHT’S DREAM  1819  tune
            Much Ado About Nothing, 3.1.107–16.           MUCH ADO ABOUT NOTHING  1009  test
                             Henry V, Epilogue. THE LIFE OF KING HENRY THE FIFTH  3026  test
                      As You Like It, 3.2.1–10.                   AS YOU LIKE IT  1079  tune
              Troilus and Cressida, 1.2.278–91.             TROILUS AND CRESSIDA   421  test
        All’s Well That Ends Well, 1.1.212–225.        ALL’S WELL THAT ENDS WELL   203  test
           All’s Well That Ends Well, 3.4.4–17.        ALL’S WELL THAT ENDS WELL  1411  test
                       Pericles, scene 10.1–14.         PERICLES, PRINCE OF TYRE  1051  tune
                     Pericles, scene 21.225–35.         PERICLES, PRINCE OF TYRE  2255  tune
                        Cymbeline, 5.5.187–204.                        CYMBELINE  2984  tune

Scanning 38 plays with the rule detector (cached after the first run)...
=== Baseline: how well does each detector find the known play sonnets? ===
      detector  n_known  median_rank  recall@10  recall@25  recall@50  recall@100  recall@250 split
rule (WEIGHTS)       10         28.0        0.4        0.5        0.7         0.7         0.8  tune
rule (WEIGHTS)       10         40.0        0.3        0.4        0.6         0.6         0.8  test
supervised clf       10          inf        0.0        0.0        0.0         0.1         0.2  tune
supervised clf       10        417.5        0.0        0.1        0.1         0.2         0.4  test
        PU clf       10       1852.0        0.0        0.0        0.0         0.0         0.1  tune
        PU clf       10        391.0        0.0        0.1        0.2         0.4         0.4  test

95% bootstrap intervals over all 20 located sonnets:
  rule (WEIGHTS)   recall@50 = 0.65  [0.45, 0.85]   median rank = 40  [6, 122]
  supervised clf   recall@50 = 0.05  [0.00, 0.15]   median rank = inf  [222, inf]
  PU clf           recall@50 = 0.10  [0.00, 0.25]   median rank = 995  [270, inf]
Intervals this wide mean a difference of one or two sonnets is not a real difference.

Rank of each known sonnet (lower is better; inf = never in the hit list):
                                          label split  rule (WEIGHTS)  supervised clf  PU clf
        The Two Gentlemen of Verona, 3.1.140–9.  test            39.0           565.0  2169.0
               The Comedy of Errors, 3.2.41–54.  tune             2.0             inf     inf
                Love’s Labour’s Lost, 1.1.80–9.  tune             3.0             inf   270.0
              Love’s Labour’s Lost, 1.1.159–74.  test            14.0           270.0    20.0
              Love’s Labour’s Lost, 4.2.106–19.  tune            12.0            81.0   216.0
               Love’s Labour’s Lost, 4.3.57–70.  test             inf            21.0    29.0
              Love’s Labour’s Lost, 5.2.401–15.  tune             6.0             inf  1780.0
                    Romeo and Juliet, Prologue.  test             7.0           173.0   269.0
                  Romeo and Juliet, 1.5.92–109.  tune             1.0           114.0     inf
Romeo and Juliet, 2.0. Introduction to Act Two.  test             4.0           102.0    63.0
         A Midsummer Night’s Dream, 5.1.108–17.  tune            47.0             inf     inf
            Much Ado About Nothing, 3.1.107–16.  test           951.0             inf     inf
                             Henry V, Epilogue.  test            41.0          1149.0     inf
                      As You Like It, 3.2.1–10.  tune            44.0             inf  1924.0
              Troilus and Cressida, 1.2.278–91.  test           117.0            54.0    51.0
        All’s Well That Ends Well, 1.1.212–225.  test           220.0             inf   513.0
           All’s Well That Ends Well, 3.4.4–17.  test             5.0             inf   656.0
                       Pericles, scene 10.1–14.  tune             inf             inf  1334.0
                     Pericles, scene 21.225–35.  tune           127.0          2528.0     inf
                        Cymbeline, 5.5.187–204.  tune             inf             inf   323.0

Tuned weights (chosen on the tune half): {'meter': np.float64(0.0), 'syllable': np.float64(0.1), 'rhyme': np.float64(0.3), 'couplet': np.float64(0.6)}
  tune: recall@50 0.70 -> 0.70; median rank 28 -> 25
  test: recall@50 0.60 -> 0.60; median rank 40 -> 44
If the TEST numbers improve, copy the tuned weights into WEIGHTS in cell 4 and re-run.
```

## Cell 9: THE SONNETS analysis

**Figure 2.** Rule-detector signal along THE SONNETS

![Rule-detector signal along THE SONNETS](run-2-figures/02-sonnets-signal.png)

**Figure 3.** Scores of the 154 canonical sonnets and the 22 known play sonnets

![Scores of the 154 canonical sonnets and the 22 known play sonnets](run-2-figures/03-sonnets-and-play-sonnets-scores.png)

**Figure 4.** Sonnets ranked by score, marked by rhyme-scheme deviation

![Sonnets ranked by score, marked by rhyme-scheme deviation](run-2-figures/04-sonnets-ranked-rhyme-deviations.png)

**Figure 5.** How often each word is used in the Sonnets

![How often each word is used in the Sonnets](run-2-figures/05-sonnets-vocabulary-histogram.png)

**Figure 6.** Sonnets clustered by rare-word usage: K-Means

![Sonnets clustered by rare-word usage: K-Means](run-2-figures/06-clusters-kmeans.png)

**Figure 7.** Sonnets clustered by rare-word usage: Agglomerative (Ward)

![Sonnets clustered by rare-word usage: Agglomerative (Ward)](run-2-figures/07-clusters-ward.png)

**Figure 8.** Ward dendrogram of the sonnets

![Ward dendrogram of the sonnets](run-2-figures/08-dendrogram-ward.png)

**Figure 9.** Sonnets clustered by rare-word usage: Spectral

![Sonnets clustered by rare-word usage: Spectral](run-2-figures/09-clusters-spectral.png)

**Figure 10.** Cluster overlap by sonnets: K-Means vs Ward

![Cluster overlap by sonnets: K-Means vs Ward](run-2-figures/10-overlap-kmeans-vs-ward.png)

**Figure 11.** Cluster overlap by sonnets: K-Means vs Spectral

![Cluster overlap by sonnets: K-Means vs Spectral](run-2-figures/11-overlap-kmeans-vs-spectral.png)

**Figure 12.** Jaccard of cluster word sets: K-Means vs Ward

![Jaccard of cluster word sets: K-Means vs Ward](run-2-figures/12-jaccard-kmeans-vs-ward.png)

**Figure 13.** Jaccard of cluster word sets: K-Means vs Spectral

![Jaccard of cluster word sets: K-Means vs Spectral](run-2-figures/13-jaccard-kmeans-vs-spectral.png)

**Figure 14.** Jaccard of cluster word sets: Ward vs Spectral

![Jaccard of cluster word sets: Ward vs Spectral](run-2-figures/14-jaccard-ward-vs-spectral.png)

**Figure 15.** Silhouette vs k against the shuffled-word null (left); bootstrap stability (right)

![Silhouette vs k against the shuffled-word null (left); bootstrap stability (right)](run-2-figures/15-cluster-stability.png)

## Cell 10: Midsummer (top hits, statistics)

```text
Analyzing A MIDSUMMER NIGHT’S DREAM...

Top 10 Scoring Segments in A MIDSUMMER NIGHT’S DREAM (Distinct Regions):
========================================

Rank 1 | Score: 0.8877 | Start Line: 2058
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

Rank 2 | Score: 0.8820 | Start Line: 1851
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

Rank 3 | Score: 0.8357 | Start Line: 1061
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

Rank 4 | Score: 0.8357 | Start Line: 1155
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

Rank 7 | Score: 0.8255 | Start Line: 1041
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

Rank 9 | Score: 0.8072 | Start Line: 1116
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

Rank 10 | Score: 0.8049 | Start Line: 1819
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

Rank 12 | Score: 0.7950 | Start Line: 1355
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

Rank 13 | Score: 0.7821 | Start Line: 763
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

Rank 14 | Score: 0.7785 | Start Line: 996
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

Rank 15 | Score: 0.7745 | Start Line: 929
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

Rank 16 | Score: 0.7715 | Start Line: 1408
--------------------
[LYSANDER] The villain is much lighter-heel’d than I:
I follow’d fast, but faster he did fly,
That fallen am I in dark uneven way,
And here will rest me.  Come, thou gentle day!
For if but once thou show me thy grey light,
I’ll find Demetrius, and revenge this spite.
[PUCK] Ho, ho, ho! Coward, why com’st thou not?
[DEMETRIUS] Abide me, if thou dar’st; for well I wot
Thou runn’st before me, shifting every place,
And dar’st not stand, nor look me in the face.
Where art thou?
[PUCK] Come hither; I am here.
[DEMETRIUS] Nay, then, thou mock’st me. Thou shalt buy this dear
If ever I thy face by daylight see:

Rank 17 | Score: 0.7692 | Start Line: 1375
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

Rank 18 | Score: 0.7658 | Start Line: 1091
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

Rank 19 | Score: 0.7426 | Start Line: 396
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

Rank 20 | Score: 0.7415 | Start Line: 1333
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
Lag-1 autocorrelation of the score signal: 0.80  (0 = white noise, near 1 = strongly red)
Share of the spectrum (inside the cone of influence) called significant: white-noise test 66.6%, red-noise test 45.8%  (about 5% expected by chance)

Total unique word types in A MIDSUMMER NIGHT’S DREAM: 3007

Lines in TEST_GROUP_POETIC: 280
Lines in TEST_GROUP_NORMAL: 1515

=== RARE-WORD PERCENTAGES (TOKEN-BASED) ===
TEST_GROUP_POETIC:  total tokens = 2238
  % of tokens with frequency 1–2:  17.07%
  % of tokens with frequency 1–3:  21.98%

TEST_GROUP_NORMAL:  total tokens = 11738
  % of tokens with frequency 1–2:  15.81%
  % of tokens with frequency 1–3:  19.76%

Comparison (Poetic minus Normal):
  Δ% (1–2):  1.26 percentage points
  Δ% (1–3):  2.22 percentage points

=== NAIVE T-TESTS (treat every line as independent; kept for comparison) ===

1-use words:
  Poetic lines:  mean=0.868, std=0.947, n=280
  Normal lines:  mean=0.802, std=0.930, n=1515
  Welch t-test: t = 1.073, p = 0.2841

1–2-use words:
  Poetic lines:  mean=1.364, std=1.165, n=280
  Normal lines:  mean=1.225, std=1.095, n=1515
  Welch t-test: t = 1.853, p = 0.06462

1–3-use words:
  Poetic lines:  mean=1.757, std=1.249, n=280
  Normal lines:  mean=1.531, std=1.192, n=1515
  Welch t-test: t = 2.798, p = 0.005406

=== WINDOW PERMUTATION TEST (unit = 14-line window) ===
   1-use words: top windows 11.018% rare tokens vs random windows 10.794% ± 1.062%  ->  p (two-sided) = 0.828
 1–2-use words: top windows 17.367% rare tokens vs random windows 16.342% ± 1.335%  ->  p (two-sided) = 0.44
 1–3-use words: top windows 22.249% rare tokens vs random windows 20.544% ± 1.358%  ->  p (two-sided) = 0.219

Generating Score Distribution Comparison...

Generating Word Usage Comparison Chart...
```

**Figure 16.** Detection signal in A Midsummer Night's Dream, with classifier overlay and top hits

![Detection signal in A Midsummer Night's Dream, with classifier overlay and top hits](run-2-figures/16-midsummer-signal.png)

**Figure 17.** Wavelet power spectrum of the Midsummer signal

![Wavelet power spectrum of the Midsummer signal](run-2-figures/17-midsummer-wavelet-power.png)

**Figure 18.** Wavelet power with significance contours

![Wavelet power with significance contours](run-2-figures/18-midsummer-wavelet-significance.png)

**Figure 19.** How often each word is used in Midsummer

![How often each word is used in Midsummer](run-2-figures/19-midsummer-vocabulary-histogram.png)

**Figure 20.** Score distribution: Midsummer windows vs all sonnets

![Score distribution: Midsummer windows vs all sonnets](run-2-figures/20-midsummer-vs-sonnets-score-kde.png)

**Figure 21.** Fraction of word types by frequency: Sonnets vs Midsummer

![Fraction of word types by frequency: Sonnets vs Midsummer](run-2-figures/21-midsummer-word-usage.png)

## Temporal analysis

**Figure 22.** Sonnets on PC1 vs PC2, coloured by composition date

![Sonnets on PC1 vs PC2, coloured by composition date](run-2-figures/22-temporal-pca-pc1-pc2.png)

**Figure 23.** Feature loadings for PC1 and PC2

![Feature loadings for PC1 and PC2](run-2-figures/23-temporal-loadings-pc1-pc2.png)

**Figure 24.** Sonnets on PC3 vs PC4, coloured by composition date

![Sonnets on PC3 vs PC4, coloured by composition date](run-2-figures/24-temporal-pca-pc3-pc4.png)

**Figure 25.** Feature loadings for PC3 and PC4

![Feature loadings for PC3 and PC4](run-2-figures/25-temporal-loadings-pc3-pc4.png)

## Combined PCA

**Figure 26.** Combined PCA: temporal metrics, LSA themes and rare-word Ward clusters

![Combined PCA: temporal metrics, LSA themes and rare-word Ward clusters](run-2-figures/26-combined-pca.png)

## Word-usage chart

**Figure 27.** Fraction of word types by frequency: Sonnets and three plays

![Fraction of word types by frequency: Sonnets and three plays](run-2-figures/27-word-usage-four-works.png)

## Sonnet Finder V2 (neural mode)

```text
Advanced Sonnet Detector Ready.
Analyzing THE TRAGEDY OF ROMEO AND JULIET with Neural Detector...

Top 10 Scoring Segments in THE TRAGEDY OF ROMEO AND JULIET (Neural Detector):
========================================

Rank 1 | Score: 0.9721 | Start Line: 700
--------------------
[ROMEO] If I profane with my unworthiest hand
This holy shrine, the gentle sin is this,
My lips, two blushing pilgrims, ready stand
To smooth that rough touch with a tender kiss.
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

Rank 2 | Score: 0.9566 | Start Line: 757
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

Rank 3 | Score: 0.9026 | Start Line: 31
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

Rank 4 | Score: 0.8282 | Start Line: 297
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

Rank 5 | Score: 0.8230 | Start Line: 1015
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

Rank 6 | Score: 0.7820 | Start Line: 1088
--------------------
[FRIAR LAWRENCE] Jesu Maria, what a deal of brine
Hath wash’d thy sallow cheeks for Rosaline!
How much salt water thrown away in waste,
To 
```

**Figure 28.** Neural-mode detection signal in Romeo and Juliet

![Neural-mode detection signal in Romeo and Juliet](run-2-figures/28-romeo-juliet-neural-signal.png)

**Figure 29.** Neural-mode score distributions: Sonnets vs three plays

![Neural-mode score distributions: Sonnets vs three plays](run-2-figures/29-neural-score-distributions.png)

## Semantic coherence (validation)

```text
Validation: known play sonnets coherence 0.060 (median, n=20) vs random play windows 0.043 (n=60);  Mann-Whitney p = 0.0764
-> Coherence does NOT distinguish known sonnets from ordinary dialogue here; differences below should not be read as evidence.
Scoring 'THE SONNETS'...
Scoring 'A MIDSUMMER NIGHT’S DREAM'...
Analyzing A MIDSUMMER NIGHT’S DREAM...
Scoring 'TROILUS AND CRESSIDA'...
Analyzing TROILUS AND CRESSIDA...
                           count   mean    std    min    25%    50%    75%  \
Work                                                                         
A MIDSUMMER NIGHT’S DREAM   25.0  0.052  0.020 -0.006  0.043  0.051  0.059   
Known play sonnets          20.0  0.067  0.044  0.012  0.038  0.060  0.081   
Random play windows         60.0  0.049  0.029  0.007  0.030  0.043  0.058   
THE SONNETS                 40.0  0.077  0.026  0.032  0.056  0.078  0.086   
TROILUS AND CRESSIDA        25.0  0.051  0.025  0.019  0.036  0.047  0.057   

                             max  
Work                              
A MIDSUMMER NIGHT’S DREAM  0.108  
Known play sonnets         0.178  
Random play windows        0.139  
THE SONNETS                0.138  
TROILUS AND CRESSIDA       0.116
```

**Figure 30.** Semantic coherence by group

![Semantic coherence by group](run-2-figures/30-coherence.png)

## Stylometry

**Figure 31.** Stylometry: PCA of top-100-word counts

![Stylometry: PCA of top-100-word counts](run-2-figures/31-stylometry-pca.png)

## Distinctness (validation)

```text
Validation Semantic_Dist: known sonnets median 0.871 vs random windows 0.783  (Mann-Whitney p = 0.0017)
Validation Style_Dist: known sonnets median 0.873 vs random windows 0.843  (Mann-Whitney p = 0.000115)
If p > 0.05, this measure can't tell a known sonnet from ordinary dialogue, so it can't confirm new candidates either.

Analyzing Contextual Distinctness for: THE TRAGEDY OF ROMEO AND JULIET
Analyzing THE TRAGEDY OF ROMEO AND JULIET...

Analyzing Contextual Distinctness for: A MIDSUMMER NIGHT’S DREAM
Analyzing A MIDSUMMER NIGHT’S DREAM...

Analyzing Contextual Distinctness for: THE TRAGEDY OF HAMLET, PRINCE OF DENMARK
Analyzing THE TRAGEDY OF HAMLET, PRINCE OF DENMARK...
                                    Work  Rank  Score  Semantic_Dist  Style_Dist                                                   Snippet
         THE TRAGEDY OF ROMEO AND JULIET     1  0.969          0.814       0.862                     If I profane with my unworthiest hand
         THE TRAGEDY OF ROMEO AND JULIET     2  0.954          0.642       0.873                  Now old desire doth in his deathbed lie,
         THE TRAGEDY OF ROMEO AND JULIET     3  0.902          0.769       0.908                    Two households, both alike in dignity,
         THE TRAGEDY OF ROMEO AND JULIET     4  0.826          0.808       0.881         Sleep dwell upon thine eyes, peace in thy breast.
         THE TRAGEDY OF ROMEO AND JULIET     5  0.826          0.846       0.829                At my poor house look to behold this night
         THE TRAGEDY OF ROMEO AND JULIET     6  0.781          0.727       0.870           This precious book of love, this unbound lover,
         THE TRAGEDY OF ROMEO AND JULIET     7  0.777          0.877       0.823             O, she doth teach the torches to burn bright!
         THE TRAGEDY OF ROMEO AND JULIET     8  0.766          0.809       0.873            For naught so vile that on the earth doth live
         THE TRAGEDY OF ROMEO AND JULIET     9  0.765          1.033       0.857                      When the devout religion of mine eye
         THE TRAGEDY OF ROMEO AND JULIET    10  0.754          0.597       0.833                           At thy good heart’s oppression.
               A MIDSUMMER NIGHT’S DREAM     1  0.888          0.667       0.878                            Now the wasted brands do glow,
               A MIDSUMMER NIGHT’S DREAM     2  0.882          0.607       0.776                 And as she fled, her mantle she did fall;
               A MIDSUMMER NIGHT’S DREAM     3  0.836          0.814       0.824                   So sorrow’s heaviness doth heavier grow
               A MIDSUMMER NIGHT’S DREAM     4  0.836          0.885       0.885         Dark night, that from the eye his function takes,
               A MIDSUMMER NIGHT’S DREAM     5  0.831          0.630       0.855               Take comfort: he no more shall see my face;
               A MIDSUMMER NIGHT’S DREAM     6  0.826          0.715       0.783          Fair love, you faint with wand’ring in the wood.
               A MIDSUMMER NIGHT’S DREAM     7  0.826          0.746       0.843               I had rather give his carcass to my hounds.
               A MIDSUMMER NIGHT’S DREAM     8  0.818          0.742       0.842          Love looks not with the eyes, but with the mind;
               A MIDSUMMER NIGHT’S DREAM     9  0.807          0.866       0.847                      Crystal is muddy. O how ripe in show
               A MIDSUMMER NIGHT’S DREAM    10  0.805          0.836       0.809                   If we offend, it is with our good will.
THE TRAGEDY OF HAMLET, PRINCE OF DENMARK     1  0.828          0.849       0.860                         But orderly to end where I begun,
THE TRAGEDY OF HAMLET, PRINCE OF DENMARK     2  0.720          0.888       0.842                  But, woe is me, you are so sick of late,
THE TRAGEDY OF HAMLET, PRINCE OF DENMARK     3  0.705          0.729       0.858                   The instances that second marriage move
THE TRAGEDY OF HAMLET, PRINCE OF DENMARK     4  0.664          0.746       0.806                  Or in th’incestuous pleasure of his bed,
THE TRAGEDY OF HAMLET, PRINCE OF DENMARK     5  0.645          0.621       0.812         And from his mother’s closet hath he dragg’d him.
THE TRAGEDY OF HAMLET, PRINCE OF DENMARK     6  0.633          0.733       0.821                     Of habits evil, is angel yet in this,
THE TRAGEDY OF HAMLET, PRINCE OF DENMARK     7  0.632          0.954       0.895               And keep you in the rear of your affection,
THE TRAGEDY OF HAMLET, PRINCE OF DENMARK     8  0.632          0.714       0.841           And England, if my love thou hold’st at aught,—
THE TRAGEDY OF HAMLET, PRINCE OF DENMARK     9  0.629          0.944       0.879 The courtier’s, soldier’s, scholar’s, eye, tongue, sword,
THE TRAGEDY OF HAMLET, PRINCE OF DENMARK    10  0.629          0.874       0.856              That have a father kill’d, a mother stain’d,
        
```

**Figure 32.** Contextual distinctness of the top 10 candidates per play

![Contextual distinctness of the top 10 candidates per play](run-2-figures/32-distinctness-top10.png)

## Distinctness, top 20

**Figure 33.** Contextual distinctness of the top 20 candidates per play

![Contextual distinctness of the top 20 candidates per play](run-2-figures/33-distinctness-top20.png)
