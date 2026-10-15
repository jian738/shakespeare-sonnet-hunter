"""Known play sonnets: the 22 used for evaluation, and sonnets found later.

play_sonnet_specs / manual_play_sonnets: the 22 known play sonnets (cell 3 used to
define them). Cell 8b splits them into tune and test halves.

DISCOVERED_SONNETS: passages identified as sonnets AFTER the detector was built
(e.g. by hand, by the professor, or by the blind annotation in annotation/).
They are kept separate and are NEVER used for tuning weights or settings; the
notebook only reports where the detector ranks them. Each entry:
    {"label": "A Midsummer Night's Dream, 3.2.x", "first_line": "...",
     "num_lines": 14, "source": "who found it"}
"""

DISCOVERED_SONNETS = []

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
