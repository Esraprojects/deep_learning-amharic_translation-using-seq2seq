"""Targeted data augmentation: correct English-Amharic pairs for "play / playing",
"garden", negation and other everyday verbs (cook, eat, read, write, work, study, run).

Error analysis showed that about half of the 205 training pairs containing
"playing" have an Amharic side without the verb ጫወት (loose or misaligned
translations), and "garden" is mostly translated as ገነት ("paradise", from the
Bible). The model therefore never learned "playing" -> እየተጫወቱ ነው. Pairs for
other everyday verbs and negation are included so that the model does not
over-generalise "playing" to every action. This script
writes a small set of template-generated pairs with correct subject-verb
agreement to data/curated/play_garden.tsv; src/finetune.py mixes them into
training. Native speakers can correct or extend the TSV by hand.

    python src/augment.py
"""
import itertools
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "curated", "play_garden.tsv")

# subject: (English, English "be", Amharic subject, person key, possessive suffix key)
SUBJECTS = [
    ("The children", "are", "ልጆቹ", "they"),
    ("The boys", "are", "ወንዶቹ ልጆች", "they"),
    ("The students", "are", "ተማሪዎቹ", "they"),
    ("My friends", "are", "ጓደኞቼ", "they"),
    ("They", "are", "እነሱ", "they"),
    ("The boy", "is", "ልጁ", "he"),
    ("My brother", "is", "ወንድሜ", "he"),
    ("He", "is", "እሱ", "he"),
    ("The girl", "is", "ልጅቷ", "she"),
    ("My sister", "is", "እህቴ", "she"),
    ("She", "is", "እሷ", "she"),
    ("We", "are", "እኛ", "we"),
    ("I", "am", "እኔ", "i"),
]

# progressive "is/are playing", past "played", infinitive-ish "wants to play"
PLAYING = {"they": "እየተጫወቱ ነው", "he": "እየተጫወተ ነው", "she": "እየተጫወተች ነው",
           "we": "እየተጫወትን ነው", "i": "እየተጫወትኩ ነው"}
PLAYED = {"they": "ተጫወቱ", "he": "ተጫወተ", "she": "ተጫወተች", "we": "ተጫወትን", "i": "ተጫወትኩ"}
WANTS = {"they": "መጫወት ይፈልጋሉ", "he": "መጫወት ይፈልጋል", "she": "መጫወት ትፈልጋለች",
         "we": "መጫወት እንፈልጋለን", "i": "መጫወት እፈልጋለሁ"}
WANTS_EN = {"they": "want", "he": "wants", "she": "wants", "we": "want", "i": "want"}
FRIENDS = {"they": ("their", "ከጓደኞቻቸው ጋር"), "he": ("his", "ከጓደኞቹ ጋር"), "she": ("her", "ከጓደኞቿ ጋር"),
           "we": ("our", "ከጓደኞቻችን ጋር"), "i": ("my", "ከጓደኞቼ ጋር")}

PLACES = [
    ("in the garden", "በአትክልት ቦታው ውስጥ"),
    ("outside", "ውጭ"),
    ("at school", "ትምህርት ቤት ውስጥ"),
    ("in the house", "ቤት ውስጥ"),
    ("on the field", "ሜዳ ላይ"),
    ("near the river", "ወንዙ አጠገብ"),
]
GAMES = [
    ("football", "እግር ኳስ"),
    ("basketball", "የቅርጫት ኳስ"),
    ("cards", "ካርታ"),
    ("a game", "ጨዋታ"),
]
GARDEN = [
    ("The garden is beautiful.", "የአትክልት ቦታው ያምራል።"),
    ("The garden is big.", "የአትክልት ቦታው ትልቅ ነው።"),
    ("There are flowers in the garden.", "በአትክልት ቦታው ውስጥ አበቦች አሉ።"),
    ("My mother works in the garden.", "እናቴ በአትክልት ቦታው ውስጥ ትሰራለች።"),
    ("My father works in the garden.", "አባቴ በአትክልት ቦታው ውስጥ ይሰራል።"),
    ("We have a small garden.", "ትንሽ የአትክልት ቦታ አለን።"),
    ("The garden is behind the house.", "የአትክልት ቦታው ከቤቱ ጀርባ ነው።"),
    ("Children like to play.", "ልጆች መጫወት ይወዳሉ።"),
    ("Let us play.", "እንጫወት።"),
    ("Do not play in the street.", "መንገድ ላይ አትጫወቱ።"),
]


NOT_PLAYING = {"they": "እየተጫወቱ አይደሉም", "he": "እየተጫወተ አይደለም", "she": "እየተጫወተች አይደለችም",
               "we": "እየተጫወትን አይደለንም", "i": "እየተጫወትኩ አይደለሁም"}

# other everyday verbs: English -ing form, Amharic progressive by person, objects
VERBS = [
    ("cooking", {"they": "እያበሰሉ ነው", "he": "እያበሰለ ነው", "she": "እያበሰለች ነው", "we": "እያበሰልን ነው", "i": "እያበሰልኩ ነው"},
     [("dinner", "እራት"), ("food", "ምግብ"), ("lunch", "ምሳ")]),
    ("eating", {"they": "እየበሉ ነው", "he": "እየበላ ነው", "she": "እየበላች ነው", "we": "እየበላን ነው", "i": "እየበላሁ ነው"},
     [("bread", "ዳቦ"), ("lunch", "ምሳ"), ("dinner", "እራት")]),
    ("reading", {"they": "እያነበቡ ነው", "he": "እያነበበ ነው", "she": "እያነበበች ነው", "we": "እያነበብን ነው", "i": "እያነበብኩ ነው"},
     [("a book", "መጽሐፍ"), ("the newspaper", "ጋዜጣ")]),
    ("writing", {"they": "እየጻፉ ነው", "he": "እየጻፈ ነው", "she": "እየጻፈች ነው", "we": "እየጻፍን ነው", "i": "እየጻፍኩ ነው"},
     [("a letter", "ደብዳቤ")]),
    ("working", {"they": "እየሰሩ ነው", "he": "እየሰራ ነው", "she": "እየሰራች ነው", "we": "እየሰራን ነው", "i": "እየሰራሁ ነው"}, []),
    ("studying", {"they": "እያጠኑ ነው", "he": "እያጠና ነው", "she": "እያጠናች ነው", "we": "እያጠናን ነው", "i": "እያጠናሁ ነው"}, []),
    ("running", {"they": "እየሮጡ ነው", "he": "እየሮጠ ነው", "she": "እየሮጠች ነው", "we": "እየሮጥን ነው", "i": "እየሮጥኩ ነው"}, []),
]
VERB_SUBJECTS = SUBJECTS + [("My mother", "is", "እናቴ", "she"), ("My father", "is", "አባቴ", "he")]
VERB_PLACES = [("at home", "ቤት ውስጥ"), ("in the kitchen", "ማዕድ ቤት ውስጥ"), ("at school", "ትምህርት ቤት ውስጥ"),
               ("in the garden", "በአትክልት ቦታው ውስጥ")]
FAMILY = {"they": "ለቤተሰባቸው", "he": "ለቤተሰቡ", "she": "ለቤተሰቧ", "we": "ለቤተሰባችን", "i": "ለቤተሰቤ"}


def generate():
    pairs = []
    for (en_s, be, am_s, p), (en_pl, am_pl) in itertools.product(SUBJECTS, PLACES):
        pairs.append((f"{en_s} {be} playing {en_pl}.", f"{am_s} {am_pl} {PLAYING[p]}።"))
        pairs.append((f"{en_s} played {en_pl}.", f"{am_s} {am_pl} {PLAYED[p]}።"))
    for (en_s, be, am_s, p), (en_g, am_g) in itertools.product(SUBJECTS, GAMES):
        pairs.append((f"{en_s} {be} playing {en_g}.", f"{am_s} {am_g} {PLAYING[p]}።"))
    for en_s, be, am_s, p in SUBJECTS:
        poss, am_f = FRIENDS[p]
        pairs.append((f"{en_s} {be} playing.", f"{am_s} {PLAYING[p]}።"))
        pairs.append((f"{en_s} {be} playing with {poss} friends.", f"{am_s} {am_f} {PLAYING[p]}።"))
        pairs.append((f"{en_s} {WANTS_EN[p]} to play.", f"{am_s} {WANTS[p]}።"))
        pairs.append((f"{en_s} {be} playing with {poss} friends in the garden.",
                      f"{am_s} {am_f} በአትክልት ቦታው ውስጥ {PLAYING[p]}።"))
    for en_s, be, am_s, p in SUBJECTS:
        pairs.append((f"{en_s} {be} not playing.", f"{am_s} {NOT_PLAYING[p]}።"))
        pairs.append((f"{en_s} {be} not playing in the garden.", f"{am_s} በአትክልት ቦታው ውስጥ {NOT_PLAYING[p]}።"))
    for en_s, be, am_s, p in VERB_SUBJECTS:
        for ing, forms, objs in VERBS:
            pairs.append((f"{en_s} {be} {ing}.", f"{am_s} {forms[p]}።"))
            for en_o, am_o in objs:
                pairs.append((f"{en_s} {be} {ing} {en_o}.", f"{am_s} {am_o} {forms[p]}።"))
            for en_pl, am_pl in VERB_PLACES[:2]:
                pairs.append((f"{en_s} {be} {ing} {en_pl}.", f"{am_s} {am_pl} {forms[p]}።"))
        cook = VERBS[0][1][p]
        pairs.append((f"{en_s} {be} cooking dinner for the family.", f"{am_s} {FAMILY[p]} እራት {cook}።"))
    pairs += GARDEN
    return pairs


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    pairs = generate()
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("en\tam\n")
        for en, am in pairs:
            f.write(f"{en}\t{am}\n")
    print(f"wrote {len(pairs)} pairs to {OUT}")


if __name__ == "__main__":
    main()
