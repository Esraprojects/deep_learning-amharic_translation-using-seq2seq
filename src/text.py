"""Text normalization and (pre-)tokenization shared by training, evaluation,
the API/Gradio app and (ported 1:1 to JavaScript) the browser demo.

Pipeline for every sentence:
    raw text -> normalize_* -> pretokenize (split punctuation) -> SentencePiece

Keep this file and web/tokenizer.js in sync.
"""
import re
import unicodedata

# ---------------------------------------------------------------------------
# English
# ---------------------------------------------------------------------------
_EN_QUOTES = {
    "‘": "'", "’": "'", "‚": "'", "′": "'", "`": "'",
    "“": '"', "”": '"', "„": '"', "«": '"', "»": '"',
    "–": "-", "—": "-", "‒": "-", "−": "-",
    "…": "...",
}
# Every character that is not a letter/digit/space becomes its own token.
_PUNCT_RE = re.compile(r"([^\w\s])", re.UNICODE)
_SPACE_RE = re.compile(r"\s+")


def normalize_en(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    for a, b in _EN_QUOTES.items():
        text = text.replace(a, b)
    text = text.replace("_", " ")
    text = text.lower()
    text = _PUNCT_RE.sub(r" \1 ", text)
    return _SPACE_RE.sub(" ", text).strip()


# ---------------------------------------------------------------------------
# Amharic (Ge'ez / Ethiopic script)
# ---------------------------------------------------------------------------
def _family(src: str, dst: str) -> dict:
    return {s: d for s, d in zip(src, dst)}


# Homophone normalization commonly applied in Amharic NLP: characters that are
# pronounced identically in modern Amharic are mapped to one canonical form.
_AM_CHAR_MAP = {}
_AM_CHAR_MAP.update(_family("ሐሑሒሓሔሕሖ", "ሀሁሂሀሄህሆ"))
_AM_CHAR_MAP.update(_family("ኀኁኂኃኄኅኆ", "ሀሁሂሀሄህሆ"))
_AM_CHAR_MAP.update(_family("ሠሡሢሣሤሥሦ", "ሰሱሲሳሴስሶ"))
_AM_CHAR_MAP.update(_family("ዐዑዒዓዔዕዖ", "አኡኢአኤእኦ"))
_AM_CHAR_MAP.update(_family("ፀፁፂፃፄፅፆ", "ጸጹጺጻጼጽጾ"))
_AM_CHAR_MAP.update({"ሃ": "ሀ", "ኣ": "አ", "ቊ": "ቁ", "ኵ": "ኩ", "ጒ": "ጉ"})
_AM_TRANS = str.maketrans(_AM_CHAR_MAP)

_AM_FULLSTOP_RE = re.compile(r"(፡\s*፡|:\s*:)")
_AM_PUNCT_RE = re.compile(r"([።፣፤፥፦፧፨!?.,;:\"'()\[\]{}«»/\\*<>|-])")


def normalize_am(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    for a, b in _EN_QUOTES.items():
        text = text.replace(a, b)
    text = _AM_FULLSTOP_RE.sub("።", text)
    text = text.translate(_AM_TRANS)
    text = _AM_PUNCT_RE.sub(r" \1 ", text)
    return _SPACE_RE.sub(" ", text).strip()


_AM_NO_SPACE_BEFORE = re.compile(r"\s+([።፣፤፥፦፧!?.,;:)\]])")
_AM_NO_SPACE_AFTER = re.compile(r"([(\[])\s+")


def detokenize_am(text: str) -> str:
    """Turn space-separated model output into readable Amharic."""
    text = _AM_NO_SPACE_BEFORE.sub(r"\1", text)
    text = _AM_NO_SPACE_AFTER.sub(r"\1", text)
    return text.strip()


ETHIOPIC_RE = re.compile(r"[ሀ-፿]")
LATIN_RE = re.compile(r"[A-Za-z]")
