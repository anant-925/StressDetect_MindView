"""
utils/text_preprocessing.py
============================
Industry-grade text preprocessing pipeline for stress detection.

Operations (applied in order)
------------------------------
1. HTML entity unescaping  (``&amp;`` → ``&``, ``&#39;`` → ``'``)
2. HTML tag stripping       (``<b>hello</b>`` → ``hello``)
3. Emoji-to-text mapping    (``😰`` → ``anxious``)
4. URL removal              (``https://…`` → space)
5. E-mail removal           (``user@host.com`` → space)
6. Repeated-char compression (``"soooo"`` → ``"soo"``)
7. Unicode NFKC normalisation (ligatures, half-width chars, etc.)
8. Whitespace normalisation  (collapse runs, strip leading/trailing)

Design notes
------------
- All operations are conservative: they strip noise without destroying
  meaningful content.
- No external dependencies beyond the Python standard library.
- ``clean_text`` is the primary public function.  It is called by both the
  training pipeline (``training/train.py``) and the inference API
  (``api/main.py``) to ensure **identical preprocessing** at train and
  inference time — a critical requirement for consistent predictions.
"""

from __future__ import annotations

import html
import re
import unicodedata

# ---------------------------------------------------------------------------
# Compiled regex patterns
# ---------------------------------------------------------------------------

# Capped tag length (<200 chars) to guard against ReDoS on malformed HTML.
_HTML_TAG_RE = re.compile(r"<[^>]{0,200}>")
_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_EMAIL_RE = re.compile(
    r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"
)
# Compress 4+ repetitions of the same character ("sooooo" → "soo")
_REPEATED_CHAR_RE = re.compile(r"(.)\1{3,}")
_WHITESPACE_RE = re.compile(r"\s+")

# ---------------------------------------------------------------------------
# Emoji → text mapping
# Covers the most common emojis appearing in stress-related social-media text.
# Each emoji is replaced by a semantically equivalent English word so that the
# model's hash-based tokeniser can process it.
# ---------------------------------------------------------------------------

_EMOJI_TEXT: dict[str, str] = {
    # Positive / happy
    "😊": "happy",
    "🙂": "happy",
    "😀": "happy",
    "😁": "happy",
    "😄": "happy",
    "😃": "happy",
    "😆": "laughing",
    "😂": "laughing",
    "🤣": "laughing",
    "🥳": "happy",
    "🎉": "happy",
    "😌": "calm",
    "😍": "love",
    "🥰": "love",
    # Sad / emotional
    "🥲": "emotional",
    "😢": "sad",
    "😭": "crying",
    "😔": "sad",
    "😞": "sad",
    "😟": "sad",
    "💔": "heartbroken",
    # Stress / anxiety
    "😰": "anxious",
    "😨": "scared",
    "😱": "terrified",
    "😓": "stressed",
    "😥": "upset",
    "🤯": "overwhelmed",
    "😤": "frustrated",
    "😡": "angry",
    "🤬": "angry",
    # Tired / sick
    "😫": "exhausted",
    "😩": "tired",
    "😴": "tired",
    "🤕": "hurt",
    "🤒": "sick",
    # Other
    "😕": "confused",
    "😶": "speechless",
    "🥺": "hopeful",
    "❤️": "love",
    "💪": "strong",
    "🙏": "grateful",
    "👍": "good",
    "👎": "bad",
    "🆘": "help",
    "⚡": "stressed",
    "💀": "dying",
    "🔥": "intense",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def clean_text(text: str, *, normalize_repeated: bool = True) -> str:
    """Clean and normalise text for model input.

    Parameters
    ----------
    text : str
        Raw user input (social-media post, journal entry, chat message …).
    normalize_repeated : bool
        When ``True`` (default), compress long repeated-character runs
        (e.g. ``"soooooo"`` → ``"soo"``).

    Returns
    -------
    str
        Cleaned text ready for tokenisation, or an empty string when the
        input is empty or not a string.
    """
    if not text or not isinstance(text, str):
        return ""

    # 1. HTML entity unescaping
    text = html.unescape(text)

    # 2. Strip HTML/XML tags
    text = _HTML_TAG_RE.sub(" ", text)

    # 3. Emoji → text  (iterate over a snapshot so replacements don't loop)
    for emoji_char, replacement in _EMOJI_TEXT.items():
        if emoji_char in text:
            text = text.replace(emoji_char, f" {replacement} ")

    # 4. Remove URLs
    text = _URL_RE.sub(" ", text)

    # 5. Remove e-mail addresses
    text = _EMAIL_RE.sub(" ", text)

    # 6. Repeated-character compression
    if normalize_repeated:
        text = _REPEATED_CHAR_RE.sub(r"\1\1", text)

    # 7. Unicode NFKC normalisation
    text = unicodedata.normalize("NFKC", text)

    # 8. Collapse whitespace
    text = _WHITESPACE_RE.sub(" ", text).strip()

    return text
