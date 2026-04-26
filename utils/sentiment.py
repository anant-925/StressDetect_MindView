"""
utils/sentiment.py
==================
Lightweight keyword-based positive-sentiment detector.

Used as a post-processing correction layer during inference to prevent
clearly positive / happy text from being mislabelled as stressed.

The detector counts positive and negative sentiment indicators in the
input text and returns a dampening factor in [0, 1].  When multiplied
by the raw model stress probability, the result ensures that sentences
like *"I am happy"* or *"I love her and she loves me back"* receive
stress scores well below 10 %.

It also detects *negated* stress phrases such as *"not stressed"* or
*"don't feel anxious"* and applies a moderate dampening factor so that
explicit denials of stress are not treated as stress indicators.

Design
------
- Only *dampens* — never *inflates* — the model score.
- Requires a strong positive signal **and** the absence of negative /
  stress indicators before applying any correction.
- Keeps a generous margin so that ambiguous or mixed-sentiment text is
  left to the model.
- Negated-stress detection replaces matched phrases with a placeholder
  before the negative-hit scan so that "not stressed" does not block
  dampening the way genuine stress words would.
"""

from __future__ import annotations

import re


# ---------------------------------------------------------------------------
# Positive-sentiment indicators (case-insensitive, word-boundary)
# ---------------------------------------------------------------------------

_POSITIVE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(
        r"\b("
        r"happy|happiness|joy|joyful|joyous|elated|bliss|blissful"
        r"|cheerful|delighted|delightful|ecstatic|euphoric"
        r"|glad|pleased|wonderful|amazing|awesome|fantastic|fabulous"
        r"|terrific|magnificent|marvelous|superb|splendid"
        r"|great|excellent|incredible|brilliant|outstanding|perfect"
        r"|phenomenal|glorious|enjoyable|pleasant|overjoyed|jubilant"
        r"|love|loved|loving|adore|adored|cherish|cherished"
        r"|grateful|thankful|blessed|fortunate|lucky"
        r"|optimistic|hopeful|enthusiastic|excited|thrilled"
        r"|content|contented|satisfied|fulfilled|peaceful"
        r"|proud|confident|empowered|inspired|motivated"
        r"|relaxed|calm|serene|tranquil|comfortable"
        r"|smile|smiling|smiled|laugh|laughing|laughed|grin|grinning|grinned"
        r"|celebrate|celebrating|celebrated|celebration"
        r"|beautiful|gorgeous"
        r")\b",
        re.IGNORECASE,
    ),
]

_POSITIVE_PHRASES: list[re.Pattern[str]] = [
    re.compile(
        r"\b("
        r"feel(?:s|ing)?\s+(?:great|good|amazing|wonderful|fantastic|awesome|fine|nice|blessed|happy|excellent|incredible|brilliant)"
        r"|love\s+(?:my|this|her|him|them|it|life)"
        r"|loves?\s+me"
        r"|(?:good|great|wonderful|amazing|fantastic|awesome|excellent)\s+(?:day|time|mood|news|life|morning|evening|night)"
        r"|having\s+(?:a\s+)?(?:great|good|wonderful|amazing|fantastic|blast|ball)"
        r"|(?:in|into)\s+(?:a\s+)?(?:great|good|wonderful|amazing|fantastic|awesome|excellent)\s+mood"
        r"|so\s+(?:happy|glad|grateful|thankful|excited|thrilled|proud|pleased)"
        r"|life\s+is\s+(?:good|great|beautiful|wonderful|amazing)"
        r"|couldn['\u2019]?t\s+be\s+(?:happier|better)"
        r"|on\s+top\s+of\s+the\s+world"
        r"|over\s+the\s+moon"
        r"|best\s+(?:day|time|thing)"
        r")\b",
        re.IGNORECASE,
    ),
]

# ---------------------------------------------------------------------------
# Negated-stress patterns — stress keywords explicitly preceded by a negator.
# These are detected BEFORE the negative-hit scan and replaced with a
# placeholder so that "not stressed" is treated as a positive/neutral
# signal rather than a stress indicator.
# ---------------------------------------------------------------------------

_NEGATED_STRESS_PATTERNS: list[re.Pattern[str]] = [
    re.compile(
        r"\b(?:not|no\s+longer|don['\u2019]?t|doesn['\u2019]?t|isn['\u2019]?t"
        r"|aren['\u2019]?t|wasn['\u2019]?t|weren['\u2019]?t|haven['\u2019]?t"
        r"|hasn['\u2019]?t|hadn['\u2019]?t|won['\u2019]?t|wouldn['\u2019]?t"
        r"|can['\u2019]?t|less|barely|hardly|never)\s+"
        r"(?:\w+\s+){0,3}"
        r"(?:stress(?:ed|ful|ing)?|anxious|anxiety|worried|worrying|worry"
        r"|depress(?:ed|ion|ing)?|overwhelm(?:ed|ing)?|panic(?:king)?"
        r"|scared|afraid|fear(?:ful)?|exhausted|nervous|tense)\b",
        re.IGNORECASE,
    ),
]

# ---------------------------------------------------------------------------
# Negative / stress indicators — if ANY of these are present we leave
# the model score alone (even when positive words also appear).
# ---------------------------------------------------------------------------

_NEGATIVE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(
        r"\b("
        r"stress(?:ed|ful|ing)?|anxious|anxiety|worried|worrying|worry"
        r"|depress(?:ed|ion|ing)?|sad|sadness|miserable|unhappy"
        r"|angry|anger|furious|frustrated|frustration|irritated"
        r"|scared|afraid|fear(?:ful)?|terrified|panic(?:king)?"
        r"|overwhelm(?:ed|ing)?|exhausted|burnt?\s*out"
        r"|hopeless|helpless|desperate|despair"
        r"|lonely|isolated|alone|abandoned"
        r"|hate|hating|loathe|detest|resent"
        r"|cry(?:ing)?|sob(?:bing)?|tears|weep(?:ing)?"
        r"|suffer(?:ing)?|pain(?:ful)?|hurt(?:ing)?"
        r"|fail(?:ed|ing|ure)?|ruin(?:ed)?"
        r"|can['\u2019]?t\s+(?:take|handle|cope|stand|bear|sleep|breathe)"
        r"|don['\u2019]?t\s+(?:know\s+what\s+to\s+do|want\s+to)"
        r"|give\s+up|giving\s+up"
        r"|breakdown|break\s+down|falling\s+apart"
        r"|insomnia|nightmare|nightmares"
        r"|debt|bankrupt|fired|layoff"
        r"|suicide|suicidal|self[- ]?harm"
        r"|deadline[s]?|overdue"
        r"|sleep\s+depriv(?:ed|ation)|sleepless(?:ness)?"
        r"|haven['\u2019]?t\s+slept|not\s+slept|no\s+sleep|no\s+rest"
        r"|piling\s+up|pile\s+up|buried\s+(?:in|under)"
        r"|falling\s+behind|can['\u2019]?t\s+keep\s+up"
        r"|wearing\s+(?:me\s+)?down|breaking\s+down|worn\s+out"
        r"|drowning\s+in|stretched\s+thin|at\s+my\s+(?:limit|breaking\s+point)"
        r"|running\s+on\s+(?:empty|no\s+sleep|fumes)"
        r")\b",
        re.IGNORECASE,
    ),
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_sentiment_dampening(text: str) -> float:
    """Return a dampening factor in ``[0.0, 1.0]`` for the stress score.

    * ``1.0`` → no change (leave the model score as-is).
    * Values ``< 1.0`` → reduce (dampen) the stress score.

    The factor is only lowered when:
    1. Negated stress phrases are found (e.g. "not stressed"), OR
    2. At least one positive indicator is found **and** zero genuine
       negative/stress indicators remain.

    Parameters
    ----------
    text : str
        The raw user input.

    Returns
    -------
    float
        Multiplicative dampening factor for the stress probability.
    """
    if not text or not text.strip():
        return 1.0

    text_lower = text.lower()

    # ── Negation detection ──
    # Replace negated stress phrases with a neutral placeholder so they
    # are not counted as genuine stress indicators below.
    processed = text_lower
    negation_hits = 0
    for pat in _NEGATED_STRESS_PATTERNS:
        matches = pat.findall(processed)
        negation_hits += len(matches)
        processed = pat.sub("__negated__", processed)

    # Count genuine (non-negated) negative hits on the processed text.
    negative_hits = 0
    for pat in _NEGATIVE_PATTERNS:
        negative_hits += len(pat.findall(processed))

    if negative_hits > 0:
        # Genuine stress indicators remain → leave model score unchanged.
        return 1.0

    # Count positive hits (single words + phrases) on the original text.
    positive_hits = 0
    for pat in _POSITIVE_PATTERNS:
        positive_hits += len(pat.findall(text_lower))
    for pat in _POSITIVE_PHRASES:
        positive_hits += len(pat.findall(text_lower))

    # ── Negation-based dampening ──
    # When stress keywords are explicitly negated and no genuine stress
    # indicators remain, apply a moderate dampening factor.
    if negation_hits > 0:
        if positive_hits >= 1:
            # Negated stress + positive words → strong dampening.
            return 0.06
        # Negated stress alone → moderate dampening.
        return 0.35 if negation_hits == 1 else 0.22

    if positive_hits == 0:
        return 1.0

    # ── Standard positive dampening ──
    # 1 positive hit  → factor 0.08  (score capped at ~8 % of raw)
    # 2 positive hits → factor 0.05
    # 3+ positive hits → factor 0.03
    if positive_hits >= 3:
        return 0.03
    if positive_hits >= 2:
        return 0.05
    return 0.08


def get_sentiment_score(text: str) -> float:
    """Return a sentiment score in ``[0.0, 1.0]`` for a piece of text.

    * ``0.0`` → strongly positive (no stress signals).
    * ``1.0`` → strongly negative / stressful.
    * ``0.5`` → neutral (no clear signal either way).

    The score is derived from the same keyword lists used by
    :func:`compute_sentiment_dampening`.
    """
    if not text or not text.strip():
        return 0.5

    text_lower = text.lower()

    negative_hits = 0
    for pat in _NEGATIVE_PATTERNS:
        negative_hits += len(pat.findall(text_lower))

    positive_hits = 0
    for pat in _POSITIVE_PATTERNS:
        positive_hits += len(pat.findall(text_lower))
    for pat in _POSITIVE_PHRASES:
        positive_hits += len(pat.findall(text_lower))

    total = positive_hits + negative_hits
    if total == 0:
        return 0.5

    # Map to [0, 1]: more negative → closer to 1.0
    return negative_hits / total
