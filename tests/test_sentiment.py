"""
tests/test_sentiment.py
=======================
Tests for the positive-sentiment dampening utility (utils/sentiment.py).
"""

import pytest

from utils.sentiment import compute_sentiment_dampening


# ---------------------------------------------------------------------------
# Clearly positive text → dampening factor < 1.0
# ---------------------------------------------------------------------------


class TestPositiveTextDampened:
    """Positive / happy sentences must produce a dampening factor well below 1."""

    @pytest.mark.parametrize(
        "text",
        [
            "I am happy",
            "I feel great",
            "I love her and she loves me back",
            "I am so grateful for everything",
            "Today was a wonderful day",
            "I feel calm and content today",
            "Life is beautiful",
            "I'm excited about the future",
            "I feel blessed and thankful",
            "I'm having a fantastic time",
            "I couldn't be happier right now",
            "I feel peaceful and relaxed",
            "I'm so proud of myself",
            "I laughed so much today",
            "I love my life",
            "I am in a great mood",
            "What an excellent day",
            "I feel incredible today",
            "This is a perfect moment",
            "I'm having a great day",
            "It's a great time to be alive",
            "I feel brilliant about my progress",
            "I'm in a wonderful mood",
            "Today is a great morning",
        ],
    )
    def test_positive_text_gets_dampened(self, text: str) -> None:
        factor = compute_sentiment_dampening(text)
        assert factor < 1.0, f"Expected dampening < 1.0, got {factor} for: {text!r}"

    @pytest.mark.parametrize(
        "text",
        [
            "I am happy",
            "I feel great",
            "I love her and she loves me back",
            "I am in a great mood",
            "What an excellent day",
        ],
    )
    def test_positive_text_makes_stress_below_10_percent(self, text: str) -> None:
        """Even if the model outputs 100 % stress probability, the dampened
        score must stay below 10 %."""
        factor = compute_sentiment_dampening(text)
        worst_case_score = 1.0 * factor
        assert worst_case_score < 0.10, (
            f"Dampened worst-case stress score {worst_case_score:.2%} >= 10 % "
            f"for: {text!r}"
        )


# ---------------------------------------------------------------------------
# Negative / stress text → no dampening (factor == 1.0)
# ---------------------------------------------------------------------------


class TestNegativeTextNotDampened:
    """Stressed or negative sentences must NOT be dampened."""

    @pytest.mark.parametrize(
        "text",
        [
            "I am so stressed about work",
            "I feel anxious and overwhelmed",
            "I can't sleep at night",
            "I am worried about my debt",
            "I feel depressed and lonely",
            "I hate everything about my life",
            "I'm scared I will fail the exam",
            "I want to give up",
            "I feel hopeless",
            "I am so frustrated and angry",
        ],
    )
    def test_negative_text_not_dampened(self, text: str) -> None:
        factor = compute_sentiment_dampening(text)
        assert factor == 1.0, (
            f"Expected factor 1.0 (no dampening), got {factor} for: {text!r}"
        )


# ---------------------------------------------------------------------------
# Mixed sentiment → no dampening (negative takes precedence)
# ---------------------------------------------------------------------------


class TestMixedSentiment:
    """When both positive and negative signals are present, negative wins."""

    @pytest.mark.parametrize(
        "text",
        [
            "I'm happy but also really stressed",
            "I love my family but I'm so worried about money",
            "Great weather but I feel anxious",
            "I feel grateful but exhausted and burnt out",
        ],
    )
    def test_mixed_sentiment_not_dampened(self, text: str) -> None:
        factor = compute_sentiment_dampening(text)
        assert factor == 1.0, (
            f"Expected factor 1.0 for mixed text, got {factor} for: {text!r}"
        )


# ---------------------------------------------------------------------------
# Neutral text → no dampening (factor == 1.0)
# ---------------------------------------------------------------------------


class TestNeutralText:
    """Neutral text without positive or negative indicators → no dampening."""

    @pytest.mark.parametrize(
        "text",
        [
            "The meeting is at 3 PM",
            "I went to the store",
            "The weather is cloudy",
            "I need to finish the report",
            "",
            "   ",
        ],
    )
    def test_neutral_text_not_dampened(self, text: str) -> None:
        factor = compute_sentiment_dampening(text)
        assert factor == 1.0, (
            f"Expected factor 1.0 for neutral text, got {factor} for: {text!r}"
        )


# ---------------------------------------------------------------------------
# Graduated dampening
# ---------------------------------------------------------------------------


class TestGraduatedDampening:
    """More positive indicators → stronger dampening."""

    def test_single_positive_word(self) -> None:
        factor = compute_sentiment_dampening("happy")
        assert 0.0 < factor <= 0.08

    def test_multiple_positive_words(self) -> None:
        factor = compute_sentiment_dampening("I am happy and grateful")
        assert factor <= 0.05

    def test_many_positive_words(self) -> None:
        factor = compute_sentiment_dampening(
            "I am happy, grateful, and excited about life"
        )
        assert factor <= 0.03


# ---------------------------------------------------------------------------
# Negation-aware dampening (negated stress phrases are treated as positive)
# ---------------------------------------------------------------------------


class TestNegationAwareDampening:
    """Phrases that negate a stress keyword should produce dampening < 1.0."""

    @pytest.mark.parametrize(
        "text",
        [
            "I am not stressed at all",
            "I don't feel anxious",
            "I no longer feel worried",
            "I am not feeling stressed today",
            "I don't feel stressed about this",
            "I am no longer anxious",
        ],
    )
    def test_negated_stress_phrase_is_dampened(self, text: str) -> None:
        factor = compute_sentiment_dampening(text)
        assert factor < 1.0, (
            f"Expected dampening < 1.0 for negated stress phrase, "
            f"got {factor} for: {text!r}"
        )

    @pytest.mark.parametrize(
        "text",
        [
            "I am so stressed",
            "I feel anxious and overwhelmed",
            "I am worried about everything",
        ],
    )
    def test_genuine_stress_phrases_not_dampened(self, text: str) -> None:
        factor = compute_sentiment_dampening(text)
        assert factor == 1.0, (
            f"Expected factor 1.0 for genuine stress phrase, "
            f"got {factor} for: {text!r}"
        )

    def test_single_negation_dampens_more_than_none(self) -> None:
        negated = compute_sentiment_dampening("I am not stressed")
        genuine = compute_sentiment_dampening("I am stressed")
        assert negated < genuine

    def test_multiple_negations_dampen_more_than_single(self) -> None:
        single = compute_sentiment_dampening("I am not stressed")
        double = compute_sentiment_dampening("I am not stressed and not anxious")
        assert double <= single
