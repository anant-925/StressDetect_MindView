"""
tests/test_text_preprocessing.py
==================================
Unit tests for utils/text_preprocessing.py — clean_text.
"""

import pytest

from utils.text_preprocessing import clean_text


class TestCleanTextBasic:
    def test_empty_string_returns_empty(self):
        assert clean_text("") == ""

    def test_none_returns_empty(self):
        assert clean_text(None) == ""  # type: ignore[arg-type]

    def test_non_string_returns_empty(self):
        assert clean_text(123) == ""  # type: ignore[arg-type]

    def test_plain_text_unchanged(self):
        text = "I am feeling stressed today"
        assert clean_text(text) == text

    def test_whitespace_only_returns_stripped(self):
        assert clean_text("   ") == ""


class TestURLRemoval:
    def test_http_url_removed(self):
        result = clean_text("check out https://example.com for info")
        assert "https://" not in result
        assert "example.com" not in result

    def test_www_url_removed(self):
        result = clean_text("visit www.example.com today")
        assert "www.example.com" not in result

    def test_text_around_url_preserved(self):
        result = clean_text("check out https://example.com for info")
        assert "check" in result
        assert "info" in result


class TestEmailRemoval:
    def test_email_removed(self):
        result = clean_text("contact me at user@example.com please")
        assert "user@example.com" not in result

    def test_text_around_email_preserved(self):
        result = clean_text("contact me at user@example.com please")
        assert "contact" in result
        assert "please" in result


class TestHTMLStripping:
    def test_bold_tag_removed(self):
        result = clean_text("<b>hello</b> world")
        assert "<b>" not in result
        assert "</b>" not in result
        assert "hello" in result

    def test_paragraph_tag_removed(self):
        result = clean_text("<p>stressed today</p>")
        assert "<p>" not in result
        assert "stressed" in result

    def test_html_entity_unescaped(self):
        result = clean_text("I &amp; my team are done")
        assert "&amp;" not in result
        assert "&" in result

    def test_numeric_entity_unescaped(self):
        result = clean_text("it&#39;s over")
        assert "&#39;" not in result
        assert "it" in result


class TestEmojiNormalization:
    def test_happy_emoji_replaced_with_text(self):
        result = clean_text("feeling 😊 today")
        assert "😊" not in result
        assert "happy" in result.lower()

    def test_crying_emoji_replaced_with_text(self):
        result = clean_text("I am 😭 all day")
        assert "😭" not in result
        assert "crying" in result.lower()

    def test_anxious_emoji_replaced_with_text(self):
        result = clean_text("so 😰 about the exam")
        assert "😰" not in result
        assert "anxious" in result.lower()

    def test_multiple_emojis_all_replaced(self):
        result = clean_text("😊😭")
        assert "😊" not in result
        assert "😭" not in result


class TestRepeatedCharNormalization:
    def test_excessive_repetition_compressed(self):
        result = clean_text("sooooooo tired")
        # "sooooooo" should be compressed to "soo" (max 2 repetitions)
        assert "oooooooo" not in result

    def test_normal_repetition_preserved(self):
        # Three repetitions or fewer remain
        result = clean_text("noo way")
        assert "noo" in result

    def test_repeated_exclamation_compressed(self):
        result = clean_text("great!!!!!")
        assert "!!!!!" not in result


class TestWhitespaceNormalization:
    def test_multiple_spaces_collapsed(self):
        result = clean_text("I   am   fine")
        assert "   " not in result

    def test_tabs_collapsed(self):
        result = clean_text("I\tam\tfine")
        assert "\t" not in result

    def test_leading_trailing_stripped(self):
        result = clean_text("  hello world  ")
        assert result == "hello world"

    def test_newlines_collapsed(self):
        result = clean_text("line one\n\nline two")
        assert "\n\n" not in result


class TestUnicodeNormalization:
    def test_half_width_chars_normalised(self):
        # Full-width digit '１' should normalise to '1'
        result = clean_text("\uff11 stress")
        assert "\uff11" not in result

    def test_ligature_normalised(self):
        # ﬁ (fi ligature) → fi
        result = clean_text("ﬁne")
        assert "ﬁ" not in result


class TestNormalizeRepeatedFlag:
    def test_flag_off_preserves_repetition(self):
        result = clean_text("sooooooo tired", normalize_repeated=False)
        assert "oooooooo" in result

    def test_flag_on_compresses(self):
        result = clean_text("sooooooo tired", normalize_repeated=True)
        assert "oooooooo" not in result


class TestCombinedCleaning:
    def test_html_url_emoji_combined(self):
        text = '<p>feeling 😰 — see https://example.com</p>'
        result = clean_text(text)
        assert "<p>" not in result
        assert "https://" not in result
        assert "😰" not in result
        # Semantic content preserved
        assert "feeling" in result
        assert "anxious" in result
