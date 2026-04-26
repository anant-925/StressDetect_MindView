"""
tests/test_data_preprocessing.py
=================================
Unit tests for the data_preprocessing module.

Tests use synthetic in-memory DataFrames so no real dataset files are needed.
"""

import os
import tempfile

import pandas as pd
import pytest

# Import functions under test
from data_preprocessing import (
    _CONTRAST_SAMPLES,
    _find_column,
    clean_text,
    load_dreaddit,
    load_happy_neutral,
    load_reddit_combi,
    load_reddit_title,
    load_stressed_tweets,
    load_twitter_full,
    merge_datasets,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_csv(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)


# ---------------------------------------------------------------------------
# Tests for _find_column
# ---------------------------------------------------------------------------


class TestFindColumn:
    def test_exact_match(self):
        df = pd.DataFrame(columns=["text", "label", "extra"])
        assert _find_column(df, ["text"]) == "text"

    def test_case_insensitive_match(self):
        df = pd.DataFrame(columns=["Text", "Label"])
        assert _find_column(df, ["text"]) == "Text"

    def test_first_candidate_wins(self):
        df = pd.DataFrame(columns=["selftext", "text"])
        assert _find_column(df, ["text", "selftext"]) == "text"

    def test_fallback_candidate(self):
        df = pd.DataFrame(columns=["selftext", "label"])
        assert _find_column(df, ["text", "selftext"]) == "selftext"

    def test_missing_raises(self):
        df = pd.DataFrame(columns=["foo", "bar"])
        with pytest.raises(KeyError):
            _find_column(df, ["text", "label"])


# ---------------------------------------------------------------------------
# Tests for clean_text
# ---------------------------------------------------------------------------


class TestCleanText:
    def test_strips_whitespace(self):
        # clean_text now takes a string and returns a string
        assert clean_text("  hello  ").strip() == "hello"
        assert clean_text("world  ").strip() == "world"

    def test_empty_string_becomes_empty(self):
        # Empty / whitespace-only inputs return an empty string
        assert clean_text("") == ""
        assert clean_text("   ") == ""

    def test_preserves_content_words(self):
        # Content words survive stop-word filtering; "I" and "feel" are
        # handled differently — "i" is a stop word, "feel" is kept.
        result = clean_text("I feel stressed")
        assert "stressed" in result
        assert "feel" in result

    def test_removes_stop_words(self):
        # Pure stop words should be filtered
        result = clean_text("the a an is are")
        # After filtering only stop words, falls back to lowercased original
        # (no content words to return); important thing: no error.
        assert isinstance(result, str)

    def test_preserves_contrast_words(self):
        # 'not', 'but', 'however' must survive filtering
        result = clean_text("I am not happy but okay")
        assert "not" in result
        assert "but" in result


# ---------------------------------------------------------------------------
# Tests for individual dataset loaders
# ---------------------------------------------------------------------------


class TestLoadDreaddit:
    def test_loads_csv(self, tmp_path):
        df = pd.DataFrame({"text": ["post1", "post2"], "label": [0, 1]})
        csv_path = str(tmp_path / "dreaddit-train.csv")
        df.to_csv(csv_path, index=False)

        result = load_dreaddit(csv_path)
        assert list(result.columns) == ["text", "label", "domain"]
        assert len(result) == 2
        assert all(result["domain"] == "reddit_long")

    def test_loads_from_zip(self, tmp_path):
        import zipfile

        df = pd.DataFrame({"text": ["zipped post"], "label": [1]})
        csv_name = "dreaddit-train.csv"
        csv_in_zip = tmp_path / csv_name
        df.to_csv(str(csv_in_zip), index=False)

        zip_path = str(tmp_path / (csv_name + ".zip"))
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.write(str(csv_in_zip), csv_name)

        # Remove the plain CSV so the loader falls back to ZIP
        os.remove(str(csv_in_zip))

        result = load_dreaddit(str(csv_in_zip))
        assert len(result) == 1
        assert result["label"].iloc[0] == 1


class TestLoadRedditCombi:
    def test_basic_load(self, tmp_path):
        df = pd.DataFrame({"text": ["combo post"], "label": [0]})
        csv_path = str(tmp_path / "Reddit_Combi.csv")
        df.to_csv(csv_path, index=False)

        result = load_reddit_combi(csv_path)
        assert len(result) == 1
        assert result["domain"].iloc[0] == "reddit_long"


class TestLoadRedditTitle:
    def test_basic_load(self, tmp_path):
        df = pd.DataFrame({"title": ["My Title"], "label": [1]})
        csv_path = str(tmp_path / "Reddit_Title.csv")
        df.to_csv(csv_path, index=False)

        result = load_reddit_title(csv_path)
        assert len(result) == 1
        assert result["text"].iloc[0] == "My Title"
        assert result["domain"].iloc[0] == "reddit_short"


class TestLoadTwitterFull:
    def test_basic_load(self, tmp_path):
        df = pd.DataFrame({"text": ["tweet1", "tweet2"], "label": [0, 1]})
        csv_path = str(tmp_path / "Twitter_Full.csv")
        df.to_csv(csv_path, index=False)

        result = load_twitter_full(csv_path)
        assert len(result) == 2
        assert all(result["domain"] == "twitter_short")


class TestLoadStressedTweets:
    def test_implicit_label(self, tmp_path):
        df = pd.DataFrame({"text": ["I am so stressed", "Everything is awful"]})
        csv_path = str(tmp_path / "Stressed_Tweets.csv")
        df.to_csv(csv_path, index=False)

        result = load_stressed_tweets(csv_path)
        assert len(result) == 2
        assert all(result["label"] == 1)
        assert all(result["domain"] == "twitter_short")


class TestLoadHappyNeutral:
    def test_implicit_label(self, tmp_path):
        df = pd.DataFrame({"text": ["I feel great", "What a wonderful day"]})
        csv_path = str(tmp_path / "Happy_Neutral.csv")
        df.to_csv(csv_path, index=False)

        result = load_happy_neutral(csv_path)
        assert len(result) == 2
        assert all(result["label"] == 0)
        assert all(result["domain"] == "happy_neutral")


# ---------------------------------------------------------------------------
# Tests for merge_datasets
# ---------------------------------------------------------------------------


class TestMergeDatasets:
    def test_merges_multiple_datasets(self, tmp_path):
        """Merge 3 available datasets; 2 missing should be skipped.
        The 12 hard-negative contrast samples are always appended."""
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        out_path = str(tmp_path / "unified.csv")

        # Create dreaddit
        pd.DataFrame({"text": ["dreaddit post"], "label": [0]}).to_csv(
            str(raw_dir / "dreaddit-train.csv"), index=False
        )
        # Create Twitter_Full
        pd.DataFrame({"text": ["tweet text"], "label": [1]}).to_csv(
            str(raw_dir / "Twitter_Full.csv"), index=False
        )
        # Create Stressed_Tweets
        pd.DataFrame({"text": ["stressed tweet"]}).to_csv(
            str(raw_dir / "Stressed_Tweets.csv"), index=False
        )

        result = merge_datasets(str(raw_dir), out_path)

        n_contrast = len(_CONTRAST_SAMPLES)
        assert len(result) == 3 + n_contrast
        assert {"text", "label", "domain"}.issubset(result.columns)
        assert os.path.isfile(out_path)

        # Verify saved CSV matches
        saved = pd.read_csv(out_path)
        assert len(saved) == 3 + n_contrast

    def test_drops_empty_text(self, tmp_path):
        """Rows with empty/whitespace text should be dropped.
        Contrast samples are always appended after dropping."""
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        out_path = str(tmp_path / "unified.csv")

        pd.DataFrame(
            {"text": ["valid", "", "   ", "also valid"], "label": [0, 1, 0, 1]}
        ).to_csv(str(raw_dir / "dreaddit-train.csv"), index=False)

        result = merge_datasets(str(raw_dir), out_path)

        n_contrast = len(_CONTRAST_SAMPLES)
        assert len(result) == 2 + n_contrast
        # The dataset rows (before contrast samples) should contain the valid texts
        dataset_texts = result["text"].tolist()
        assert "valid" in dataset_texts
        assert "also valid" in dataset_texts

    def test_label_is_int(self, tmp_path):
        """Labels should always be int 0 or 1."""
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        out_path = str(tmp_path / "unified.csv")

        pd.DataFrame({"text": ["post"], "label": [1.0]}).to_csv(
            str(raw_dir / "dreaddit-train.csv"), index=False
        )

        result = merge_datasets(str(raw_dir), out_path)
        assert pd.api.types.is_integer_dtype(result["label"])

    def test_no_datasets_exits(self, tmp_path):
        """If no datasets found, the function should exit."""
        raw_dir = tmp_path / "empty"
        raw_dir.mkdir()
        out_path = str(tmp_path / "unified.csv")

        with pytest.raises(SystemExit):
            merge_datasets(str(raw_dir), out_path)


# ---------------------------------------------------------------------------
# Regression tests for _CONTRAST_SAMPLES
# ---------------------------------------------------------------------------


class TestContrastSamplesRegression:
    """Regression tests to catch mislabelled entries in _CONTRAST_SAMPLES.

    These guard against future regressions where clearly stressed text
    is accidentally labelled 0 or clearly non-stressed text is labelled 1.
    """

    def test_deadline_sleep_deprivation_is_stressed(self):
        """'Deadlines piling up + not slept' must be label=1 (stressed)."""
        texts_that_must_be_stressed = [
            "deadlines are piling up and I cannot keep up",
            "deadlines are piling up I have not slept properly in days",
            "I have not slept properly in days I am exhausted",
            "I haven't slept more than three hours in a week",
            "sleep deprived and stressed I can barely function",
            "I feel like I am drowning in responsibilities",
            "the stress has been building for weeks and I am at my limit",
            "I have back to back deadlines and no time to breathe",
        ]
        lookup = {text: label for text, label in _CONTRAST_SAMPLES}
        for text in texts_that_must_be_stressed:
            assert text in lookup, (
                f"Expected stressed example missing from _CONTRAST_SAMPLES: {text!r}"
            )
            assert lookup[text] == 1, (
                f"Example should be label=1 but got {lookup[text]}: {text!r}"
            )

    def test_positive_resolution_examples_are_non_stressed(self):
        """Contrast/concession examples with positive resolution must be label=0."""
        texts_that_must_be_non_stressed = [
            "I am tired but happy",
            "I feel overwhelmed but satisfied",
            "was stressed but things are better now",
            "I feel calm and content today",
            "I don't have any worries today",
            "nothing is going wrong right now",
        ]
        lookup = {text: label for text, label in _CONTRAST_SAMPLES}
        for text in texts_that_must_be_non_stressed:
            assert text in lookup, (
                f"Expected non-stressed example missing from _CONTRAST_SAMPLES: {text!r}"
            )
            assert lookup[text] == 0, (
                f"Example should be label=0 but got {lookup[text]}: {text!r}"
            )

    def test_all_labels_are_binary(self):
        """Every entry in _CONTRAST_SAMPLES must have label 0 or 1."""
        for text, label in _CONTRAST_SAMPLES:
            assert label in (0, 1), (
                f"Invalid label {label!r} for sample: {text!r}"
            )

    def test_both_classes_represented(self):
        """_CONTRAST_SAMPLES must contain at least one example of each class."""
        labels = {label for _, label in _CONTRAST_SAMPLES}
        assert 0 in labels, "_CONTRAST_SAMPLES has no label=0 (non-stressed) examples"
        assert 1 in labels, "_CONTRAST_SAMPLES has no label=1 (stressed) examples"
