"""
tests/test_dataset.py
=====================
Unit tests for data/dataset.py — StressDataset and sliding-window chunking.
"""

import torch
import pytest

from data.dataset import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_STRIDE,
    SimpleVocab,
    StressDataset,
    collate_fn,
    create_dataloaders,
)


# ---------------------------------------------------------------------------
# SimpleVocab
# ---------------------------------------------------------------------------


class TestSimpleVocab:
    def test_initial_state(self):
        vocab = SimpleVocab()
        assert len(vocab) == 2  # PAD + UNK
        assert vocab.pad_idx == 0
        assert vocab.unk_idx == 1

    def test_build_from_texts(self):
        texts = ["hello world", "hello again", "world hello"]
        vocab = SimpleVocab().build(texts, min_freq=2)
        assert "hello" in vocab.token2idx
        assert "world" in vocab.token2idx
        # "again" appears only once with min_freq=2, should be excluded
        assert "again" not in vocab.token2idx

    def test_encode_known_tokens(self):
        vocab = SimpleVocab().build(["cat dog", "dog cat"], min_freq=1)
        ids = vocab.encode("cat dog")
        assert len(ids) == 2
        assert all(i > 1 for i in ids)

    def test_encode_unknown_tokens(self):
        vocab = SimpleVocab().build(["cat dog"], min_freq=1)
        ids = vocab.encode("cat unknown")
        assert ids[1] == vocab.unk_idx


# ---------------------------------------------------------------------------
# StressDataset
# ---------------------------------------------------------------------------


class TestStressDataset:
    def test_short_text_single_chunk(self):
        """A short text (< chunk_size) produces exactly 1 chunk."""
        texts = ["hello world"]
        labels = [1]
        domains = ["twitter_short"]
        ds = StressDataset(texts, labels, domains, chunk_size=10, stride=5)
        assert len(ds) == 1
        item = ds[0]
        assert item["input_ids"].shape == (10,)
        assert item["label"] == 1
        assert item["domain"] == "twitter_short"
        assert item["doc_index"] == 0

    def test_long_text_multiple_chunks(self):
        """A long text should be split into multiple overlapping chunks."""
        # Create a text with 25 words
        text = " ".join([f"word{i}" for i in range(25)])
        ds = StressDataset(
            [text], [0], ["reddit_long"],
            chunk_size=10, stride=5,
        )
        # 25 tokens with chunk_size=10, stride=5:
        # chunks at start: 0, 5, 10, 15 → 4 chunks
        # (start=15, end=25 covers remaining tokens, then break)
        assert len(ds) == 4
        # All chunks have same label and doc_index
        for i in range(4):
            assert ds[i]["label"] == 0
            assert ds[i]["doc_index"] == 0

    def test_padding_applied(self):
        """Chunks shorter than chunk_size should be zero-padded."""
        ds = StressDataset(
            ["hello"], [1], ["twitter_short"],
            chunk_size=10, stride=5,
        )
        item = ds[0]
        # "hello" → 1 token, padded to 10
        assert item["input_ids"].shape == (10,)
        assert item["input_ids"][1].item() == 0  # padding

    def test_empty_text_produces_padded_chunk(self):
        ds = StressDataset([""], [0], ["twitter_short"], chunk_size=5, stride=3)
        assert len(ds) == 1
        assert torch.all(ds[0]["input_ids"] == 0)

    def test_multiple_documents(self):
        """Multiple documents each produce their own chunks."""
        texts = ["word1 word2", "word3 word4 word5 word6 word7"]
        labels = [0, 1]
        domains = ["twitter_short", "reddit_long"]
        ds = StressDataset(texts, labels, domains, chunk_size=3, stride=2)
        # Doc 0: 2 tokens → 1 chunk
        # Doc 1: 5 tokens, chunk_size=3, stride=2 → chunks at 0, 2, 4 → 3 chunks
        assert len(ds) >= 2

        # Verify doc_indices
        doc_indices = [ds[i]["doc_index"] for i in range(len(ds))]
        assert 0 in doc_indices
        assert 1 in doc_indices

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="same length"):
            StressDataset(["text"], [1, 2], ["domain"])


# ---------------------------------------------------------------------------
# Collate function
# ---------------------------------------------------------------------------


class TestCollateFunction:
    def test_collate_batches_correctly(self):
        ds = StressDataset(
            ["hello world", "bye now"], [0, 1],
            ["twitter_short", "reddit_long"],
            chunk_size=5, stride=3,
        )
        batch = [ds[i] for i in range(min(2, len(ds)))]
        collated = collate_fn(batch)
        assert isinstance(collated["input_ids"], torch.Tensor)
        assert isinstance(collated["labels"], torch.Tensor)
        assert isinstance(collated["domains"], list)


# ---------------------------------------------------------------------------
# create_dataloaders
# ---------------------------------------------------------------------------


class TestCreateDataloaders:
    def test_creates_three_loaders(self):
        texts = [f"text {i}" for i in range(20)]
        labels = [i % 2 for i in range(20)]
        domains = ["twitter_short"] * 20
        train_dl, val_dl, test_dl, vocab = create_dataloaders(
            texts, labels, domains,
            chunk_size=5, stride=3, batch_size=4,
        )
        assert train_dl is not None
        assert val_dl is not None
        assert test_dl is not None
        assert isinstance(vocab, SimpleVocab)

    def test_loaders_produce_batches(self):
        texts = [f"this is sample text number {i}" for i in range(30)]
        labels = [i % 2 for i in range(30)]
        domains = ["reddit_long"] * 30
        train_dl, _, _, _ = create_dataloaders(
            texts, labels, domains,
            chunk_size=10, stride=5, batch_size=4,
        )
        batch = next(iter(train_dl))
        assert "input_ids" in batch
        assert "labels" in batch
        assert batch["input_ids"].ndim == 2
