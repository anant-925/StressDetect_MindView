"""
data/dataset.py
===============
Phase 1: PyTorch DataLoader with Sliding-Window Chunking

Provides ``StressDataset`` — a ``torch.utils.data.Dataset`` that splits
long texts into overlapping chunks of ``chunk_size`` tokens with a
configurable ``stride``, preventing truncation loss for long Reddit posts.

Each chunk is treated as an independent sample during training / inference,
and results can be aggregated per-document at evaluation time via
``doc_index``.

Usage
-----
>>> from data.dataset import StressDataset, create_dataloaders
>>> dataset = StressDataset(texts, labels, domains)
>>> train_dl, val_dl, test_dl = create_dataloaders(texts, labels, domains)
"""

from __future__ import annotations

from typing import Optional

import torch
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------------
# Vocabulary builder (simple word-level tokenizer)
# ---------------------------------------------------------------------------

_PAD_TOKEN = "<PAD>"
_UNK_TOKEN = "<UNK>"


class SimpleVocab:
    """Minimal word-level vocabulary for the CNN model.

    Assigns a unique integer to each token seen during ``build()``.
    """

    def __init__(self) -> None:
        self.token2idx: dict[str, int] = {_PAD_TOKEN: 0, _UNK_TOKEN: 1}
        self.idx2token: dict[int, str] = {0: _PAD_TOKEN, 1: _UNK_TOKEN}
        self.pad_idx: int = 0
        self.unk_idx: int = 1

    def build(self, texts: list[str], min_freq: int = 2) -> "SimpleVocab":
        """Build vocabulary from a list of texts.

        Parameters
        ----------
        texts : list[str]
            Raw text strings.
        min_freq : int
            Minimum token frequency to be included.

        Returns
        -------
        SimpleVocab
            self, for method chaining.
        """
        freq: dict[str, int] = {}
        for text in texts:
            for token in text.lower().split():
                freq[token] = freq.get(token, 0) + 1

        for token, count in freq.items():
            if count >= min_freq and token not in self.token2idx:
                idx = len(self.token2idx)
                self.token2idx[token] = idx
                self.idx2token[idx] = token

        return self

    def encode(self, text: str) -> list[int]:
        """Convert a text string to a list of token indices."""
        return [
            self.token2idx.get(t, self.unk_idx) for t in text.lower().split()
        ]

    def __len__(self) -> int:
        return len(self.token2idx)


# ---------------------------------------------------------------------------
# Sliding-Window Chunking Dataset
# ---------------------------------------------------------------------------

DEFAULT_CHUNK_SIZE: int = 200
DEFAULT_STRIDE: int = 50


class StressDataset(Dataset):
    """PyTorch Dataset with sliding-window chunking for long texts.

    Parameters
    ----------
    texts : list[str]
        Raw text strings.
    labels : list[int]
        Binary labels (0 = no stress, 1 = stress).
    domains : list[str]
        Domain tags (e.g. ``'reddit_long'``, ``'twitter_short'``).
    vocab : SimpleVocab, optional
        Pre-built vocabulary. If ``None``, one is built from ``texts``.
    chunk_size : int
        Maximum number of tokens per chunk.
    stride : int
        Step size between consecutive chunks.
    """

    def __init__(
        self,
        texts: list[str],
        labels: list[int],
        domains: list[str],
        vocab: SimpleVocab | None = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        stride: int = DEFAULT_STRIDE,
    ) -> None:
        if not (len(texts) == len(labels) == len(domains)):
            raise ValueError(
                "texts, labels, and domains must have the same length"
            )

        self.chunk_size = chunk_size
        self.stride = stride

        # Build or reuse vocabulary
        if vocab is None:
            self.vocab = SimpleVocab().build(texts)
        else:
            self.vocab = vocab

        # Pre-compute all chunks
        self._chunks: list[torch.Tensor] = []
        self._labels: list[int] = []
        self._domains: list[str] = []
        self._doc_indices: list[int] = []  # maps chunk → original doc

        for doc_idx, (text, label, domain) in enumerate(
            zip(texts, labels, domains)
        ):
            token_ids = self.vocab.encode(text)

            if len(token_ids) == 0:
                # Empty text → single padded chunk
                chunk = torch.zeros(chunk_size, dtype=torch.long)
                self._chunks.append(chunk)
                self._labels.append(label)
                self._domains.append(domain)
                self._doc_indices.append(doc_idx)
                continue

            # Generate sliding-window chunks
            chunks_created = 0
            for start in range(0, len(token_ids), stride):
                end = start + chunk_size
                chunk_ids = token_ids[start:end]

                # Pad if shorter than chunk_size
                if len(chunk_ids) < chunk_size:
                    chunk_ids = chunk_ids + [self.vocab.pad_idx] * (
                        chunk_size - len(chunk_ids)
                    )

                self._chunks.append(torch.tensor(chunk_ids, dtype=torch.long))
                self._labels.append(label)
                self._domains.append(domain)
                self._doc_indices.append(doc_idx)
                chunks_created += 1

                # Stop if we've consumed the entire text
                if end >= len(token_ids):
                    break

    def __len__(self) -> int:
        return len(self._chunks)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | int | str]:
        return {
            "input_ids": self._chunks[idx],
            "label": self._labels[idx],
            "domain": self._domains[idx],
            "doc_index": self._doc_indices[idx],
        }


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------


def collate_fn(batch: list[dict]) -> dict[str, torch.Tensor | list]:
    """Custom collate function for ``StressDataset``.

    Stacks ``input_ids`` and ``label`` into tensors; keeps ``domain``
    and ``doc_index`` as lists.
    """
    input_ids = torch.stack([item["input_ids"] for item in batch])
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    domains = [item["domain"] for item in batch]
    doc_indices = [item["doc_index"] for item in batch]

    return {
        "input_ids": input_ids,
        "labels": labels,
        "domains": domains,
        "doc_indices": doc_indices,
    }


def create_dataloaders(
    texts: list[str],
    labels: list[int],
    domains: list[str],
    vocab: SimpleVocab | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    stride: int = DEFAULT_STRIDE,
    batch_size: int = 32,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader, SimpleVocab]:
    """Create train / validation / test DataLoaders.

    Parameters
    ----------
    texts, labels, domains : list
        Raw data arrays.
    vocab : SimpleVocab, optional
        Pre-built vocabulary; built from training split if ``None``.
    chunk_size, stride : int
        Sliding-window parameters.
    batch_size : int
        Batch size for all loaders.
    train_ratio, val_ratio : float
        Proportions for the train and validation splits.
        Test ratio = ``1 - train_ratio - val_ratio``.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    tuple[DataLoader, DataLoader, DataLoader, SimpleVocab]
        ``(train_loader, val_loader, test_loader, vocab)``
    """
    import random

    n = len(texts)
    indices = list(range(n))
    random.seed(seed)
    random.shuffle(indices)

    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]

    def _select(idx_list: list[int]) -> tuple[list[str], list[int], list[str]]:
        return (
            [texts[i] for i in idx_list],
            [labels[i] for i in idx_list],
            [domains[i] for i in idx_list],
        )

    train_texts, train_labels, train_domains = _select(train_idx)
    val_texts, val_labels, val_domains = _select(val_idx)
    test_texts, test_labels, test_domains = _select(test_idx)

    # Build vocab from training data only
    if vocab is None:
        vocab = SimpleVocab().build(train_texts)

    train_ds = StressDataset(
        train_texts, train_labels, train_domains,
        vocab=vocab, chunk_size=chunk_size, stride=stride,
    )
    val_ds = StressDataset(
        val_texts, val_labels, val_domains,
        vocab=vocab, chunk_size=chunk_size, stride=stride,
    )
    test_ds = StressDataset(
        test_texts, test_labels, test_domains,
        vocab=vocab, chunk_size=chunk_size, stride=stride,
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn,
    )

    return train_loader, val_loader, test_loader, vocab
