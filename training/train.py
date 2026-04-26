"""
training/train.py
=================
Trains the stress detection model on the unified dataset with:
- document-level stratified splitting (label + domain)
- regularization (dropout, weight decay, label smoothing)
- validation F1 tracking, early stopping, and threshold calibration
- optional happy/neutral evaluation set to monitor false positives

The CNN tokenizer used here is intentionally identical to ``_simple_tokenize``
in ``api/main.py`` (hash-based, vocab_size=10000) so that saved checkpoints
work correctly at inference time without any vocabulary file.

Usage (local or Google Colab)
------------------------------
  # 1. Prepare data (only needed once)
  python data_preprocessing.py

# 2. Train (CNN)
  python training/train.py

  # Optional flags
  python training/train.py --epochs 15 --batch-size 64 --lr 1e-3 \
      --data data/processed/unified_stress.csv \
      --output checkpoints/model.pt

After training the checkpoint is automatically picked up by the API server
(``uvicorn api.main:app``).
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import random
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset

# Allow running from the repo root as well as from inside training/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.architecture import (  # noqa: E402
    DeBERTaStressClassifier,
    MiniLMStressClassifier,
    OptimizedMultichannelCNN,
)
from utils.sentiment import get_sentiment_score  # noqa: E402
from utils.text_preprocessing import clean_text  # noqa: E402


# ---------------------------------------------------------------------------
# Constants — must stay in sync with api/main.py
# ---------------------------------------------------------------------------

VOCAB_SIZE = 10_000   # _DEFAULT_VOCAB_SIZE in api/main.py
CHUNK_SIZE = 200      # _CHUNK_SIZE in api/main.py
STRIDE = 50          # Stride 50 with CHUNK_SIZE=200 → 150-token overlap (75% of chunk)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
THRESHOLD_MIN = 0.05
THRESHOLD_MAX = 0.95
THRESHOLD_STEPS = 19
MIN_RECALL_THRESHOLD = 0.6
TRANSFORMER_LR = 2e-5


# ---------------------------------------------------------------------------
# Tokenizer (identical to _simple_tokenize in api/main.py)
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[int]:
    """Hash each whitespace-delimited token into [1, VOCAB_SIZE-1].

    Index 0 is reserved for padding.  Uses ``hashlib.md5`` for a
    fully deterministic mapping that is stable across all platforms,
    Python processes, and interpreter restarts (unlike ``hash()`` which
    is randomised by ``PYTHONHASHSEED``).  This guarantees that a model
    trained on Colab produces identical token IDs when served on Windows.
    """
    tokens = text.lower().split()
    return [
        int(hashlib.md5(t.encode("utf-8"), usedforsecurity=False).hexdigest(), 16)
        % (VOCAB_SIZE - 1) + 1
        for t in tokens
    ]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


def _load_csv(data_path: str) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError(f"{data_path} must have 'text' and 'label' columns.")

    df = df.dropna(subset=["text"]).reset_index(drop=True)
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"] != ""].reset_index(drop=True)

    # Apply the same preprocessing pipeline used at inference time so that
    # the model trains on clean tokens identical to what it will see in
    # production (URLs, HTML, emojis, repeated chars all normalised).
    df["text"] = df["text"].apply(clean_text)
    df = df[df["text"] != ""].reset_index(drop=True)

    df["label"] = df["label"].astype(int)
    if "domain" not in df.columns:
        df["domain"] = "unknown"
    return df


def _describe_dataset(df: pd.DataFrame) -> None:
    print("\nDataset summary")
    print("-" * 50)
    print("Label distribution:")
    print(df["label"].value_counts().sort_index().to_string())
    print("\nDomain distribution:")
    print(df["domain"].value_counts().to_string())

    print("\nLabel by domain:")
    table = pd.crosstab(df["domain"], df["label"])
    print(table.to_string())

    missing = []
    for domain in table.index:
        row = table.loc[domain]
        for label in (0, 1):
            if row.get(label, 0) == 0:
                missing.append((domain, label))
    if missing:
        print(
            "\nWarning: Some domains contain only one label. "
            "Consider adding happy/neutral negatives to reduce false positives."
        )


def _stratified_split(
    df: pd.DataFrame, val_ratio: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = random.Random(seed)
    group_keys = (df["label"].astype(str) + "|" + df["domain"].astype(str)).tolist()

    groups: dict[str, list[int]] = {}
    for idx, key in enumerate(group_keys):
        groups.setdefault(key, []).append(idx)

    train_idx: list[int] = []
    val_idx: list[int] = []
    for key in sorted(groups.keys()):
        indices = groups[key]
        rng.shuffle(indices)
        n_val = int(round(len(indices) * val_ratio))
        if len(indices) > 1:
            # Keep at least one train sample and one val sample.
            n_val = max(1, min(n_val, len(indices) - 1))
        else:
            n_val = 0

        if n_val:
            val_idx.extend(indices[:n_val])
            train_idx.extend(indices[n_val:])
        else:
            train_idx.extend(indices)

    if not val_idx:
        fallback = max(1, int(round(len(train_idx) * val_ratio)))
        val_idx = train_idx[:fallback]
        train_idx = train_idx[fallback:]

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)
    return train_df, val_df


class _StressChunkDataset(Dataset):
    """CNN dataset with sliding-window chunking.

    Optionally accepts per-sample ``rewards`` (non-negative floats) that are
    used by ``weighted_loss`` during RL-style fine-tuning via
    ``training/retrain.py``.  When ``rewards`` is ``None`` every sample
    gets an implicit weight of ``1.0`` and the standard ``CrossEntropyLoss``
    is applied unchanged.
    """

    def __init__(
        self,
        texts: list[str],
        labels: list[int],
        features: np.ndarray | None = None,
        chunk_size: int = CHUNK_SIZE,
        stride: int = STRIDE,
        rewards: list[float] | None = None,
    ) -> None:
        self._chunks: list[torch.Tensor] = []
        self._labels: list[int] = []
        self._features: list[torch.Tensor] | None = None
        self._rewards: list[float] | None = None

        if rewards is not None:
            if len(rewards) != len(texts):
                raise ValueError("rewards must align with texts.")
            self._rewards = []

        feature_rows = None
        if features is not None:
            if len(features) != len(texts):
                raise ValueError("Features must align with texts.")
            self._features = []
            feature_rows = [torch.tensor(row, dtype=torch.float) for row in features]

        for idx, (text, label) in enumerate(zip(texts, labels)):
            token_ids = _tokenize(text)
            label = int(label)
            feature_tensor = feature_rows[idx] if feature_rows is not None else None
            reward_val = rewards[idx] if rewards is not None else None

            if len(token_ids) == 0:
                self._chunks.append(torch.zeros(chunk_size, dtype=torch.long))
                self._labels.append(label)
                if self._features is not None:
                    self._features.append(feature_tensor)
                if self._rewards is not None:
                    self._rewards.append(reward_val)
                continue

            for start in range(0, len(token_ids), stride):
                end = start + chunk_size
                chunk = token_ids[start:end]
                if len(chunk) < chunk_size:
                    chunk = chunk + [0] * (chunk_size - len(chunk))
                self._chunks.append(torch.tensor(chunk, dtype=torch.long))
                self._labels.append(label)
                if self._features is not None:
                    self._features.append(feature_tensor)
                if self._rewards is not None:
                    self._rewards.append(reward_val)
                if end >= len(token_ids):
                    break

    def __len__(self) -> int:
        return len(self._chunks)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = {
            "input_ids": self._chunks[idx],
            "label": torch.tensor(self._labels[idx], dtype=torch.long),
        }
        if self._features is not None:
            item["features"] = self._features[idx]
        if self._rewards is not None:
            item["reward"] = torch.tensor(self._rewards[idx], dtype=torch.float)
        return item


class _TransformerDataset(Dataset):
    """Transformer dataset with tokenizer-based encoding."""

    def __init__(
        self,
        texts: list[str],
        labels: list[int],
        tokenizer,
        max_length: int,
    ) -> None:
        encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
        self._encodings = encodings
        self._labels = labels
        self._texts = texts

    def __len__(self) -> int:
        return len(self._labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = {k: torch.tensor(v[idx]) for k, v in self._encodings.items()}
        item["label"] = torch.tensor(self._labels[idx], dtype=torch.long)

        sentiment = get_sentiment_score(self._texts[idx])
        item["sentiment"] = torch.tensor(sentiment, dtype=torch.float)

        return item


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def _accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=-1)
    return (preds == labels).float().mean().item()


def weighted_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    rewards: torch.Tensor,
) -> torch.Tensor:
    """Per-sample reward-weighted cross-entropy loss.

    Good predictions (high reward) are reinforced; bad predictions
    (which were corrected in the dataset) are penalised by the same
    magnitude, implementing the RL-style update described in the
    problem statement.

    Parameters
    ----------
    logits : Tensor, shape ``(B, C)``
    labels : Tensor, shape ``(B,)``
    rewards : Tensor, shape ``(B,)``
        Non-negative scalar weight for each sample (e.g. ``1.5`` for
        feedback-derived samples, ``1.0`` for baseline samples).

    Returns
    -------
    Tensor
        Scalar mean weighted loss.
    """
    per_sample = nn.functional.cross_entropy(logits, labels, reduction="none")
    return (per_sample * rewards).mean()


class FocalLoss(nn.Module):
    """Focal loss (Lin et al. 2017) for binary stress classification.

    Reduces the loss contribution of easy, well-classified examples and
    focuses training on hard, ambiguous inputs — preventing a handful of
    common keyword patterns from dominating the gradient updates.

    Parameters
    ----------
    gamma : float
        Focusing parameter.  ``gamma=0`` recovers standard cross-entropy.
        ``gamma=2`` is the value recommended by Lin et al.
    weight : Tensor or None
        Per-class weight tensor forwarded to the underlying cross-entropy
        (handles label imbalance in the same way as ``nn.CrossEntropyLoss``).
    label_smoothing : float
        Label smoothing coefficient forwarded to cross-entropy.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        weight: torch.Tensor | None = None,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self._ce = nn.CrossEntropyLoss(
            weight=weight,
            reduction="none",
            label_smoothing=label_smoothing,
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = self._ce(logits, targets)          # (B,) per-sample CE loss
        pt = torch.exp(-ce)                     # p(correct class)
        return ((1.0 - pt) ** self.gamma * ce).mean()


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    is_train: bool,
    collect_probs: bool = False,
    warmup_scheduler=None,
    fp_penalty_weight: float = 0.0,
) -> tuple[float, float, np.ndarray | None, np.ndarray | None]:
    model.train(is_train)
    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0
    all_probs: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["label"].to(device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
                sentiment = batch.get("sentiment")
                if sentiment is not None:
                    sentiment = sentiment.to(device)
                    output = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        sentiment=sentiment,
                    )
                else:
                    output = model(input_ids=input_ids, attention_mask=attention_mask)
            else:
                features = batch.get("features")
                if features is not None:
                    features = features.to(device)
                    output = model(input_ids, aux_features=features)
                else:
                    output = model(input_ids)
            logits = output["logits"]

            # Use reward-weighted loss when the batch provides per-sample
            # rewards (i.e. during RL-style fine-tuning from feedback data).
            reward_weights = batch.get("reward")
            if reward_weights is not None:
                reward_weights = reward_weights.to(device)
                loss = weighted_loss(logits, labels, reward_weights)
            else:
                loss = criterion(logits, labels)

            # Soft FPR penalty: penalise the mean stress probability assigned
            # to negative samples in this batch.  Unlike threshold calibration
            # (which only acts at evaluation time), this pushes the gradient
            # directly toward lower false-positive rates during training.
            if is_train and fp_penalty_weight > 0.0:
                neg_mask = (labels == 0).float()
                n_neg = neg_mask.sum()
                if n_neg > 0:
                    stress_probs = torch.softmax(logits, dim=-1)[:, 1]
                    fpr_penalty = (stress_probs * neg_mask).sum() / n_neg
                    loss = loss + fp_penalty_weight * fpr_penalty

            if is_train and optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                if warmup_scheduler is not None:
                    warmup_scheduler.step()

            total_loss += loss.item()
            total_acc += _accuracy(logits, labels)
            n_batches += 1

            if collect_probs:
                probs = torch.softmax(logits, dim=-1)[:, 1]
                all_probs.append(probs.detach().cpu())
                all_labels.append(labels.detach().cpu())

    if collect_probs and all_probs:
        probs_np = torch.cat(all_probs).numpy()
        labels_np = torch.cat(all_labels).numpy()
    else:
        probs_np = None
        labels_np = None

    return (
        total_loss / max(n_batches, 1),
        total_acc / max(n_batches, 1),
        probs_np,
        labels_np,
    )


def _compute_metrics(
    labels: np.ndarray, probs: np.ndarray, threshold: float
) -> dict[str, float | list[list[int]]]:
    preds = (probs >= threshold).astype(int)
    precision = precision_score(labels, preds, zero_division=0)
    recall = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": cm.tolist(),
    }


def _find_best_threshold(
    labels: np.ndarray,
    probs: np.ndarray,
    max_fpr: float = 0.20,
    min_threshold: float = 0.50,
) -> tuple[float, dict[str, float | list[list[int]]]]:
    """Find the best decision threshold subject to FPR and minimum-threshold constraints.

    Why this matters
    ----------------
    Optimising purely for F1 on a stress-heavy dataset causes the search to
    collapse to very low thresholds (~0.15) that label almost everything as
    stressed.  Such a model has high recall but an FPR of 75–95%, making it
    clinically useless.

    Parameters
    ----------
    labels : np.ndarray
        Ground-truth binary labels (0 = no stress, 1 = stress).
    probs : np.ndarray
        Model stress probabilities (softmax output for class 1).
    max_fpr : float
        Maximum false positive rate allowed.  Thresholds that exceed this on
        the validation set are rejected regardless of their F1.
    min_threshold : float
        Hard lower bound on the selected threshold.  The deployed model will
        never be more aggressive than this value.

    Returns
    -------
    tuple
        ``(best_threshold, best_metrics)``
    """
    best_threshold = min_threshold
    best_metrics = _compute_metrics(labels, probs, best_threshold)
    best_f1 = best_metrics["f1"]

    for threshold in np.linspace(min_threshold, THRESHOLD_MAX, THRESHOLD_STEPS):
        t = float(threshold)
        metrics = _compute_metrics(labels, probs, t)
        cm = metrics["confusion_matrix"]
        tn, fp = cm[0][0], cm[0][1]
        fpr = fp / max(tn + fp, 1)
        if (
            metrics["f1"] > best_f1
            and metrics["recall"] >= MIN_RECALL_THRESHOLD
            and fpr <= max_fpr
        ):
            best_f1 = metrics["f1"]
            best_threshold = t
            best_metrics = metrics

    return best_threshold, best_metrics


def _build_model_and_tokenizer(
    model_type: str,
    dropout: float,
    max_length: int,
    aux_dim: int = 0,
) -> tuple[nn.Module, object | None, str | None]:
    if model_type == "cnn":
        model = OptimizedMultichannelCNN(
            vocab_size=VOCAB_SIZE,
            embed_dim=128,
            num_filters=64,
            kernel_sizes=(2, 3, 5),
            num_classes=2,
            dropout=dropout,
            aux_dim=aux_dim,
        )
        return model, None, None

    if model_type == "deberta":
        model = DeBERTaStressClassifier(dropout=dropout)
    elif model_type == "minilm":
        model = MiniLMStressClassifier(dropout=dropout)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    from transformers import AutoTokenizer

    model_name = model.MODEL_NAME
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.model_max_length = max_length
    return model, tokenizer, model_name


def _load_eval_set(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError(f"{path} must have 'text' and 'label' columns.")
    df = df.dropna(subset=["text"]).reset_index(drop=True)
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"] != ""].reset_index(drop=True)
    df["label"] = df["label"].astype(int)
    return df


def _prepare_feature_frame(
    df: pd.DataFrame,
    feature_cols: list[str],
    means: pd.Series | None = None,
    stds: pd.Series | None = None,
) -> tuple[np.ndarray, pd.Series, pd.Series]:
    features = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    if means is None:
        means = features.mean()
    if stds is None:
        stds = features.std().replace(0, 1)
    features = features.fillna(means)
    normalized = (features - means) / stds
    return normalized.to_numpy(dtype=np.float32), means, stds


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(
    data_path: str,
    output_path: str,
    epochs: int,
    batch_size: int,
    lr: float,
    val_ratio: float,
    seed: int,
    device_str: str,
    model_type: str,
    dropout: float,
    weight_decay: float,
    label_smoothing: float,
    class_weighted: bool,
    patience: int,
    eval_set_path: str | None,
    max_length: int,
    max_fpr: float = 0.20,
    min_threshold: float = 0.50,
    fp_penalty_weight: float = 0.2,
) -> None:
    torch.manual_seed(seed)
    device = torch.device(device_str if torch.cuda.is_available() or device_str == "cpu" else "cpu")
    print(f"Using device: {device}")

    # ── Data ──
    print(f"\nLoading dataset from: {data_path}")
    df = _load_csv(data_path)
    _describe_dataset(df)
    train_df, val_df = _stratified_split(df, val_ratio, seed)

    candidate_cols = [
        col for col in df.columns if col not in {"text", "label", "domain"}
    ]
    if candidate_cols:
        raw_candidates = df[candidate_cols]
        numeric_candidates = raw_candidates.apply(
            pd.to_numeric, errors="coerce"
        )
        coerced = numeric_candidates.isna() & raw_candidates.notna()
        if coerced.any().any():
            bad_cols = [
                col for col in candidate_cols if coerced[col].any()
            ]
            print(
                "Warning: Non-numeric values coerced to NaN in columns: "
                + ", ".join(bad_cols)
            )
        coverage = numeric_candidates.notna().mean()
        feature_cols = [
            col
            for col in numeric_candidates.columns
            if coverage[col] >= 0.5
        ]
        for col in feature_cols:
            df[col] = numeric_candidates[col]
    else:
        feature_cols = []
    if feature_cols:
        print(f"Numeric feature columns detected: {len(feature_cols):,}")
    elif model_type == "cnn":
        print("No numeric feature columns detected; training text-only CNN.")

    print(
        f"\nDocuments: {len(df):,}  |  Train: {len(train_df):,}  |  Val: {len(val_df):,}"
    )

    aux_dim = len(feature_cols) if model_type == "cnn" and feature_cols else 0
    model, tokenizer, model_name = _build_model_and_tokenizer(
        model_type=model_type,
        dropout=dropout,
        max_length=max_length,
        aux_dim=aux_dim,
    )
    model = model.to(device)

    train_features = None
    val_features = None
    eval_features = None
    feature_means = None
    feature_stds = None
    if feature_cols and model_type == "cnn":
        train_features, feature_means, feature_stds = _prepare_feature_frame(
            train_df, feature_cols
        )
        val_features, _, _ = _prepare_feature_frame(
            val_df, feature_cols, feature_means, feature_stds
        )
    elif feature_cols and model_type != "cnn":
        print(
            "Note: numeric features are available but only the CNN uses them."
        )

    if model_type == "cnn":
        train_ds = _StressChunkDataset(
            train_df["text"].tolist(),
            train_df["label"].tolist(),
            features=train_features,
        )
        val_ds = _StressChunkDataset(
            val_df["text"].tolist(),
            val_df["label"].tolist(),
            features=val_features,
        )
    else:
        train_ds = _TransformerDataset(
            train_df["text"].tolist(),
            train_df["label"].tolist(),
            tokenizer,
            max_length=max_length,
        )
        val_ds = _TransformerDataset(
            val_df["text"].tolist(),
            val_df["label"].tolist(),
            tokenizer,
            max_length=max_length,
        )

    print(f"Chunks: {len(train_ds):,} train  |  {len(val_ds):,} val")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=0
    )

    eval_loader = None
    if eval_set_path and os.path.isfile(eval_set_path):
        eval_df = _load_eval_set(eval_set_path)
        if model_type == "cnn":
            if feature_cols and feature_means is not None and feature_stds is not None:
                for col in feature_cols:
                    if col not in eval_df.columns:
                        eval_df[col] = np.nan
                eval_features, _, _ = _prepare_feature_frame(
                    eval_df, feature_cols, feature_means, feature_stds
                )
            eval_ds = _StressChunkDataset(
                eval_df["text"].tolist(),
                eval_df["label"].tolist(),
                features=eval_features,
            )
        else:
            eval_ds = _TransformerDataset(
                eval_df["text"].tolist(),
                eval_df["label"].tolist(),
                tokenizer,
                max_length=max_length,
            )
        eval_loader = DataLoader(
            eval_ds, batch_size=batch_size, shuffle=False, num_workers=0
        )
        print(f"Happy/neutral eval set: {len(eval_ds):,} samples")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}\n")

    class_weights = None
    if class_weighted:
        counts = train_df["label"].value_counts().reindex([0, 1], fill_value=0)
        total = counts.sum()
        weights = [
            total / (2 * counts[0]) if counts[0] > 0 else 1.0,
            total / (2 * counts[1]) if counts[1] > 0 else 1.0,
        ]
        class_weights = torch.tensor(weights, dtype=torch.float, device=device)

    criterion = (
        FocalLoss(gamma=2.0, weight=class_weights, label_smoothing=label_smoothing)
        if model_type == "cnn"
        else nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
    )
    if model_type == "cnn":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        # Cosine annealing with linear warmup: better convergence than
        # ReduceLROnPlateau for text CNNs — avoids premature learning-rate
        # collapses caused by noisy F1 on small validation sets.
        total_steps = len(train_loader) * epochs
        warmup_steps = min(int(0.1 * total_steps), 200)

        def _lr_lambda(current_step: int) -> float:
            """Linear warmup then cosine decay."""
            if current_step < warmup_steps:
                return float(current_step) / max(warmup_steps, 1)
            progress = float(current_step - warmup_steps) / max(
                total_steps - warmup_steps, 1
            )
            return max(0.01, 0.5 * (1.0 + math.cos(math.pi * progress)))

        warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, _lr_lambda
        )
        scheduler = None  # per-step updates only
    else:
        from transformers import get_linear_schedule_with_warmup

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=TRANSFORMER_LR, weight_decay=weight_decay
        )

        total_steps = len(train_loader) * epochs
        warmup_steps = int(0.1 * total_steps)

        warmup_scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
        scheduler = None

    best_val_f1 = 0.0
    best_threshold = min_threshold
    epochs_since_improve = 0
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    _EPOCH_HEADER = (
        f"{'Epoch':>6}  {'Train Loss':>11}  {'Train Acc':>10}  "
        f"{'Val Loss':>9}  {'Val F1':>8}  {'Val Prec':>9}  {'Val Rec':>8}  "
        f"{'Val FPR':>8}  {'Thresh':>7}  {'Time':>6}"
    )
    print(_EPOCH_HEADER)
    print("-" * len(_EPOCH_HEADER))

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc, _, _ = _run_epoch(
            model, train_loader, criterion, optimizer, device, is_train=True,
            warmup_scheduler=warmup_scheduler,
            fp_penalty_weight=fp_penalty_weight,
        )
        vl_loss, _, val_probs, val_labels = _run_epoch(
            model,
            val_loader,
            criterion,
            None,
            device,
            is_train=False,
            collect_probs=True,
        )
        elapsed = time.time() - t0

        if val_probs is None or val_labels is None:
            raise RuntimeError("Validation probabilities were not collected.")

        threshold, val_metrics = _find_best_threshold(
            val_labels, val_probs,
            max_fpr=max_fpr,
            min_threshold=min_threshold,
        )
        val_f1 = val_metrics["f1"]
        val_precision = val_metrics["precision"]
        val_recall = val_metrics["recall"]
        val_cm = val_metrics["confusion_matrix"]
        val_tn, val_fp = val_cm[0][0], val_cm[0][1]
        val_fpr = val_fp / max(val_tn + val_fp, 1)

        if scheduler is not None:
            scheduler.step(val_f1)

        marker = " ←" if val_f1 > best_val_f1 else ""
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_threshold = threshold
            epochs_since_improve = 0
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "decision_threshold": float(best_threshold),
                "model_type": model_type,
                "dropout": float(dropout),
            }
            if model_type != "cnn" and model_name:
                checkpoint["model_name"] = model_name
                checkpoint["tokenizer_max_length"] = int(max_length)
            if model_type == "cnn":
                checkpoint["chunk_size"] = int(CHUNK_SIZE)
                checkpoint["stride"] = int(STRIDE)
                if feature_cols:
                    checkpoint["feature_dim"] = len(feature_cols)
                    checkpoint["feature_columns"] = feature_cols
                    checkpoint["feature_means"] = feature_means.tolist()
                    checkpoint["feature_stds"] = feature_stds.tolist()
            torch.save(checkpoint, output_path)
        else:
            epochs_since_improve += 1

        print(
            f"{epoch:>6}  {tr_loss:>11.4f}  {tr_acc:>9.2%}  "
            f"{vl_loss:>9.4f}  {val_f1:>7.2%}  {val_precision:>8.2%}  "
            f"{val_recall:>7.2%}  {val_fpr:>7.2%}  {threshold:>7.2f}  {elapsed:>5.1f}s{marker}"
        )
        cm = val_metrics["confusion_matrix"]
        print(
            "  Confusion matrix: "
            f"TN={cm[0][0]} FP={cm[0][1]} FN={cm[1][0]} TP={cm[1][1]}"
        )

        if eval_loader is not None:
            _, _, eval_probs, eval_labels = _run_epoch(
                model,
                eval_loader,
                criterion,
                None,
                device,
                is_train=False,
                collect_probs=True,
            )
            if eval_probs is not None and eval_labels is not None:
                eval_metrics = _compute_metrics(
                    eval_labels, eval_probs, threshold
                )
                eval_cm = eval_metrics["confusion_matrix"]
                tn, fp = eval_cm[0][0], eval_cm[0][1]
                fp_rate = fp / max(tn + fp, 1)
                print(
                    "  Happy/neutral eval — "
                    f"FP rate: {fp_rate:.2%}, "
                    f"F1: {eval_metrics['f1']:.2%}"
                )

        if patience > 0 and epochs_since_improve >= patience:
            print(
                f"\nEarly stopping: no F1 improvement in {patience} epochs."
            )
            break

    print(f"\nBest validation F1: {best_val_f1:.2%}")
    print(f"Best decision threshold: {best_threshold:.2f}")
    print(f"Checkpoint saved to:       {output_path}")
    print(
        "\nTo use with the API, start the server normally:\n"
        "  uvicorn api.main:app --host 0.0.0.0 --port 8000\n"
        "The checkpoint is loaded automatically on the first /analyze request."
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train stress detection models with calibrated thresholds.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--data",
        default=os.path.join(ROOT_DIR, "data", "processed", "unified_stress.csv"),
        help="Path to the unified CSV produced by data_preprocessing.py",
    )
    p.add_argument(
        "--output",
        default=os.path.join(ROOT_DIR, "checkpoints", "model.pt"),
        help="Destination path for the best checkpoint",
    )
    p.add_argument(
        "--model",
        default="cnn",
        choices=["cnn", "deberta", "minilm"],
        help="Model backbone to train",
    )
    p.add_argument("--epochs",     type=int,   default=10,   help="Number of training epochs")
    p.add_argument("--batch-size", type=int,   default=64,   help="Mini-batch size")
    p.add_argument("--lr",         type=float, default=1e-3, help="Initial learning rate")
    p.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Weight decay for Adam/AdamW",
    )
    p.add_argument(
        "--dropout",
        type=float,
        default=0.3,
        help="Dropout probability (CNN) / classifier dropout (transformers)",
    )
    p.add_argument(
        "--label-smoothing",
        type=float,
        default=0.0,
        help="Label smoothing for cross-entropy loss",
    )
    p.add_argument(
        "--class-weighted",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use inverse-frequency class weights (enabled by default). "
            "Balances the gradient contribution of the minority class (no-stress) "
            "which is heavily outnumbered in typical datasets. "
            "Pass --no-class-weighted to disable."
        ),
    )
    p.add_argument(
        "--patience",
        type=int,
        default=3,
        help="Early stopping patience (epochs without F1 improvement)",
    )
    p.add_argument("--val-ratio",  type=float, default=0.1,  help="Fraction of data used for validation")
    p.add_argument("--seed",       type=int,   default=42,   help="Random seed")
    p.add_argument(
        "--eval-set",
        default=os.path.join(ROOT_DIR, "data", "eval", "happy_neutral_eval.csv"),
        help="Optional happy/neutral eval CSV (text,label)",
    )
    p.add_argument(
        "--max-length",
        type=int,
        default=256,
        help="Max sequence length for transformer models",
    )
    p.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to train on (falls back to cpu if CUDA is unavailable)",
    )
    p.add_argument(
        "--max-fpr",
        type=float,
        default=0.20,
        help=(
            "Maximum false positive rate allowed during threshold calibration. "
            "Thresholds that produce FPR > this value are rejected, preventing "
            "the model from collapsing to very low decision thresholds."
        ),
    )
    p.add_argument(
        "--min-threshold",
        type=float,
        default=0.50,
        help=(
            "Hard lower bound for the decision threshold. "
            "Ensures the model never classifies more than 50%% of the "
            "probability space as stressed by default."
        ),
    )
    p.add_argument(
        "--fp-penalty",
        type=float,
        default=0.2,
        help=(
            "Weight of the soft false-positive penalty term added to the "
            "training loss each batch.  At each step the mean stress "
            "probability assigned to negative (label=0) samples in the "
            "batch is multiplied by this weight and added to the main loss, "
            "directly penalising false positives during gradient descent.  "
            "Set to 0.0 to disable.  Increase (e.g. 0.4) when the "
            "happy/neutral FP rate remains high after training."
        ),
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if not os.path.isfile(args.data):
        print(f"ERROR: Dataset not found: {args.data}")
        print("Run  python data_preprocessing.py  first to create the unified CSV.")
        sys.exit(1)

    train(
        data_path=args.data,
        output_path=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        val_ratio=args.val_ratio,
        seed=args.seed,
        device_str=args.device,
        model_type=args.model,
        dropout=args.dropout,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        class_weighted=args.class_weighted,
        patience=args.patience,
        eval_set_path=args.eval_set,
        max_length=args.max_length,
        max_fpr=args.max_fpr,
        min_threshold=args.min_threshold,
        fp_penalty_weight=args.fp_penalty,
    )
