"""
training/retrain.py
===================
Periodic RL-style retraining from collected feedback.

This script loads the experience-replay buffer written by ``/feedback``
API calls, merges those samples with the original supervised training data,
and fine-tunes the existing model checkpoint using reward-weighted loss.

How the RL signal works
-----------------------
Each feedback event produces one row in the ``experience`` table:

- ``text``   — original input text
- ``label``  — *corrected* label (same as predicted when feedback=1,
               flipped when feedback=0)
- ``reward`` — non-negative weight (``1.5`` for any feedback, vs ``1.0``
               for baseline samples), indicating how strongly the gradient
               should reinforce or correct that sample.

By combining the corrected label with a higher loss weight the model is:
- reinforced on patterns it gets right (same label, weight 1.5)
- corrected on patterns it gets wrong (flipped label, weight 1.5)

This is RL in spirit without requiring a separate policy-gradient objective.

Usage
-----
    python training/retrain.py

    # Optional overrides
    python training/retrain.py \\
        --checkpoint checkpoints/model.pt \\
        --output     checkpoints/model.pt \\
        --data       data/processed/unified_stress.csv \\
        --db         stress_detection.db \\
        --epochs     3 \\
        --min-feedback 10
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.feedback import FeedbackStore  # noqa: E402
from training.train import (  # noqa: E402
    CHUNK_SIZE,
    ROOT_DIR,
    STRIDE,
    VOCAB_SIZE,
    FocalLoss,
    _StressChunkDataset,
    _accuracy,
    _build_model_and_tokenizer,
    _find_best_threshold,
    _load_csv,
    _run_epoch,
    _stratified_split,
    weighted_loss,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DEFAULT_CHECKPOINT = os.path.join(ROOT_DIR, "checkpoints", "model.pt")
_DEFAULT_DATA = os.path.join(
    ROOT_DIR, "data", "processed", "unified_stress.csv"
)
_DEFAULT_DB = os.environ.get("STRESS_DB_PATH", "stress_detection.db")


def _load_feedback_samples(
    db_path: str,
    min_samples: int,
) -> pd.DataFrame:
    """Return feedback experience as a DataFrame with reward weights.

    Returns an empty DataFrame when fewer than ``min_samples`` rows exist.
    """
    store = FeedbackStore(db_path)
    rows = store.get_experience_for_training(min_samples=min_samples)
    store.close()

    if not rows:
        return pd.DataFrame(columns=["text", "label", "reward", "domain"])

    df = pd.DataFrame(rows)
    df["domain"] = "feedback"
    return df


def _build_feedback_dataset(
    df: pd.DataFrame,
    chunk_size: int = CHUNK_SIZE,
    stride: int = STRIDE,
) -> _StressChunkDataset:
    """Build a reward-weighted CNN dataset from feedback rows."""
    return _StressChunkDataset(
        texts=df["text"].tolist(),
        labels=df["label"].tolist(),
        chunk_size=chunk_size,
        stride=stride,
        rewards=df["reward"].tolist(),
    )


# ---------------------------------------------------------------------------
# Main retraining function
# ---------------------------------------------------------------------------

def retrain(
    checkpoint_path: str = _DEFAULT_CHECKPOINT,
    output_path: str = _DEFAULT_CHECKPOINT,
    data_path: str | None = _DEFAULT_DATA,
    db_path: str = _DEFAULT_DB,
    epochs: int = 3,
    batch_size: int = 32,
    lr: float = 5e-4,
    val_ratio: float = 0.15,
    seed: int = 42,
    device_str: str = "cuda",
    min_feedback: int = 10,
    max_fpr: float = 0.30,
    min_threshold: float = 0.50,
) -> None:
    """Fine-tune the checkpoint with reward-weighted feedback samples.

    Parameters
    ----------
    checkpoint_path : str
        Path to the existing model checkpoint to fine-tune.
    output_path : str
        Destination path for the updated checkpoint.
    data_path : str | None
        Path to the original supervised CSV.  When provided, feedback
        samples are *merged* with the original data so that the model
        does not catastrophically forget prior knowledge.
    db_path : str
        Path to the SQLite database containing the experience table.
    epochs : int
        Number of fine-tuning epochs.
    batch_size : int
        Mini-batch size.
    lr : float
        Learning rate for fine-tuning (should be lower than initial training).
    val_ratio : float
        Fraction of original data held out for validation.
    seed : int
        Random seed.
    device_str : str
        ``"cuda"`` or ``"cpu"``.
    min_feedback : int
        Minimum number of feedback rows required before retraining starts.
    max_fpr : float
        Maximum false positive rate constraint forwarded to
        ``_find_best_threshold``.  Prevents the retrained model from
        selecting a pathologically low decision threshold.
    min_threshold : float
        Hard lower bound on the decision threshold (forwarded to
        ``_find_best_threshold``).
    """
    torch.manual_seed(seed)
    device = torch.device(
        device_str if torch.cuda.is_available() or device_str == "cpu" else "cpu"
    )
    print(f"Using device: {device}")

    # ── Load feedback ──
    print(f"\nLoading feedback from: {db_path}")
    fb_df = _load_feedback_samples(db_path, min_feedback)
    if fb_df.empty:
        print(
            f"Not enough feedback yet (need >= {min_feedback} rows). "
            "Run the API and collect more user feedback first."
        )
        return
    print(f"Feedback samples: {len(fb_df)}")

    # ── Load & split original data (optional) ──
    val_df = pd.DataFrame(columns=["text", "label", "domain"])
    baseline_train_df = pd.DataFrame(columns=["text", "label", "domain"])

    if data_path and os.path.isfile(data_path):
        print(f"Loading original dataset from: {data_path}")
        orig_df = _load_csv(data_path)
        baseline_train_df, val_df = _stratified_split(orig_df, val_ratio, seed)
        print(
            f"Original train: {len(baseline_train_df):,}, "
            f"Original val: {len(val_df):,}"
        )
    else:
        print("No original dataset path provided — training on feedback only.")

    # ── Load checkpoint ──
    model_type = "cnn"
    dropout = 0.3
    decision_threshold = 0.5
    extra_checkpoint_keys: dict = {}

    if os.path.isfile(checkpoint_path):
        try:
            ckpt = torch.load(
                checkpoint_path, map_location="cpu", weights_only=True
            )
            if isinstance(ckpt, dict):
                model_type = ckpt.get("model_type", "cnn")
                dropout = float(ckpt.get("dropout", 0.3))
                threshold_val = ckpt.get("decision_threshold")
                if isinstance(threshold_val, torch.Tensor):
                    threshold_val = float(threshold_val.item())
                if isinstance(threshold_val, (float, int)):
                    decision_threshold = float(threshold_val)
                # Preserve all existing checkpoint metadata
                extra_checkpoint_keys = {
                    k: v
                    for k, v in ckpt.items()
                    if k not in {
                        "model_state_dict",
                        "decision_threshold",
                        "model_type",
                        "dropout",
                    }
                }
        except Exception as exc:
            print(f"Warning: could not read checkpoint ({exc}); using defaults.")
    else:
        print(f"No checkpoint found at {checkpoint_path}; starting from scratch.")
        ckpt = None

    model, _, _ = _build_model_and_tokenizer(model_type, dropout, max_length=256)
    if ckpt is not None and isinstance(ckpt, dict):
        state_dict = ckpt.get("model_state_dict", ckpt)
        try:
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError:
            model.load_state_dict(state_dict, strict=False)
    model = model.to(device)

    # ── Build datasets ──
    # Combine baseline (weight 1.0) and feedback (weight 1.5)
    all_texts: list[str] = []
    all_labels: list[int] = []
    all_rewards: list[float] = []

    if not baseline_train_df.empty:
        all_texts.extend(baseline_train_df["text"].tolist())
        all_labels.extend(baseline_train_df["label"].tolist())
        all_rewards.extend([1.0] * len(baseline_train_df))

    all_texts.extend(fb_df["text"].tolist())
    all_labels.extend(fb_df["label"].tolist())
    all_rewards.extend(fb_df["reward"].tolist())

    if not all_texts:
        print("No training data available. Exiting.")
        return

    train_ds = _StressChunkDataset(
        texts=all_texts,
        labels=all_labels,
        rewards=all_rewards,
    )
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=0
    )

    # Validation uses only original data (no rewards needed)
    val_loader = None
    if not val_df.empty:
        val_ds = _StressChunkDataset(
            texts=val_df["text"].tolist(),
            labels=val_df["label"].tolist(),
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False, num_workers=0
        )

    print(
        f"Training on {len(all_texts):,} samples "
        f"({len(fb_df):,} feedback + {len(baseline_train_df):,} baseline)"
    )

    # ── Fine-tuning loop ──
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = FocalLoss()

    best_val_f1 = 0.0
    best_threshold = decision_threshold
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc, _, _ = _run_epoch(
            model, train_loader, criterion, optimizer, device, is_train=True
        )
        elapsed = time.time() - t0
        print(
            f"Epoch {epoch}/{epochs}  "
            f"loss={tr_loss:.4f}  acc={tr_acc:.2%}  ({elapsed:.1f}s)"
        )

        if val_loader is not None:
            _, _, val_probs, val_labels_arr = _run_epoch(
                model,
                val_loader,
                criterion,
                None,
                device,
                is_train=False,
                collect_probs=True,
            )
            if val_probs is not None and val_labels_arr is not None:
                threshold, metrics = _find_best_threshold(
                    val_labels_arr, val_probs,
                    max_fpr=max_fpr,
                    min_threshold=min_threshold,
                )
                val_f1 = metrics["f1"]
                print(
                    f"  Val F1={val_f1:.2%}  threshold={threshold:.2f}"
                )
                if val_f1 >= best_val_f1:
                    best_val_f1 = val_f1
                    best_threshold = threshold

    # ── Save updated checkpoint ──
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "decision_threshold": float(best_threshold),
        "model_type": model_type,
        "dropout": dropout,
        **extra_checkpoint_keys,
    }
    torch.save(checkpoint, output_path)

    print(f"\nRetrained checkpoint saved to: {output_path}")
    if val_loader is not None:
        print(f"Best validation F1: {best_val_f1:.2%}")
    print(f"Decision threshold:  {best_threshold:.2f}")
    print(
        "\nRestart the API server to pick up the updated weights:\n"
        "  uvicorn api.main:app --host 0.0.0.0 --port 8000\n"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Retrain the stress model using RL-style feedback data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--checkpoint",
        default=_DEFAULT_CHECKPOINT,
        help="Path to the existing model checkpoint to fine-tune",
    )
    p.add_argument(
        "--output",
        default=_DEFAULT_CHECKPOINT,
        help="Destination path for the updated checkpoint (can be the same)",
    )
    p.add_argument(
        "--data",
        default=_DEFAULT_DATA,
        help="Path to the original supervised CSV for regularisation",
    )
    p.add_argument(
        "--db",
        default=_DEFAULT_DB,
        help="Path to the SQLite database containing the experience table",
    )
    p.add_argument("--epochs",   type=int,   default=3,    help="Fine-tuning epochs")
    p.add_argument("--batch-size", type=int, default=32,   help="Mini-batch size")
    p.add_argument("--lr",       type=float, default=5e-4, help="Learning rate")
    p.add_argument(
        "--min-feedback", type=int, default=10,
        help="Minimum feedback rows required before retraining",
    )
    p.add_argument(
        "--device", default="cuda", choices=["cuda", "cpu"],
        help="Compute device",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    retrain(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        data_path=args.data,
        db_path=args.db,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        min_feedback=args.min_feedback,
        device_str=args.device,
    )
