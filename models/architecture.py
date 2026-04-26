"""
models/architecture.py
======================
Phase 2 → Phase 5: Model Architecture

Tier 1 — **OptimizedMultichannelCNN** (PyTorch from scratch)
    Parallel 1D convolution channels (kernel sizes 2, 3, 5) with
    **min_len trimming** to prevent shape mismatch on concatenation,
    followed by **multi-head self-attention** (default 4 heads) and a
    classification head.

Tier 2 — Transformer wrappers
    Lightweight wrappers around HuggingFace models:
    - ``DeBERTaStressClassifier`` (DeBERTa-v3-Small)
    - ``MiniLMStressClassifier``  (MiniLM-L6-v2)

Calibration
-----------
- ``TemperatureScaling`` — post-hoc probability calibration (Guo et al. 2017).
  Wraps any classifier and divides logits by a learned scalar ``T`` before
  softmax, reducing overconfidence.

Attention
---------
- ``MultiHeadSelfAttention`` (new default) — scaled dot-product attention
  split across ``num_heads`` independent subspaces then projected back.
  Produces richer features and more interpretable per-token importance
  weights compared to single-head dot-product attention.
- ``DotProductSelfAttention`` kept for backward compatibility (single head).

Design Guardrails
-----------------
- Conv1D outputs are trimmed to ``min_len`` before concatenation —
  this is the CRITICAL guard against tensor shape mismatch.
- Self-attention returns attention weights alongside the pooled vector
  for downstream explainability / heatmap rendering.
"""

from __future__ import annotations

import hashlib
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Stop-word dampening
# ---------------------------------------------------------------------------
# Common English stop words that carry little semantic signal for stress
# detection but tend to dominate attention weights (e.g. 'I', 'the', 'a').
# Reducing their embedding magnitude before the conv layers prevents the
# attention mechanism from over-emphasising them, which was observed in
# heatmap analysis.
# ---------------------------------------------------------------------------

_STOP_WORDS: frozenset[str] = frozenset({
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves",
    "you", "your", "yours", "yourself", "yourselves",
    "he", "him", "his", "himself", "she", "her", "hers", "herself",
    "it", "its", "itself", "they", "them", "their", "theirs", "themselves",
    "what", "which", "who", "whom", "this", "that", "these", "those",
    "am", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "having", "do", "does", "did", "doing",
    "a", "an", "the", "and", "but", "if", "or", "because", "as",
    "until", "while", "of", "at", "by", "for", "with", "about",
    "against", "between", "through", "during", "before", "after",
    "above", "below", "to", "from", "up", "down", "in", "out",
    "on", "off", "over", "under", "again", "further", "then", "once",
    "here", "there", "when", "where", "why", "how", "all", "both",
    "each", "few", "more", "most", "other", "some", "such", "no",
    "nor", "not", "only", "own", "same", "so", "than", "too", "very",
    "s", "t", "can", "will", "just", "don", "should", "now",
    "d", "ll", "m", "o", "re", "ve", "y",
})


def _compute_stop_word_ids(vocab_size: int) -> set[int]:
    """Return hash-based token IDs for :data:`_STOP_WORDS`.

    Uses the same ``md5`` hashing scheme as the project tokenizers in
    ``api/main.py`` and ``training/train.py`` so that the IDs match at
    both training and inference time.
    """
    ids: set[int] = set()
    for word in _STOP_WORDS:
        token_id = (
            int(
                hashlib.md5(
                    word.encode("utf-8"), usedforsecurity=False
                ).hexdigest(),
                16,
            )
            % (vocab_size - 1)
            + 1
        )
        ids.add(token_id)
    return ids


# ---------------------------------------------------------------------------
# Tier 1: OptimizedMultichannelCNN
# ---------------------------------------------------------------------------


class DotProductSelfAttention(nn.Module):
    """Simple scaled dot-product self-attention over a sequence.

    Input shape : ``(batch, seq_len, hidden)``
    Output shape: ``(batch, hidden)`` (attended pool) + ``(batch, seq_len)``

    Kept for backward compatibility.  New code should prefer
    :class:`MultiHeadSelfAttention`.
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.scale = hidden_dim ** 0.5

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : Tensor, shape ``(B, L, H)``

        Returns
        -------
        pooled : Tensor, shape ``(B, H)``
        weights : Tensor, shape ``(B, L)``  — attention weights (for heatmaps)
        """
        q = self.query(x)  # (B, L, H)
        k = self.key(x)    # (B, L, H)
        v = self.value(x)  # (B, L, H)

        scores = torch.bmm(q, k.transpose(1, 2)) / self.scale  # (B, L, L)
        attn = F.softmax(scores, dim=-1)  # (B, L, L)

        context = torch.bmm(attn, v)  # (B, L, H)

        # Pool: mean of attention-weighted values
        pooled = context.mean(dim=1)  # (B, H)

        # Per-token importance: mean attention received from all queries
        weights = attn.mean(dim=1)  # (B, L)

        return pooled, weights


class MultiHeadSelfAttention(nn.Module):
    """Multi-head scaled dot-product self-attention (Vaswani et al. 2017).

    Splits the hidden dimension into ``num_heads`` independent subspaces,
    computes scaled dot-product attention within each head, then concatenates
    and projects the results.  This produces richer feature representations
    than single-head attention and yields more interpretable per-token
    importance weights for heatmap rendering.

    Input shape : ``(batch, seq_len, hidden)``
    Output shape: ``(batch, hidden)`` (attended pool) + ``(batch, seq_len)``

    Parameters
    ----------
    hidden_dim : int
        Total hidden dimension.  Must be divisible by ``num_heads``.
    num_heads : int
        Number of parallel attention heads.  Default: 4.
    dropout : float
        Dropout applied to attention weights during training.
    """

    def __init__(
        self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by "
                f"num_heads ({num_heads})."
            )
        self.num_heads = num_heads
        self.d_k = hidden_dim // num_heads
        self.scale = self.d_k ** 0.5

        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : Tensor, shape ``(B, L, H)``

        Returns
        -------
        pooled : Tensor, shape ``(B, H)``
        weights : Tensor, shape ``(B, L)``  — per-token importance (for heatmaps)
        """
        B, L, H = x.shape

        # Project and reshape to (B, num_heads, L, d_k)
        q = self.query(x).view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        k = self.key(x).view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        v = self.value(x).view(B, L, self.num_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention: (B, num_heads, L, L)
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        attn = F.softmax(scores, dim=-1)       # (B, num_heads, L, L)
        attn = self.attn_dropout(attn)

        # Context: (B, num_heads, L, d_k)
        context = torch.matmul(attn, v)

        # Merge heads: (B, L, H)
        context = context.transpose(1, 2).contiguous().view(B, L, H)

        # Output projection
        out = self.out_proj(context)            # (B, L, H)

        # Mean-pool across the sequence
        pooled = out.mean(dim=1)                # (B, H)

        # Per-token importance: average attention weight received across
        # all heads and all query positions
        weights = attn.mean(dim=1).mean(dim=1)  # (B, L)

        return pooled, weights


# ---------------------------------------------------------------------------
# Temperature Scaling — post-hoc probability calibration
# ---------------------------------------------------------------------------


class TemperatureScaling(nn.Module):
    """Post-hoc calibration via temperature scaling (Guo et al. 2017).

    Divides logits by a single learnable scalar temperature ``T > 0`` before
    softmax.

    * ``T > 1`` → probabilities are smoothed toward 0.5 (reduces overconfidence).
    * ``T < 1`` → probabilities become more extreme.
    * ``T = 1`` → no effect (identity).

    The temperature is calibrated on a held-out validation set by minimising
    NLL loss.  During inference with an uncalibrated model, keep ``T = 1.0``.

    Parameters
    ----------
    temperature : float
        Initial temperature.  Defaults to 1.0 (no calibration).

    Example
    -------
    >>> ts = TemperatureScaling(temperature=1.5)
    >>> scaled_logits = ts(logits)          # use before softmax
    >>> ts.calibrate(val_logits, val_labels) # fit T on a held-out set
    """

    def __init__(self, temperature: float = 1.0) -> None:
        super().__init__()
        self.temperature = nn.Parameter(
            torch.ones(1) * max(temperature, 1e-6)
        )

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Return temperature-scaled logits."""
        return logits / self.temperature.clamp(min=1e-6)

    def calibrate(
        self, logits: torch.Tensor, labels: torch.Tensor
    ) -> None:
        """Fit the temperature on a held-out (logits, labels) set.

        Uses L-BFGS to minimise NLL.  ``logits`` and ``labels`` should
        be collected on the validation set *before* calling this method.

        Parameters
        ----------
        logits : Tensor, shape ``(N, C)``
            Raw (uncalibrated) model logits.
        labels : Tensor, shape ``(N,)``
            Ground-truth class indices.
        """
        from torch.optim import LBFGS

        nll = nn.CrossEntropyLoss()
        optimizer = LBFGS([self.temperature], lr=0.01, max_iter=50)

        def _eval() -> torch.Tensor:
            optimizer.zero_grad()
            loss = nll(self.forward(logits), labels)
            loss.backward()
            return loss

        optimizer.step(_eval)


class OptimizedMultichannelCNN(nn.Module):
    """Multi-channel 1D CNN with multi-head self-attention for stress detection.

    Architecture
    ------------
    1. Embedding layer (with stop-word dampening)
    2. Three parallel Conv1D branches (kernel sizes 2, 3, 5)
    3. **min_len trimming** — outputs are trimmed to the shortest length
       before concatenation to prevent tensor shape mismatches.
    4. Multi-head self-attention (default 4 heads)
    5. Classification head (FC → Dropout → FC)

    Parameters
    ----------
    vocab_size : int
        Size of the token vocabulary.
    embed_dim : int
        Embedding dimension.
    num_filters : int
        Number of filters per Conv1D branch.
    kernel_sizes : tuple[int, ...]
        Kernel sizes for the parallel Conv1D branches.
    num_classes : int
        Number of output classes (default 2: stress / no-stress).
    dropout : float
        Dropout probability.
    aux_dim : int
        Optional numeric feature dimension appended to pooled CNN features.
    num_attention_heads : int
        Number of attention heads.  Must divide ``num_filters * len(kernel_sizes)``
        evenly.  Set to 1 to use single-head dot-product attention (legacy).
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        num_filters: int = 64,
        kernel_sizes: tuple[int, ...] = (2, 3, 5),
        num_classes: int = 2,
        dropout: float = 0.3,
        aux_dim: int = 0,
        stop_word_dampening: float = 0.3,
        num_attention_heads: int = 4,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # ── Stop-word dampening ──
        # Build a per-token-ID lookup: 1.0 for stop words, 0.0 otherwise.
        # During forward() each embedding is scaled by
        #   1.0 - is_stop * (1.0 - stop_word_dampening)
        # so content words keep their full magnitude while stop-word
        # embeddings are reduced to ``stop_word_dampening`` of their
        # original magnitude.
        self.stop_word_dampening = stop_word_dampening
        stop_ids = _compute_stop_word_ids(vocab_size)
        stop_mask = torch.zeros(vocab_size, dtype=torch.float)
        for sid in stop_ids:
            stop_mask[sid] = 1.0
        # persistent=False → not part of state_dict, avoids checkpoint compat issues
        self.register_buffer("_stop_word_lookup", stop_mask, persistent=False)

        # Parallel Conv1D branches
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(embed_dim, num_filters, kernel_size=ks, padding=0)
                for ks in kernel_sizes
            ]
        )

        total_filters = num_filters * len(kernel_sizes)

        # ── Attention ──
        # Use multi-head attention when possible; fall back to single-head
        # dot-product attention if total_filters is not divisible by num_heads.
        if num_attention_heads > 1 and total_filters % num_attention_heads == 0:
            self.attention: nn.Module = MultiHeadSelfAttention(
                total_filters, num_heads=num_attention_heads, dropout=dropout
            )
        else:
            self.attention = DotProductSelfAttention(total_filters)

        self.aux_dim = aux_dim
        aux_hidden = min(aux_dim, total_filters // 2) if aux_dim > 0 else 0
        self.aux_projection = (
            nn.Sequential(
                nn.Linear(aux_dim, aux_hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            if aux_dim > 0
            else None
        )

        combined_dim = total_filters + aux_hidden

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, combined_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(combined_dim // 2, num_classes),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        input_ids: torch.Tensor,
        aux_features: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        input_ids : Tensor, shape ``(B, L)``
            Token indices.

        Returns
        -------
        dict with keys:
            ``logits``  : Tensor, shape ``(B, num_classes)``
            ``attention_weights`` : Tensor, shape ``(B, seq_len')``
        """
        # Embedding: (B, L) → (B, L, E)
        x = self.embedding(input_ids)
        x = self.dropout(x)

        # ── Stop-word dampening ──
        # Reduce embedding magnitudes for stop-word tokens so that the
        # subsequent conv + attention layers do not over-emphasise them.
        is_stop = self._stop_word_lookup[input_ids]  # (B, L), 0 or 1
        dampening = 1.0 - is_stop * (1.0 - self.stop_word_dampening)  # (B, L)
        x = x * dampening.unsqueeze(-1)  # (B, L, E)

        # Conv1D expects (B, C, L) — transpose
        x_t = x.transpose(1, 2)  # (B, E, L)

        # Apply parallel convolutions + ReLU
        conv_outputs = []
        for conv in self.convs:
            c = F.relu(conv(x_t))  # (B, F, L')
            conv_outputs.append(c)

        # ─── CRITICAL: Trim to min_len to prevent shape mismatch ───
        min_len = min(c.size(2) for c in conv_outputs)
        conv_outputs = [c[:, :, :min_len] for c in conv_outputs]

        # Concatenate along the filter dimension: (B, F*3, min_len)
        merged = torch.cat(conv_outputs, dim=1)

        # Transpose back for attention: (B, min_len, F*3)
        merged = merged.transpose(1, 2)

        # Self-attention (multi-head or single-head)
        pooled, attn_weights = self.attention(merged)  # (B, F*3), (B, min_len)

        if self.aux_projection is not None:
            if aux_features is None:
                aux_features = torch.zeros(
                    pooled.size(0),
                    self.aux_dim,
                    device=pooled.device,
                )
            aux_emb = self.aux_projection(aux_features)
            pooled = torch.cat([pooled, aux_emb], dim=1)

        # Classification
        logits = self.classifier(pooled)  # (B, num_classes)

        return {"logits": logits, "attention_weights": attn_weights}


# ---------------------------------------------------------------------------
# Tier 2: Transformer wrappers
# ---------------------------------------------------------------------------


class DeBERTaStressClassifier(nn.Module):
    """Stress classifier wrapping ``microsoft/deberta-v3-small``.

    Uses the HuggingFace ``transformers`` library for the backbone and
    adds a simple classification head.
    """

    MODEL_NAME = "microsoft/deberta-v3-small"

    def __init__(self, num_classes: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        from transformers import AutoModel

        self.backbone = AutoModel.from_pretrained(self.MODEL_NAME)
        hidden = self.backbone.config.hidden_size
        # +1 for optional sentiment feature
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden + 1, num_classes),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        sentiment: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        outputs = self.backbone(
            input_ids=input_ids, attention_mask=attention_mask
        )
        # CLS token pooling
        pooled = outputs.last_hidden_state[:, 0, :]

        if sentiment is not None:
            sentiment = sentiment.unsqueeze(1) if sentiment.dim() == 1 else sentiment
            pooled = torch.cat([pooled, sentiment], dim=1)
        else:
            # Append neutral sentiment (0.5) when not provided
            neutral = torch.full(
                (pooled.size(0), 1), 0.5,
                device=pooled.device, dtype=pooled.dtype,
            )
            pooled = torch.cat([pooled, neutral], dim=1)

        logits = self.classifier(pooled)
        return {"logits": logits}


class MiniLMStressClassifier(nn.Module):
    """Stress classifier wrapping ``sentence-transformers/all-MiniLM-L6-v2``.

    Uses mean pooling over the last hidden state as the sentence
    representation.
    """

    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self, num_classes: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        from transformers import AutoModel

        self.backbone = AutoModel.from_pretrained(self.MODEL_NAME)
        hidden = self.backbone.config.hidden_size
        # +1 for optional sentiment feature
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden + 1, num_classes),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        sentiment: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        outputs = self.backbone(
            input_ids=input_ids, attention_mask=attention_mask
        )
        # Mean pooling
        hidden_states = outputs.last_hidden_state
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(
                min=1e-9
            )
        else:
            pooled = hidden_states.mean(dim=1)

        if sentiment is not None:
            sentiment = sentiment.unsqueeze(1) if sentiment.dim() == 1 else sentiment
            pooled = torch.cat([pooled, sentiment], dim=1)
        else:
            neutral = torch.full(
                (pooled.size(0), 1), 0.5,
                device=pooled.device, dtype=pooled.dtype,
            )
            pooled = torch.cat([pooled, neutral], dim=1)

        logits = self.classifier(pooled)
        return {"logits": logits}
