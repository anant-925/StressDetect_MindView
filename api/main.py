"""
api/main.py
===========
Phase 3: Secure FastAPI Backend

Endpoints
---------
- ``POST /register``  — Hash password via bcrypt, create user profile.
- ``POST /login``     — Verify password, return JWT token.
- ``POST /analyze``   — (JWT required) Run CNN inference, temporal analysis,
  and recommendation engine. Returns scores, interventions, attention weights.
- ``GET  /history``   — (JWT required) Retrieve past analysis sessions.

Security
--------
- Passwords are NEVER stored in plaintext (bcrypt).
- JWT tokens authenticate all ``/analyze`` and ``/history`` requests.
- Stress history is encrypted at rest via Fernet (AES-256).

Persistence
-----------
- User accounts and analysis sessions are stored in a SQLite database.
- Sessions survive server restarts and are available upon re-login.
"""

from __future__ import annotations

import hashlib
import logging
import os
import statistics
import threading
import time
from typing import Any, Optional

import torch
from fastapi import Depends, FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from database.db import DatabaseManager
from database.feedback import FeedbackStore
from intervention.engine import RecommendationEngine
from intervention.temporal_model import SecureTemporalModel
from models.architecture import (
    DeBERTaStressClassifier,
    MiniLMStressClassifier,
    OptimizedMultichannelCNN,
)
from security.auth import (
    create_jwt_token,
    decode_jwt_token,
    hash_password,
    verify_password,
)
from utils.llm_reward import get_llm_reward
from utils.reward import compute_combined_reward
from utils.sentiment import compute_sentiment_dampening, get_sentiment_score
from utils.text_preprocessing import clean_text

# ---------------------------------------------------------------------------
# App & global state
# ---------------------------------------------------------------------------

_APP_START_TIME = time.time()

app = FastAPI(
    title="Stress Detection API",
    description="Secure, intervention-oriented stress detection system",
    version="2.0.0",
)

# ---------------------------------------------------------------------------
# CORS — allow all origins in development / single-server deployments.
# For production, restrict ``allow_origins`` to your frontend domain(s).
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# SQLite-backed user + session store
_db = DatabaseManager()

# Feedback / experience-replay store (same DB file)
_feedback_store = FeedbackStore()

# Singletons
_recommendation_engine = RecommendationEngine()
_temporal_model = SecureTemporalModel()

# Security
_bearer_scheme = HTTPBearer()

# Model (lazy-loaded on first request)
_model: Optional[torch.nn.Module] = None
_vocab: Optional[Any] = None
_model_type: str = "cnn"
_decision_threshold: float = 0.5
_tokenizer: Optional[Any] = None
_tokenizer_max_length: int = 256
_feature_dim: int = 0
_DEFAULT_VOCAB_SIZE = 10000
_CHECKPOINT_PATH = os.environ.get(
    "STRESS_MODEL_CHECKPOINT", "checkpoints/model.pt"
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Inference guardrails
# ---------------------------------------------------------------------------

# The decision threshold is NEVER allowed to fall below this value, even if
# the training checkpoint was produced with a pathologically low threshold
# (e.g. 0.15) caused by unconstrained F1-only threshold calibration.
_MIN_DECISION_THRESHOLD: float = 0.50

# Clip raw model probabilities to this range before applying the threshold.
# Extreme values (< 0.10 or > 0.90) can amplify noise and indicate a
# poorly calibrated model; restricting the range keeps decisions sane.
_PROB_CLIP_MIN: float = 0.10
_PROB_CLIP_MAX: float = 0.90

# Dead-zone gating: predictions within this distance of the adaptive
# threshold are relabelled "uncertain" to avoid committing to a potentially
# false high/low near the decision boundary.
_CONFIDENCE_DEAD_ZONE: float = 0.07

# Ensemble Monte-Carlo Dropout: number of stochastic forward passes and the
# standard-deviation threshold above which the prediction is flagged as
# uncertain by the ensemble.
_ENSEMBLE_PASSES: int = 3
_ENSEMBLE_UNCERTAINTY_STD: float = 0.08
# Serialises MC-Dropout ensemble passes so concurrent requests do not
# interfere with each other's Dropout layer state.
_inference_lock: threading.Lock = threading.Lock()


def _enable_dropout(m: torch.nn.Module) -> None:
    """Set only Dropout layers to train mode for MC-Dropout ensemble passes."""
    if isinstance(m, torch.nn.Dropout):
        m.train()


def _classify_stress_level(stress_prob: float, decision_threshold: float) -> str:
    """Map a stress probability to a 4-way human-readable level.

    Bands (all relative to ``decision_threshold``, default 0.50):

    ============  ================  ================================
    Level         Probability range  Meaning
    ============  ================  ================================
    low           < threshold−0.10  Confidently not stressed
    uncertain     ±0.10 of thresh.  Near the boundary; unclear
    moderate      threshold+0.10 …  Clearly stressed but manageable
                  threshold+0.25
    high          ≥ threshold+0.25  High-stress; escalate
    ============  ================  ================================
    """
    low_bound = decision_threshold - 0.10
    uncertain_upper = decision_threshold + 0.10
    high_lower = decision_threshold + 0.25
    if stress_prob >= high_lower:
        return "high"
    if stress_prob >= uncertain_upper:
        return "moderate"
    if stress_prob >= low_bound:
        return "uncertain"
    return "low"


def _compute_confidence(stress_prob: float, decision_threshold: float) -> float:
    """Return a confidence score in [0, 1] derived from distance to the threshold.

    A score of 1.0 means the prediction is maximally far from the boundary
    (e.g. stress_prob = 0.0 or 1.0 with threshold = 0.5).  Values close to
    0.0 indicate the prediction is right on the decision boundary.
    """
    dist = abs(stress_prob - decision_threshold)
    return float(min(dist / _MIN_DECISION_THRESHOLD, 1.0))


# ---------------------------------------------------------------------------
# Short-input handler for common single-word inputs
# ---------------------------------------------------------------------------

_STRESS_WORDS = frozenset([
    "tired", "exhausted", "overwhelmed", "burnt", "done",
    "stressed", "anxious", "depressed", "hopeless", "miserable",
])
_NEUTRAL_WORDS = frozenset([
    "fine", "ok", "okay", "good", "alright",
])

# ---------------------------------------------------------------------------
# Inference post-processing: signal strength and contrast filtering
# ---------------------------------------------------------------------------

# High-frequency function words that carry no stress signal on their own.
# Used to measure how much *meaningful* content is in the input.
_LOW_SIGNAL_WORDS: frozenset[str] = frozenset({
    "i", "me", "my", "we", "you", "he", "she", "it", "they",
    "am", "is", "are", "was", "were", "be", "been", "being",
    "a", "an", "the", "and", "or", "of", "to", "in", "on",
    "at", "by", "for", "with", "as", "this", "that", "do",
    "did", "does", "have", "has", "had", "will", "would", "can",
    "could", "should", "may", "might", "shall",
})

# Contrast conjunctions that signal a positive override following a
# stress-trigger phrase (e.g. "I am stressed but happy").
_CONTRAST_CONJUNCTIONS: frozenset[str] = frozenset({
    "but", "however", "although", "though", "yet", "despite",
    "nevertheless", "nonetheless", "whereas",
})


def _handle_short_input(text: str) -> float | None:
    """Return a preset stress probability for very short inputs.

    Returns ``None`` when the input is not recognised as a common
    single-word pattern and should be passed to the model instead.
    """
    cleaned = text.lower().strip()
    if cleaned in _STRESS_WORDS:
        return 0.8
    if cleaned in _NEUTRAL_WORDS:
        return 0.2
    return None


def _apply_signal_filter(text: str, stress_prob: float) -> float:
    """Dampen stress probability when the input lacks meaningful content words.

    Inputs consisting almost entirely of low-signal function words (e.g.
    "I am the") carry no semantic content and should not trigger a high
    stress prediction.  Applies a 0.70 dampening factor when fewer than
    three content words are detected.

    Parameters
    ----------
    text : str
        Raw input text.
    stress_prob : float
        Current stress probability (after model + sentiment correction).

    Returns
    -------
    float
        Dampened stress probability.
    """
    content_words = [
        w for w in text.lower().split() if w not in _LOW_SIGNAL_WORDS
    ]
    if len(content_words) < 3:
        stress_prob *= 0.7
    return stress_prob


def _apply_contrast_filter(text: str, stress_prob: float) -> float:
    """Dampen stress probability when a contrast conjunction is present.

    Phrases like "I am stressed *but* happy" or "exhausted *however* grateful"
    carry a positive override that should suppress the stress score.  A 0.80
    dampening factor is applied whenever any contrast conjunction is found,
    regardless of position.

    Parameters
    ----------
    text : str
        Raw input text.
    stress_prob : float
        Current stress probability.

    Returns
    -------
    float
        Dampened stress probability.
    """
    tokens = set(text.lower().split())
    if tokens & _CONTRAST_CONJUNCTIONS:
        stress_prob *= 0.8
    return stress_prob

def _get_model() -> torch.nn.Module:
    """Lazy-load or create the CNN model.

    If a checkpoint file exists at ``_CHECKPOINT_PATH``, the function
    attempts to load the saved ``model_state_dict``.  When the checkpoint
    was produced by an *older* architecture (e.g. one that used a single
    ``fc`` layer instead of the current ``attention`` + ``classifier``
    head), loading with ``strict=True`` would raise a ``RuntimeError``.

    To stay backward-compatible the loader:
    1. Tries ``strict=True`` first.
    2. On key-mismatch ``RuntimeError``, retries with ``strict=False``
       so that all *compatible* weights (embedding, conv layers) are
       restored while new layers keep their random initialisation.
    3. Logs every missing / unexpected key for transparency.

    If no checkpoint exists the model is created with random weights.
    """
    global _model, _decision_threshold, _model_type, _tokenizer, _tokenizer_max_length, _feature_dim
    if _model is None:
        checkpoint = None
        if os.path.isfile(_CHECKPOINT_PATH):
            try:
                checkpoint = torch.load(
                    _CHECKPOINT_PATH, map_location="cpu", weights_only=True,
                )
            except Exception as exc:
                logger.warning(
                    "Failed to read checkpoint %s (%s); using randomly "
                    "initialised weights.",
                    _CHECKPOINT_PATH,
                    exc,
                )

        if isinstance(checkpoint, dict):
            _model_type = checkpoint.get("model_type", "cnn")
            threshold = checkpoint.get("decision_threshold")
            # Backward-compatible: threshold may be serialized as a tensor.
            if isinstance(threshold, torch.Tensor):
                threshold = float(threshold.item())
            if isinstance(threshold, (float, int)):
                _decision_threshold = float(threshold)
            # Safety guard: never use a threshold below the minimum, regardless
            # of what the checkpoint reports.  Unconstrained F1-only calibration
            # during training can produce pathological values like 0.15.
            _decision_threshold = max(_decision_threshold, _MIN_DECISION_THRESHOLD)
            _tokenizer_max_length = int(
                checkpoint.get("tokenizer_max_length", _tokenizer_max_length)
            )
            _feature_dim = int(checkpoint.get("feature_dim", 0))
            feature_columns = checkpoint.get("feature_columns")
            if _feature_dim == 0 and isinstance(feature_columns, list):
                _feature_dim = len(feature_columns)
            dropout = float(
                checkpoint.get(
                    "dropout",
                    0.3 if _model_type == "cnn" else 0.1,
                )
            )
        else:
            _model_type = "cnn"
            dropout = 0.3
            _feature_dim = 0

        if _model_type == "deberta":
            _model = DeBERTaStressClassifier(dropout=dropout)
        elif _model_type == "minilm":
            _model = MiniLMStressClassifier(dropout=dropout)
        else:
            _model = OptimizedMultichannelCNN(
                vocab_size=_DEFAULT_VOCAB_SIZE,
                embed_dim=128,
                num_filters=64,
                kernel_sizes=(2, 3, 5),
                num_classes=2,
                dropout=dropout,
                aux_dim=_feature_dim,
            )
            if _feature_dim > 0:
                logger.info(
                    "Checkpoint expects %d auxiliary features; inference "
                    "will use zero-filled features unless provided.",
                    _feature_dim,
                )

        if _model_type in {"deberta", "minilm"}:
            from transformers import AutoTokenizer

            model_name = None
            if isinstance(checkpoint, dict):
                model_name = checkpoint.get("model_name")
            if model_name is None:
                model_name = _model.MODEL_NAME
            _tokenizer = AutoTokenizer.from_pretrained(model_name)

        if checkpoint is not None:
            state_dict = (
                checkpoint.get("model_state_dict", checkpoint)
                if isinstance(checkpoint, dict)
                else checkpoint
            )

            try:
                _model.load_state_dict(state_dict, strict=True)
                logger.info("Loaded checkpoint from %s", _CHECKPOINT_PATH)
            except RuntimeError as exc:
                logger.warning(
                    "Strict checkpoint load failed (%s); retrying with "
                    "strict=False to restore compatible weights.",
                    exc,
                )
                result = _model.load_state_dict(state_dict, strict=False)
                if result.missing_keys:
                    logger.warning(
                        "Missing keys (randomly initialised): %s",
                        result.missing_keys,
                    )
                if result.unexpected_keys:
                    logger.warning(
                        "Unexpected keys (ignored): %s",
                        result.unexpected_keys,
                    )
        else:
            logger.info(
                "No checkpoint found at %s; using randomly initialised "
                "weights.",
                _CHECKPOINT_PATH,
            )

        _model.eval()
    return _model


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=8)


class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=1)


class InterventionResponse(BaseModel):
    title: str
    description: str
    category: str
    priority: int


class AnalyzeResponse(BaseModel):
    stress_score: float
    stress_label: str
    stress_level: str  # "low" | "moderate" | "high" | "uncertain"
    confidence: float  # how far the prediction is from the decision boundary [0, 1]
    temporal: dict
    interventions: list[InterventionResponse]
    is_crisis: bool
    crisis_message: Optional[str] = None
    matched_triggers: list[str]
    attention_weights: list[float]
    requires_escalation: bool = False  # True when 3+ consecutive above-threshold sessions
    is_uncertain: bool = False  # True when ensemble std is high or near-boundary


class SessionResponse(BaseModel):
    """A single past analysis session."""

    id: int
    stress_score: float
    stress_label: str
    temporal_data: dict
    interventions: list[dict]
    is_crisis: bool
    crisis_message: Optional[str] = None
    matched_triggers: list[str]
    attention_weights: list[float]
    created_at: float


class HistoryResponse(BaseModel):
    """Paginated list of past analysis sessions."""

    sessions: list[SessionResponse]
    total: int


class FeedbackRequest(BaseModel):
    """User-submitted feedback on a single prediction."""

    text: str = Field(..., min_length=1)
    prediction: float = Field(..., ge=0.0, le=1.0)
    user_feedback: int = Field(..., ge=0, le=1,
                               description="1 = prediction was correct, 0 = wrong")


class FeedbackResponse(BaseModel):
    """Acknowledgement returned after storing feedback."""

    status: str
    reward: float
    llm_reward: Optional[int] = None
    feedback_id: int


class FeedbackStatsResponse(BaseModel):
    """Aggregated feedback statistics for the authenticated user."""

    total: int
    mean_reward: float
    n_correct: int
    n_wrong: int
    accuracy_rate: float


class PersonalizationResponse(BaseModel):
    """Per-user score adjustment derived from their feedback history."""

    user_bias: float
    feedback_count: int
    description: str


# ---------------------------------------------------------------------------
# Auth dependency
# ---------------------------------------------------------------------------


def _get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(_bearer_scheme),
) -> str:
    """Decode the JWT and return the username (``sub`` claim)."""
    try:
        payload = decode_jwt_token(credentials.credentials)
        username: str | None = payload.get("sub")
        if username is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token: missing subject",
            )
        if not _db.user_exists(username):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found",
            )
        return username
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Could not validate credentials: {exc}",
        )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
def health() -> dict:
    """Liveness / readiness probe.

    Returns the service status, uptime in seconds, and whether the
    prediction model has been loaded into memory.
    """
    return {
        "status": "ok",
        "uptime_seconds": round(time.time() - _APP_START_TIME, 1),
        "model_loaded": _model is not None,
        "model_type": _model_type,
    }


@app.get("/model/info")
def model_info() -> dict:
    """Return metadata about the currently loaded prediction model.

    Useful for the UI settings panel and for debugging.
    """
    return {
        "model_type": _model_type,
        "decision_threshold": _decision_threshold,
        "vocab_size": _DEFAULT_VOCAB_SIZE,
        "checkpoint_path": _CHECKPOINT_PATH,
        "checkpoint_exists": os.path.isfile(_CHECKPOINT_PATH),
        "prob_clip_min": _PROB_CLIP_MIN,
        "prob_clip_max": _PROB_CLIP_MAX,
        "feature_dim": _feature_dim,
    }


@app.post("/register", response_model=TokenResponse, status_code=201)
def register(req: RegisterRequest) -> TokenResponse:
    """Register a new user with bcrypt-hashed password."""
    if _db.user_exists(req.username):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already exists",
        )

    _db.create_user(req.username, hash_password(req.password))

    token = create_jwt_token({"sub": req.username})
    return TokenResponse(access_token=token)


@app.post("/login", response_model=TokenResponse)
def login(req: LoginRequest) -> TokenResponse:
    """Verify credentials and return a JWT token."""
    user = _db.get_user(req.username)
    if user is None or not verify_password(req.password, user["password_hash"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
        )

    token = create_jwt_token({"sub": req.username})
    return TokenResponse(access_token=token)


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(
    req: AnalyzeRequest,
    username: str = Depends(_get_current_user),
) -> AnalyzeResponse:
    """Run full stress analysis pipeline (JWT required).

    Pipeline:
    0. Clean and normalise the input text (HTML, URLs, emojis, etc.).
    1. Tokenize text and run OptimizedMultichannelCNN inference.
    2. Decrypt user's temporal history, update profile, re-encrypt.
    3. Run RecommendationEngine.
    4. Persist session to database.
    5. Return scores, interventions, and attention weights.
    """
    model = _get_model()

    # ── 0. Text preprocessing ──
    # Normalise input before any downstream processing so that the text
    # seen by the model exactly matches what was seen during training.
    text = clean_text(req.text)
    if not text:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Input text is empty after preprocessing.",
        )

    # ── 0b. Short-input shortcut ──
    short_result = _handle_short_input(text)

    # ── 1. Model Inference ──
    attn_weights: list[float] = []
    is_uncertain_ensemble: bool = False
    if short_result is not None:
        stress_prob = short_result
    elif _model_type == "cnn":
        tokens = _simple_tokenize(text)
        input_tensor = torch.tensor([tokens], dtype=torch.long)

        # Single eval-mode pass — captures attention weights.
        with torch.no_grad():
            if _feature_dim > 0:
                # No auxiliary features at inference time; use zero-filled inputs.
                aux_features = torch.zeros(
                    (1, _feature_dim), dtype=torch.float
                )
                output = model(input_tensor, aux_features=aux_features)
            else:
                output = model(input_tensor)
        logits = output["logits"]
        attn_weights = output["attention_weights"][0].tolist()
        p_eval = float(torch.softmax(logits, dim=-1)[0, 1])

        # Ensemble MC-Dropout: additional stochastic passes with dropout only.
        # We selectively set Dropout layers to train mode instead of the whole
        # model, so the BatchNorm / LayerNorm statistics stay in eval mode.
        # The lock serialises model-state mutations so concurrent requests
        # do not interfere with each other's Dropout state.
        ensemble_probs: list[float] = [p_eval]

        with _inference_lock:
            model.apply(_enable_dropout)
            try:
                for _ in range(_ENSEMBLE_PASSES - 1):
                    with torch.no_grad():
                        if _feature_dim > 0:
                            out = model(input_tensor, aux_features=aux_features)
                        else:
                            out = model(input_tensor)
                    ensemble_probs.append(
                        float(torch.softmax(out["logits"], dim=-1)[0, 1])
                    )
            finally:
                model.eval()

        stress_prob = statistics.mean(ensemble_probs)
        ensemble_std = statistics.pstdev(ensemble_probs)
        is_uncertain_ensemble = ensemble_std > _ENSEMBLE_UNCERTAINTY_STD
    else:
        if _tokenizer is None:
            raise HTTPException(
                status_code=500,
                detail="Tokenizer not initialized for transformer model.",
            )
        encoded = _tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=_tokenizer_max_length,
        )
        sentiment_val = get_sentiment_score(text)
        sentiment_tensor = torch.tensor([sentiment_val], dtype=torch.float)
        with torch.no_grad():
            output = model(
                input_ids=encoded["input_ids"],
                attention_mask=encoded.get("attention_mask"),
                sentiment=sentiment_tensor,
            )
        logits = output["logits"]
        attn_weights = []

    if short_result is None and _model_type != "cnn":
        # CNN stress_prob is already averaged across ensemble passes above.
        probs = torch.softmax(logits, dim=-1)
        stress_prob = float(probs[0, 1])

    if short_result is None:
        # ── Sentiment correction ──
        dampening = compute_sentiment_dampening(text)
        stress_prob = stress_prob * dampening

        # ── Signal-strength filter ──
        stress_prob = _apply_signal_filter(text, stress_prob)

        # ── Contrast-phrase filter ──
        stress_prob = _apply_contrast_filter(text, stress_prob)

    # ── Probability calibration ──
    stress_prob = float(min(max(stress_prob, _PROB_CLIP_MIN), _PROB_CLIP_MAX))

    stress_label = (
        "stress" if stress_prob >= _decision_threshold else "no_stress"
    )

    # ── Multi-level classification + confidence ──
    stress_level = _classify_stress_level(stress_prob, _decision_threshold)
    confidence = _compute_confidence(stress_prob, _decision_threshold)

    # ── 2. Temporal Analysis (decrypt → compute → re-encrypt) ──
    user_data = _db.get_user(username)
    analysis, new_encrypted = _temporal_model.process(
        score=stress_prob,
        encrypted_history=user_data["encrypted_history"] if user_data else None,
    )
    _db.update_encrypted_history(username, new_encrypted)

    # ── Dead-zone gating ──
    # If the probability is within _CONFIDENCE_DEAD_ZONE of the adaptive
    # threshold, the prediction is too close to the boundary to be reliable.
    # Override the label to "uncertain" to avoid a false high/low call.
    if abs(stress_prob - analysis.adaptive_threshold) < _CONFIDENCE_DEAD_ZONE:
        stress_level = "uncertain"

    # ── Layer 4: Escalation detection ──
    # Query the 2 most recent saved sessions (before saving the current one).
    # If the last 3 sessions (including the current) all exceed the adaptive
    # threshold, flag the user for escalation to a professional.
    recent_sessions = _db.get_sessions(username, limit=2, offset=0)
    past_scores = [s["stress_score"] for s in recent_sessions]
    all_recent_scores = [stress_prob] + past_scores
    requires_escalation = (
        analysis.score_count >= 3
        and len(all_recent_scores) >= 3
        and all(s >= analysis.adaptive_threshold for s in all_recent_scores[:3])
    )

    # ── 3. Recommendation Engine ──
    recommendation = _recommendation_engine.recommend(
        text=text,
        stress_score=stress_prob,
        is_volatile=analysis.is_volatile,
        requires_escalation=requires_escalation,
    )

    # ── 4. Build response ──
    temporal_dict = {
        "stress_velocity": analysis.stress_velocity,
        "adaptive_threshold": analysis.adaptive_threshold,
        "exceeds_threshold": analysis.exceeds_threshold,
        "is_volatile": analysis.is_volatile,
        "volatility": analysis.volatility,
        "score_count": analysis.score_count,
    }

    interventions_list = [
        {
            "title": iv.title,
            "description": iv.description,
            "category": iv.category,
            "priority": iv.priority,
        }
        for iv in recommendation.interventions
    ]

    # ── 5. Persist session to database ──
    _db.save_session(
        username=username,
        stress_score=stress_prob,
        stress_label=stress_label,
        temporal_data=temporal_dict,
        interventions=interventions_list,
        is_crisis=recommendation.is_crisis,
        crisis_message=recommendation.crisis_message,
        matched_triggers=recommendation.matched_triggers,
        attention_weights=attn_weights,
    )

    return AnalyzeResponse(
        stress_score=stress_prob,
        stress_label=stress_label,
        stress_level=stress_level,
        confidence=confidence,
        temporal=temporal_dict,
        interventions=[
            InterventionResponse(**iv) for iv in interventions_list
        ],
        is_crisis=recommendation.is_crisis,
        crisis_message=recommendation.crisis_message,
        matched_triggers=recommendation.matched_triggers,
        attention_weights=attn_weights,
        requires_escalation=requires_escalation,
        is_uncertain=is_uncertain_ensemble or stress_level == "uncertain",
    )


@app.get("/history", response_model=HistoryResponse)
def history(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    username: str = Depends(_get_current_user),
) -> HistoryResponse:
    """Retrieve past analysis sessions for the authenticated user.

    Sessions are returned newest-first and support pagination via
    ``limit`` and ``offset`` query parameters.
    """
    sessions = _db.get_sessions(username, limit=limit, offset=offset)
    total = _db.get_session_count(username)
    return HistoryResponse(
        sessions=[SessionResponse(**s) for s in sessions],
        total=total,
    )


@app.post("/feedback", response_model=FeedbackResponse)
def submit_feedback(
    req: FeedbackRequest,
    username: str = Depends(_get_current_user),
) -> FeedbackResponse:
    """Store user feedback on a prediction and compute the RL reward signal.

    Pipeline
    --------
    1. Compute a ``±1`` reward from the user's binary rating.
    2. Optionally call an LLM judge (if ``OPENAI_API_KEY`` or
       ``GEMINI_API_KEY`` is set) and average with the user reward.
    3. Persist the feedback event and a corrected training sample to the
       ``feedback`` / ``experience`` tables.
    4. Return the reward so the UI can display it to the user.
    """
    llm_r = get_llm_reward(req.text, req.prediction)
    reward = compute_combined_reward(req.user_feedback, llm_r)

    feedback_id = _feedback_store.save_feedback(
        username=username,
        text=req.text,
        prediction=req.prediction,
        user_feedback=req.user_feedback,
        reward=reward,
        llm_reward=llm_r,
    )

    return FeedbackResponse(
        status="saved",
        reward=reward,
        llm_reward=llm_r,
        feedback_id=feedback_id,
    )


@app.get("/feedback/stats", response_model=FeedbackStatsResponse)
def feedback_stats(
    username: str = Depends(_get_current_user),
) -> FeedbackStatsResponse:
    """Return aggregated feedback statistics for the authenticated user."""
    stats = _feedback_store.get_user_stats(username)
    return FeedbackStatsResponse(**stats)


@app.get("/personalization", response_model=PersonalizationResponse)
def personalization(
    username: str = Depends(_get_current_user),
) -> PersonalizationResponse:
    """Return a per-user stress-score bias derived from feedback history.

    The bias is a small additive correction (−0.1 to +0.1) that shifts the
    model's raw prediction toward what past feedback suggests is accurate
    for this specific user.  A positive bias indicates the model has
    historically under-predicted stress for this user; a negative bias
    indicates over-prediction.

    The correction can be applied at inference time by client code.
    """
    stats = _feedback_store.get_user_stats(username)
    total = stats["total"]

    if total == 0:
        return PersonalizationResponse(
            user_bias=0.0,
            feedback_count=0,
            description="No feedback yet — bias is neutral.",
        )

    # Derive bias: mean_reward of +1 means the model is mostly right (no
    # correction needed); mean_reward near -1 means it is mostly wrong.
    # We map [-1, +1] → [+0.1, -0.1]: if the model is wrong more often,
    # nudge the score up (positive bias) to force the threshold to be met.
    mean_r = stats["mean_reward"]
    user_bias = round(-mean_r * 0.1, 4)

    accuracy_pct = int(stats["accuracy_rate"] * 100)
    description = (
        f"Based on {total} feedback event(s), model accuracy for you is "
        f"~{accuracy_pct}%.  Bias adjustment: {user_bias:+.4f}."
    )

    return PersonalizationResponse(
        user_bias=user_bias,
        feedback_count=total,
        description=description,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CHUNK_SIZE = 200


def _simple_tokenize(text: str) -> list[int]:
    """Hash-based tokenization for inference without a stored vocabulary.

    Maps each whitespace-delimited token to an index in [1, VOCAB_SIZE-1]
    via ``hashlib.md5`` — a fully deterministic hash that produces the
    same token IDs on every platform and Python process regardless of
    ``PYTHONHASHSEED``.  Index 0 is reserved for padding.

    This must stay in sync with ``_tokenize`` in ``training/train.py``
    so that a checkpoint trained on Colab loads and infers correctly on
    Windows (or any other machine).
    """
    tokens = text.lower().split()
    ids = [
        int(hashlib.md5(t.encode("utf-8"), usedforsecurity=False).hexdigest(), 16)
        % (_DEFAULT_VOCAB_SIZE - 1) + 1
        for t in tokens
    ]

    # Pad or truncate to CHUNK_SIZE
    if len(ids) < _CHUNK_SIZE:
        ids = ids + [0] * (_CHUNK_SIZE - len(ids))
    else:
        ids = ids[:_CHUNK_SIZE]

    return ids
