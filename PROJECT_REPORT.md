# StressDetect — MindView
## A Deep Learning–Powered Stress Detection and Intervention Platform

---

## Title Page

**Project Title:** StressDetect — MindView: An End-to-End Stress Detection Platform  
**Technology Stack:** Python · PyTorch · FastAPI · Streamlit · SQLite · Docker  
**Repository:** https://github.com/anant-925/StressDetect_MindView  
**License:** MIT  
**Classification:** Research / Educational Prototype — Not a Medical Device

---

## Table of Contents

1. [Summary](#1-summary)
   - 1.1 Brief Overview
   - 1.2 Objectives
2. [Introduction](#2-introduction)
   - 2.1 Background and Context
   - 2.2 Problem Statement
   - 2.3 Project Objectives
   - 2.4 Scope of the Project
3. [System Requirements](#3-system-requirements)
   - 3.1 Functional Requirements
4. [Design and Implementation](#4-design-and-implementation)
   - 4.1 System Architecture
   - 4.2 UML Diagrams
   - 4.3 AI & ML Algorithms
   - 4.4 Algorithms and Complexities
   - 4.5 Code Snippets
   - 4.6 Output Descriptions
5. [Conclusion](#5-conclusion)
   - 5.1 Summary & Achievement of Objectives
   - 5.2 Future Work and Recommendations
6. [References](#6-references)

---

## 1. Summary

### 1.1 Brief Overview

**StressDetect — MindView** is an end-to-end, production-grade stress detection platform that accepts free-form natural language text from users and returns a quantified stress assessment together with personalised, safety-first mental-health interventions. The system spans the full ML stack: from raw text preprocessing through deep learning inference, temporal modelling, and a rule-based intervention engine, to a polished Streamlit web dashboard backed by a secured FastAPI REST API.

The platform is designed to operate in a single-process mode (FastAPI + Streamlit co-hosted on HuggingFace Spaces port 7860) or as separate microservices behind a reverse proxy. A Docker container and Makefile provide reproducible local setup, while a model-checkpoint download script enables zero-configuration deployment.

### 1.2 Objectives

| # | Objective |
|---|-----------|
| O1 | Detect stress in user-submitted text with high precision and recall, using a deep-learning classifier trained on publicly available social-media corpora. |
| O2 | Provide uncertainty-aware inference through Monte-Carlo (MC) Dropout ensemble passes so that near-boundary predictions are labelled "uncertain" rather than confidently misclassified. |
| O3 | Track each user's stress trajectory over time using a lightweight temporal model that computes velocity, adaptive thresholds, and volatility flags. |
| O4 | Generate personalised, context-sensitive interventions aligned to detected stress triggers (sleep, money, exams, work, etc.) via a three-layer recommendation engine. |
| O5 | Guarantee user safety through a crisis circuit-breaker that unconditionally surfaces emergency helplines when self-harm language is detected. |
| O6 | Persist session history with military-grade encryption (AES-256 / Fernet) and authenticate all sensitive API endpoints via JWT. |
| O7 | Support continuous model improvement through an RL-style feedback loop that combines user binary feedback with optional LLM-judge reward signals. |

---

## 2. Introduction

### 2.1 Background and Context

Mental health conditions — especially stress, anxiety, and depression — have reached global epidemic proportions. The World Health Organization estimates that depression and anxiety cost the global economy US $1 trillion annually in lost productivity. Social media, chat applications, and digital journalling have created enormous volumes of self-reported emotional text, presenting an opportunity to develop passive, always-available screening tools.

Natural Language Processing (NLP) and deep learning have demonstrated strong results on sentiment classification and emotion detection tasks. However, most academic systems stop at a single model score and do not translate that score into actionable guidance for users. This project bridges that gap by coupling a high-accuracy text classifier with a four-layer safety engine and a personalization feedback loop.

### 2.2 Problem Statement

Existing consumer wellness apps rely on hand-crafted questionnaires (e.g. PHQ-9, GAD-7) that require clinical overhead, are point-in-time only, and produce no immediate, tailored guidance. Free-text diary or messaging interfaces lack automated triage. Clinical resources are often inaccessible (cost, availability, stigma). The challenge is therefore to:

> *Build a lightweight, real-time NLP system that can infer stress levels from arbitrary free-text input, personalise its recommendations over time, protect user privacy, and escalate to emergency services when necessary.*

### 2.3 Project Objectives

The project was structured in five iterative phases:

- **Phase 1 — Data Pipeline:** Prepare a multi-domain dataset and implement a sliding-window PyTorch DataLoader.
- **Phase 2 — Model Architecture:** Implement a multi-channel 1D CNN with multi-head self-attention, temperature calibration, and temporal stress modelling.
- **Phase 3 — Secure Backend:** Build a FastAPI service with JWT authentication, bcrypt password storage, and Fernet-encrypted history.
- **Phase 4 — Intervention Engine:** Implement the four-layer safety-first recommendation engine with crisis detection.
- **Phase 5 — Dashboard & Feedback Loop:** Build a Streamlit dashboard and wire up an RL-style reward signal from user and LLM feedback.

### 2.4 Scope of the Project

**In scope:**
- Free-text stress detection for English-language input of arbitrary length
- Real-time personalised intervention delivery
- User account management, session history, and feedback collection
- Model retraining from feedback
- Docker-based deployment for HuggingFace Spaces

**Out of scope:**
- Multi-language support (beyond English)
- Speech / audio input
- Wearable physiological sensor integration
- Clinical diagnosis or medical advice
- HIPAA / GDPR compliance (prototype only)

---

## 3. System Requirements

### 3.1 Functional Requirements

#### FR-01 — User Registration and Authentication

- The system shall allow new users to register with a unique username and password.
- Passwords shall be hashed with bcrypt (cost factor ≥ 10) before storage.
- Successful registration and login shall return a JWT Bearer token valid for 7 days (configurable via `JWT_EXPIRATION_MINUTES`).
- All `/analyze`, `/history`, `/feedback`, and `/personalization` endpoints shall reject requests without a valid JWT.

#### FR-02 — Stress Analysis

- The system shall accept free-form text of any length via `POST /analyze`.
- The system shall preprocess the text (HTML stripping, emoji-to-text, URL removal, Unicode normalisation).
- The system shall run inference using the trained model and return:
  - `stress_score` (float in [0, 1])
  - `stress_label` ("low" | "moderate" | "high" | "uncertain")
  - `confidence` (float in [0, 1])
  - `uncertainty` (float — ensemble std deviation)
  - `attention_weights` (list of per-token importance floats)
  - `temporal_data` (velocity, adaptive threshold, volatility flag)
  - `interventions` (list of recommended actions)
  - `is_crisis` (bool) and `crisis_message` (str or null)

#### FR-03 — Temporal Stress Tracking

- The system shall maintain an encrypted per-user sliding-window history (up to 50 sessions).
- The system shall compute stress velocity (slope over last 5 sessions) using linear regression.
- The system shall compute an adaptive personalised threshold: `min(max(μ + 1.5σ, 0.5), 0.95)`.
- The system shall flag volatility when the standard deviation of the last 5 scores exceeds 0.25.

#### FR-04 — Intervention Engine

- **Layer 1 (Crisis Circuit Breaker):** If the text matches any crisis keyword pattern (suicidal ideation, self-harm), the system shall immediately return the emergency payload including the 988 Suicide & Crisis Lifeline and halt all further processing.
- **Layer 2 (Context Matcher):** The system shall detect up to 8 trigger categories (sleep, money, exam, work, relationship, health, grief, loneliness) and return 2 targeted interventions per matched trigger.
- **Layer 3 (Preventive Nudges):** If the temporal model flags the user as volatile, the system shall prepend grounding exercises.
- **Layer 4 (Escalation):** If the user has had 3+ consecutive above-threshold sessions, the system shall append a professional-referral intervention.

#### FR-05 — Session History

- The system shall persist every analysis session in a SQLite database (encrypted at rest via Fernet).
- The system shall allow authenticated retrieval of the last N sessions via `GET /history`.
- Session records shall include: timestamp, score, label, temporal data, interventions, and attention weights.

#### FR-06 — User Feedback and Retraining

- The system shall accept binary feedback (1 = correct, 0 = wrong) via `POST /feedback`.
- The system shall optionally invoke an LLM judge (OpenAI or Gemini) to produce a reward signal.
- The combined reward shall be stored for experience replay during model retraining (`python training/retrain.py`).

#### FR-07 — Model Management

- The system shall support three model variants: `cnn` (OptimizedMultichannelCNN), `deberta` (DeBERTa-v3-Small), and `minilm` (MiniLM-L6-v2).
- The default checkpoint shall be downloaded automatically via `scripts/download_model.py` from HuggingFace Hub.
- `GET /model/info` shall return model type, vocabulary size, and decision threshold.

#### FR-08 — Health Check

- `GET /health` shall return uptime, model loaded status, and database session count.

---

## 4. Design and Implementation

### 4.1 System Architecture

The platform follows a three-tier architecture:

```
┌─────────────────────────────────────────┐
│         Streamlit Dashboard (UI)        │
│  Port 7860  —  ui/app.py               │
│  Pages: Dashboard | History | Settings  │
└───────────────┬─────────────────────────┘
                │ HTTP REST (JSON)
                ▼
┌─────────────────────────────────────────┐
│         FastAPI Backend (API)           │
│  Port 8000  —  api/main.py             │
│  ┌─────────┐  ┌───────────┐  ┌───────┐ │
│  │ Auth /  │  │ Inference │  │History│ │
│  │ JWT     │  │ Pipeline  │  │ CRUD  │ │
│  └─────────┘  └─────┬─────┘  └───┬───┘ │
└────────────────────-│─────────────│─────┘
                       │             │
        ┌──────────────┘             │
        ▼                            ▼
┌──────────────────┐      ┌──────────────────┐
│  ML / AI Layer   │      │  Persistence     │
│                  │      │  Layer           │
│ ┌──────────────┐ │      │                  │
│ │ CNN/DeBERTa  │ │      │ SQLite DB        │
│ │ (PyTorch)    │ │      │ (users, sessions,│
│ └──────────────┘ │      │  feedback)       │
│ ┌──────────────┐ │      │                  │
│ │ Temporal     │ │      │ Fernet AES-256   │
│ │ Model        │ │      │ encryption       │
│ └──────────────┘ │      └──────────────────┘
│ ┌──────────────┐ │
│ │ Intervention │ │
│ │ Engine       │ │
│ └──────────────┘ │
└──────────────────┘
```

**Module Layout:**

| Module | Path | Purpose |
|--------|------|---------|
| Entry Point | `app.py` | Boots FastAPI + Streamlit in one process (Spaces mode) |
| REST API | `api/main.py` | All routes, auth, inference, persistence |
| UI | `ui/app.py` | Three-page Streamlit dashboard |
| CNN Model | `models/architecture.py` | OptimizedMultichannelCNN, DeBERTa, MiniLM |
| Temporal Model | `models/temporal_stress_profile.py` | Velocity, threshold, volatility |
| Intervention Engine | `intervention/engine.py` | Four-layer safety engine |
| Secure Temporal Wrapper | `intervention/temporal_model.py` | Encrypt/decrypt temporal history |
| Text Preprocessing | `utils/text_preprocessing.py` | HTML, emoji, URL, normalisation |
| Security | `security/auth.py` | JWT, bcrypt, Fernet |
| Database | `database/db.py` | SQLite CRUD |
| Feedback | `database/feedback.py` | Experience replay store |
| Training | `training/train.py` | Full model training script |
| Retraining | `training/retrain.py` | Feedback-based fine-tuning |
| Data Pipeline | `data/dataset.py` | PyTorch Dataset with sliding windows |
| Data Preprocessing | `scripts/data_preprocessing.py` | Dataset preparation |
| Reward Utils | `utils/reward.py` | RL reward computation |
| LLM Reward | `utils/llm_reward.py` | OpenAI/Gemini judge integration |
| Sentiment | `utils/sentiment.py` | Sentiment dampening |

---

### 4.2 UML Diagrams

#### 4.2.1 Use Case Diagram

```
                  ┌───────────────────────────────────────────────────┐
                  │               StressDetect System                  │
                  │                                                     │
  ┌──────────┐    │   ┌──────────────────┐  ┌─────────────────────┐  │
  │  Visitor  │───┼──▶│  Register        │  │  Download Model     │  │
  └──────────┘    │   └──────────────────┘  └─────────────────────┘  │
                  │                                                     │
  ┌──────────┐    │   ┌──────────────────┐                            │
  │   User   │───┼──▶│  Login           │                            │
  └────┬─────┘    │   └──────────────────┘                            │
       │          │                                                     │
       ├──────────┼──▶┌──────────────────┐                            │
       │          │   │  Submit Text for  │                            │
       │          │   │  Stress Analysis  │◀──────extends──────┐       │
       │          │   └──────────────────┘                     │       │
       │          │           │                        ┌────────────┐ │
       │          │           │ includes               │  Crisis    │ │
       │          │           ▼                        │  Detection │ │
       │          │   ┌──────────────────┐             └────────────┘ │
       │          │   │  View            │                            │
       ├──────────┼──▶│  Interventions   │                            │
       │          │   └──────────────────┘                            │
       │          │                                                     │
       ├──────────┼──▶┌──────────────────┐                            │
       │          │   │  View Session    │                            │
       │          │   │  History         │                            │
       │          │   └──────────────────┘                            │
       │          │                                                     │
       ├──────────┼──▶┌──────────────────┐                            │
       │          │   │  Submit Feedback  │                            │
       │          │   └──────────────────┘                            │
       │          │                                                     │
       └──────────┼──▶┌──────────────────┐                            │
                  │   │  View            │                            │
                  │   │  Personalization │                            │
                  │   └──────────────────┘                            │
                  │                                                     │
  ┌──────────┐    │   ┌──────────────────┐                            │
  │  Admin/  │───┼──▶│  Retrain Model   │                            │
  │  Trainer │    │   │  from Feedback   │                            │
  └──────────┘    │   └──────────────────┘                            │
                  └───────────────────────────────────────────────────┘
```

#### 4.2.2 Activity Diagram — Stress Analysis Flow

```
User submits text
       │
       ▼
┌─────────────┐
│ Clean Text  │  (HTML strip, emoji→text, URL remove, NFKC normalize)
└──────┬──────┘
       │
       ▼
┌─────────────────┐
│ Short-input      │ ─── single-word "stressed/fine" → preset score
│ Handler          │
└────────┬────────┘
         │ (long input)
         ▼
┌────────────────────┐
│ Tokenise (MD5-hash │ vocab_size=10,000 · chunk_size=200 · stride=50
│ sliding-window)    │
└────────┬───────────┘
         │
         ▼
┌────────────────────┐
│ MC-Dropout         │ 3 stochastic forward passes
│ Ensemble Inference │ → mean probability + std deviation
└────────┬───────────┘
         │
         ▼
┌────────────────────┐
│ Signal Filter      │ Dampen if <3 content words
│ Contrast Filter    │ Dampen if contrast conjunction ("but happy")
│ Sentiment Dampen   │ Positive sentiment → lower score
└────────┬───────────┘
         │
         ▼
┌────────────────────┐
│ Classify Level     │ low / uncertain / moderate / high
└────────┬───────────┘
         │
         ▼
┌────────────────────┐
│ Temporal Model     │ Load encrypted history → decrypt →
│ (SecureTemporalModel│ compute velocity / threshold / volatility →
│                    │ re-encrypt
└────────┬───────────┘
         │
         ▼
┌────────────────────────────┐
│ Intervention Engine        │
│ Layer 1: Crisis check ─────┼──▶ HALT → return emergency payload
│ Layer 2: Trigger matching  │
│ Layer 3: Volatile nudges   │
│ Layer 4: Escalation check  │
└────────┬───────────────────┘
         │
         ▼
┌────────────────────┐
│ Persist session to │
│ SQLite (encrypted) │
└────────┬───────────┘
         │
         ▼
   Return JSON response
```

#### 4.2.3 Sequence Diagram — POST /analyze

```
User       Streamlit UI       FastAPI          Model       DB/TemporalModel  Intervention
 │               │               │               │               │                │
 │ submit text   │               │               │               │                │
 │──────────────▶│               │               │               │                │
 │               │ POST /analyze │               │               │                │
 │               │──────────────▶│               │               │                │
 │               │               │ verify JWT    │               │                │
 │               │               │───────────────X               │                │
 │               │               │ clean_text()  │               │                │
 │               │               │───────────────▶               │                │
 │               │               │ tokenize()    │               │                │
 │               │               │───────────────▶               │                │
 │               │               │ MC-Dropout x3 │               │                │
 │               │               │───────────────▶               │                │
 │               │               │  mean+std     │               │                │
 │               │               │◀──────────────│               │                │
 │               │               │ apply filters │               │                │
 │               │               │───────────────X               │                │
 │               │               │ get user history              │                │
 │               │               │───────────────────────────────▶                │
 │               │               │ temporal.process(score)       │                │
 │               │               │───────────────────────────────▶                │
 │               │               │  TemporalAnalysis + new_enc   │                │
 │               │               │◀──────────────────────────────│                │
 │               │               │ engine.recommend(text)                         │
 │               │               │────────────────────────────────────────────────▶
 │               │               │  RecommendationPayload                         │
 │               │               │◀───────────────────────────────────────────────│
 │               │               │ save_session()                │                │
 │               │               │───────────────────────────────▶                │
 │               │ AnalysisResponse                              │                │
 │               │◀──────────────│               │               │                │
 │ render UI     │               │               │               │                │
 │◀──────────────│               │               │               │                │
```

#### 4.2.4 Class Diagram

```
┌────────────────────────────────────────────────────────────────────────────┐
│                        StressDetect — Core Classes                          │
└────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────┐
│  OptimizedMultichannelCNN        │
│──────────────────────────────────│
│ + embedding: nn.Embedding        │
│ + convs: nn.ModuleList           │
│ + attention: MultiHeadSelfAttn.  │
│ + classifier: nn.Sequential      │
│──────────────────────────────────│
│ + forward(x) → (logits, weights) │
│ + enable_dropout()               │
└────────────────┬─────────────────┘
                 │ uses
         ┌───────┴────────────┐
         │                    │
┌────────▼──────────┐  ┌──────▼──────────────────┐
│MultiHeadSelfAttn. │  │TemperatureScaling        │
│───────────────────│  │──────────────────────────│
│+ num_heads: int   │  │+ temperature: nn.Param.  │
│+ d_k: int         │  │──────────────────────────│
│+ query, key, value│  │+ forward(logits)         │
│+ out_proj         │  │+ calibrate(logits,labels)│
│───────────────────│  └──────────────────────────┘
│+ forward(x)       │
└───────────────────┘

┌──────────────────────────────────┐
│  TemporalStressProfile           │
│──────────────────────────────────│
│ - _history: deque[tuple]         │
│ - _max_history: int              │
│ - _velocity_window: int          │
│ - _volatility_threshold: float   │
│──────────────────────────────────│
│ + add_score(score) → Analysis    │
│ - _compute_velocity() → float    │
│ - _compute_adaptive_threshold()  │
│ - _compute_volatility() → float  │
└──────────────────────────────────┘

┌──────────────────────────────────┐
│  SecureTemporalModel             │
│──────────────────────────────────│
│ - _max_history: int              │
│──────────────────────────────────│
│ + process(score, enc_history)    │
│   → (TemporalAnalysis, str)      │
└────────────────┬─────────────────┘
                 │ wraps
┌────────────────▼─────────────────┐
│  TemporalStressProfile           │
└──────────────────────────────────┘

┌──────────────────────────────────┐
│  RecommendationEngine            │
│──────────────────────────────────│
│ (no state — pure functions)      │
│──────────────────────────────────│
│ + recommend(text, score,         │
│             is_volatile,         │
│             requires_escalation) │
│   → RecommendationPayload        │
└──────────────────────────────────┘

┌──────────────────────────────────┐
│  DatabaseManager                 │
│──────────────────────────────────│
│ - _conn: sqlite3.Connection      │
│──────────────────────────────────│
│ + create_user(username, hash)    │
│ + get_user(username) → dict      │
│ + update_encrypted_history(...)  │
│ + save_session(...) → int        │
│ + get_sessions(username) → list  │
└──────────────────────────────────┘

┌──────────────────────────────────┐
│  FeedbackStore                   │
│──────────────────────────────────│
│ - _conn: sqlite3.Connection      │
│──────────────────────────────────│
│ + add_feedback(...)              │
│ + get_pending_feedback() → list  │
│ + get_stats(username) → dict     │
└──────────────────────────────────┘
```

#### 4.2.5 State Diagram — Stress Level State Machine

```
                   ┌──────────────┐
                   │  [*] Start   │
                   └──────┬───────┘
                           │ user submits text
                           ▼
                   ┌──────────────┐
                   │  Preprocessing│
                   └──────┬───────┘
                           │ text cleaned
                           ▼
             ┌─────────────────────────┐
             │       Inference          │
             │  (MC-Dropout Ensemble)   │
             └─────────────┬───────────┘
                            │
              ┌─────────────┼──────────────────┐
              │             │                  │
              ▼             ▼                  ▼
     ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
     │   CRISIS     │ │  UNCERTAIN   │ │ STRESS SCORE │
     │  DETECTED    │ │  (±0.07 of   │ │  COMPUTED    │
     │  [HALT]      │ │   threshold) │ │              │
     └──────┬───────┘ └──────┬───────┘ └──────┬───────┘
            │                │                 │
            ▼                │         ┌───────┴────────┐
     ┌────────────┐           │         │                │
     │ Emergency  │           │    ┌────▼────┐   ┌───────▼──┐
     │ Payload    │           │    │MODERATE │   │  HIGH    │
     │ + 988 Line │           │    │ (thresh │   │ (thresh+ │
     └────────────┘           │    │+0.10 to │   │  0.25+)  │
                              │    │ +0.25)  │   └──────┬───┘
                              │    └────┬────┘          │
                              │         │               │ 3+ sessions
                              │    ┌────▼────────┐      │ above threshold
                              └───▶│  TEMPORAL   │◀─────┘
                                   │  ANALYSIS   │
                                   │ + VOLATILE? │
                                   └──────┬──────┘
                                          │
                                          ▼
                                   ┌────────────┐
                                   │INTERVENTION│
                                   │  SELECTED  │
                                   └──────┬─────┘
                                          │
                                          ▼
                                   ┌────────────┐
                                   │  SESSION   │
                                   │  SAVED     │
                                   └──────┬─────┘
                                          │
                                          ▼
                                   ┌────────────┐
                                   │  RESPONSE  │
                                   │  RETURNED  │
                                   └────────────┘
```

#### 4.2.6 Sequence Diagram — User Registration and Login

```
User       Streamlit        FastAPI          bcrypt / JWT       SQLite
  │             │               │                  │               │
  │  fill form  │               │                  │               │
  │────────────▶│               │                  │               │
  │             │ POST /register│                  │               │
  │             │──────────────▶│                  │               │
  │             │               │ hash_password()  │               │
  │             │               │─────────────────▶│               │
  │             │               │   bcrypt_hash    │               │
  │             │               │◀─────────────────│               │
  │             │               │ create_user()    │               │
  │             │               │──────────────────────────────────▶
  │             │               │ create_jwt_token()│               │
  │             │               │─────────────────▶│               │
  │             │               │  access_token    │               │
  │             │               │◀─────────────────│               │
  │             │ {token}       │                  │               │
  │             │◀──────────────│                  │               │
  │ stored token│               │                  │               │
  │◀────────────│               │                  │               │
  │             │               │                  │               │
  │ POST /login │               │                  │               │
  │────────────▶│──────────────▶│ get_user()       │               │
  │             │               │──────────────────────────────────▶
  │             │               │ verify_password()│               │
  │             │               │─────────────────▶│               │
  │             │               │ create_jwt_token()               │
  │             │               │─────────────────▶│               │
  │             │ {token}       │                  │               │
  │◀────────────│◀──────────────│                  │               │
```

---

### 4.3 AI & ML Algorithms Implementation

#### 4.3.1 Text Preprocessing Pipeline

The `clean_text` function in `utils/text_preprocessing.py` applies eight operations in sequence:

1. **HTML entity unescaping** — `html.unescape()` converts `&amp;` → `&`, `&#39;` → `'`.
2. **HTML/XML tag stripping** — Regex `<[^>]{0,200}>` (length-capped against ReDoS).
3. **Emoji-to-text mapping** — 50+ emoji replaced with semantically equivalent English words (e.g. `😰` → `anxious`).
4. **URL removal** — `https?://\S+` replaced with a space.
5. **Email address removal** — Full RFC-5321 pattern replaced.
6. **Repeated-character compression** — `soooooo` → `soo` (4+ repetitions compressed to 2).
7. **Unicode NFKC normalisation** — Ligatures, full-width characters, and directional marks resolved.
8. **Whitespace normalisation** — Multiple spaces/newlines collapsed to a single space.

#### 4.3.2 Hash-Based Tokeniser

The tokeniser is intentionally identical between training (`training/train.py`) and inference (`api/main.py`), ensuring there is no train-serve skew:

```python
def _tokenize(text: str) -> list[int]:
    tokens = text.lower().split()
    return [
        int(hashlib.md5(t.encode("utf-8"), usedforsecurity=False).hexdigest(), 16)
        % (VOCAB_SIZE - 1) + 1
        for t in tokens
    ]
```

- Vocabulary size: **10,000** (index 0 reserved for padding).
- Uses MD5 for platform-independent determinism (`PYTHONHASHSEED` is not involved).
- Long texts are split into overlapping chunks of 200 tokens with a stride of 50 (75% overlap).

#### 4.3.3 OptimizedMultichannelCNN

The primary model is a five-layer architecture:

1. **Embedding Layer** — `nn.Embedding(10000, 128)` with stop-word dampening (stop-word embeddings scaled to 0.3× their original magnitude to prevent attention over-emphasis).

2. **Parallel Conv1D Branches** — Three branches with kernel sizes 2, 3, and 5, each producing 64 feature maps. All three outputs are trimmed to the minimum sequence length before concatenation (the *min_len guard* preventing shape mismatch).

3. **Concatenation** — 192 total filters (3 × 64) concatenated along the feature axis.

4. **Multi-Head Self-Attention** — 4 heads, each operating in a 48-dimensional subspace (`192 / 4`). Scaled dot-product attention: `Attention(Q,K,V) = softmax(QKᵀ / √dₖ) · V`.

5. **Classification Head** — `Linear(192, 128) → ReLU → Dropout(0.3) → Linear(128, 2)`.

Temperature scaling (post-hoc calibration, Guo et al. 2017) can optionally wrap any classifier; it learns a single temperature scalar `T` on a held-out validation set by minimising NLL with L-BFGS.

#### 4.3.4 Transformer Wrappers

Two transformer wrappers are provided for comparison:

- **DeBERTaStressClassifier** — Wraps `microsoft/deberta-v3-small` from HuggingFace. Uses the `[CLS]` token representation fed through a two-layer classification head with dropout.
- **MiniLMStressClassifier** — Wraps `sentence-transformers/all-MiniLM-L6-v2`. Mean-pools all token embeddings before the classification head.

Both wrappers share the same interface as `OptimizedMultichannelCNN` so the training and inference pipelines are model-agnostic.

#### 4.3.5 MC-Dropout Uncertainty Estimation

At inference time, the model is set to evaluation mode except for Dropout layers, which are kept in training mode. Three stochastic forward passes are run under a threading lock (to prevent concurrent requests interfering with Dropout state). The mean and standard deviation of the three stress probabilities are reported:

```python
probs = []
with _inference_lock:
    model.eval()
    model.apply(_enable_dropout)
    for _ in range(_ENSEMBLE_PASSES):   # 3 passes
        with torch.no_grad():
            logits, _ = model(input_ids)
        prob = F.softmax(logits, dim=-1)[:, 1].item()
        probs.append(prob)
    model.eval()  # restore full eval mode

mean_prob = statistics.mean(probs)
uncertainty = statistics.stdev(probs)
```

If `uncertainty > 0.08`, the prediction is flagged as uncertain regardless of the raw score.

#### 4.3.6 Temporal Stress Modelling

The `TemporalStressProfile` class tracks up to 50 `(timestamp, score)` pairs per user. Three statistics are computed on each new entry:

**Stress Velocity** — 1-D linear regression slope over the last 5 scores:

```
v_s = coeffs[0]  of  np.polyfit(x=[0..4], y=last_5_scores, deg=1)
```

A positive velocity indicates deteriorating stress; negative indicates improvement.

**Adaptive Threshold** — Personalised trigger point:

```
τ = min(max(μ + 1.5σ, 0.5), 0.95)
```

where μ and σ are the mean and standard deviation over all stored history. This auto-adjusts for users with chronically high or low baseline stress levels.

**Volatility** — Standard deviation of the last 5 scores. Flagged volatile when `σ > 0.25`.

#### 4.3.7 Intervention Engine (Four Layers)

```
Layer 1: Crisis Circuit Breaker  [regex, ALWAYS runs first, HALTS on match]
Layer 2: Context Trigger Matcher [8 keyword categories → targeted interventions]
Layer 3: Preventive Nudges       [shown when temporal model flags volatility]
Layer 4: Escalation Tracker      [shown when 3+ consecutive above-threshold sessions]
```

Layer 2 recognises eight categories via compiled regexes: `sleep`, `money`, `exam`, `work`, `relationship`, `health`, `grief`, and `loneliness`. Each trigger maps to two curated interventions (breathing, grounding, cognitive reframe, or resource referral).

#### 4.3.8 RL-Style Feedback Loop

The feedback loop implements a simplified experience-replay mechanism:

- **User reward:** `+1.0` if the prediction was correct, `-1.0` if wrong.
- **LLM reward:** An optional LLM judge (OpenAI GPT-4 or Google Gemini) is called asynchronously via `utils/llm_reward.py`. It returns `+1` or `-1` based on its own assessment of the text-label pair.
- **Combined reward:** `(user_r + llm_r) / 2`.
- **Loss weight:** `reward_to_weight(reward) = 1.5` (feedback samples are weighted 50% higher than default samples during retraining).

The retraining script (`training/retrain.py`) loads feedback samples from the `FeedbackStore`, applies the weighted loss, and fine-tunes the existing checkpoint.

#### 4.3.9 Sentiment Dampening

`utils/sentiment.py` provides a lightweight VADER-style positive-negative word count balance:

```
sentiment_score = (positive_count - negative_count) / max(total_count, 1)
```

When `sentiment_score > 0.2` (predominantly positive text), the model's stress probability is multiplied by `0.85` to reduce false-positive rates on happy or neutral text.

---

### 4.4 Algorithms and Their Complexities

| Algorithm | Description | Time Complexity | Space Complexity |
|-----------|-------------|-----------------|------------------|
| `clean_text` | 8-step regex pipeline | O(n · k) where n = text length, k = emoji count | O(n) |
| `_tokenize` (MD5 hash) | Hash each whitespace-delimited token | O(n · L) where L = avg token length | O(n) |
| Sliding-window chunking | Generate overlapping chunks | O(n / stride) | O(n) |
| Embedding lookup | `nn.Embedding` forward | O(batch · seq · embed_dim) | O(vocab · embed) |
| Conv1D (3 branches) | Parallel 1D convolutions | O(B · L · embed · kernel · filters) | O(B · L · filters) |
| Multi-head self-attention | Scaled dot-product | O(B · H · L²) | O(B · H · L²) |
| CNN forward pass | Full model | O(B · L · filters · L) | O(B · L · filters) |
| MC-Dropout ensemble | 3 × CNN forward pass | 3 × O(CNN) | O(CNN) |
| Stress velocity | `np.polyfit` (deg=1) over n≤5 points | O(n) | O(n) |
| Adaptive threshold | Mean + std over history ≤ 50 | O(H) where H ≤ 50 | O(H) |
| Volatility | `np.std` over n≤5 points | O(n) | O(n) |
| Crisis regex match | RE compiled pattern | O(n) | O(1) |
| Trigger matching | 8 compiled regexes | O(8 · n) | O(1) |
| Intervention sorting | Sort by priority | O(k log k) interventions | O(k) |
| bcrypt hash | Cost factor 12 | O(2^12) (constant ~300ms) | O(1) |
| JWT encode/decode | HMAC-SHA256 | O(payload size) | O(1) |
| Fernet encrypt/decrypt | AES-256-CBC | O(data size) | O(data size) |
| `save_session` (SQLite) | Single INSERT | O(log N) B-tree | O(1) |
| `get_sessions` (SQLite) | SELECT with INDEX | O(log N + K) | O(K sessions) |

*B = batch size, L = sequence length, H = number of attention heads, N = total rows, K = returned rows.*

---

### 4.5 Code Snippets

#### 4.5.1 Text Preprocessing

```python
# utils/text_preprocessing.py

def clean_text(text: str, *, normalize_repeated: bool = True) -> str:
    if not text or not isinstance(text, str):
        return ""

    text = html.unescape(text)                          # 1. HTML entities
    text = _HTML_TAG_RE.sub(" ", text)                  # 2. Strip tags
    for emoji_char, replacement in _EMOJI_TEXT.items(): # 3. Emoji→text
        if emoji_char in text:
            text = text.replace(emoji_char, f" {replacement} ")
    text = _URL_RE.sub(" ", text)                       # 4. Remove URLs
    text = _EMAIL_RE.sub(" ", text)                     # 5. Remove emails
    if normalize_repeated:
        text = _REPEATED_CHAR_RE.sub(r"\1\1", text)     # 6. Compress chars
    text = unicodedata.normalize("NFKC", text)          # 7. Unicode NFKC
    text = _WHITESPACE_RE.sub(" ", text).strip()        # 8. Whitespace
    return text
```

#### 4.5.2 Multi-Head Self-Attention Forward Pass

```python
# models/architecture.py  (MultiHeadSelfAttention.forward)

def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    B, L, H = x.shape

    # Project and split into heads: (B, num_heads, L, d_k)
    q = self.query(x).view(B, L, self.num_heads, self.d_k).transpose(1, 2)
    k = self.key(x).view(B, L, self.num_heads, self.d_k).transpose(1, 2)
    v = self.value(x).view(B, L, self.num_heads, self.d_k).transpose(1, 2)

    # Scaled dot-product attention: (B, num_heads, L, L)
    scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
    attn = F.softmax(scores, dim=-1)
    attn = self.attn_dropout(attn)

    # Context: (B, num_heads, L, d_k) → merge: (B, L, H)
    context = torch.matmul(attn, v)
    context = context.transpose(1, 2).contiguous().view(B, L, H)

    # Output projection and mean-pool
    out = self.out_proj(context)          # (B, L, H)
    pooled = out.mean(dim=1)              # (B, H)

    # Per-token importance weights for heatmap rendering
    weights = attn.mean(dim=1).mean(dim=1)  # (B, L)

    return pooled, weights
```

#### 4.5.3 MC-Dropout Ensemble Inference

```python
# api/main.py  (simplified)

_ENSEMBLE_PASSES = 3

def _mc_inference(model, input_ids):
    probs = []
    with _inference_lock:
        model.eval()
        model.apply(_enable_dropout)   # keep Dropout in train mode
        for _ in range(_ENSEMBLE_PASSES):
            with torch.no_grad():
                logits, attn_weights = model(input_ids)
            prob = F.softmax(logits, dim=-1)[:, 1].item()
            probs.append(prob)
        model.eval()                   # restore full eval mode

    mean_prob  = statistics.mean(probs)
    uncertainty = statistics.stdev(probs)
    return mean_prob, uncertainty, attn_weights
```

#### 4.5.4 Adaptive Threshold Computation

```python
# models/temporal_stress_profile.py

def _compute_adaptive_threshold(self) -> float:
    scores = self.scores
    if len(scores) < MIN_CALIBRATION_POINTS:   # < 3 data points
        return ADAPTIVE_THRESHOLD_FLOOR        # default 0.5

    arr = np.array(scores, dtype=np.float64)
    mu    = float(np.mean(arr))
    sigma = float(np.std(arr))

    raw = mu + 1.5 * sigma
    return min(max(raw, 0.5), 0.95)            # clamped to [0.5, 0.95]
```

#### 4.5.5 Crisis Circuit Breaker

```python
# intervention/engine.py

_CRISIS_PATTERN = re.compile(
    r"\b(suicide|suicidal|kill\s+myself|end\s+it\s+all"
    r"|end\s+my\s+life|want\s+to\s+die|self[- ]?harm(?:ing)?)\b",
    re.IGNORECASE,
)

def recommend(self, text, stress_score=0.0, is_volatile=False,
              requires_escalation=False):
    payload = RecommendationPayload()

    # Layer 1: Crisis check — runs FIRST, halts everything else
    if _CRISIS_PATTERN.search(text):
        payload.is_crisis = True
        payload.crisis_message = _EMERGENCY_MESSAGE
        payload.interventions.append(Intervention(
            title="Immediate Crisis Support",
            description=_EMERGENCY_MESSAGE,
            category="emergency",
            priority=10,
        ))
        return payload   # HALT

    # Layers 2–4 only reached when no crisis detected
    ...
```

#### 4.5.6 RL Reward Computation

```python
# utils/reward.py

def compute_combined_reward(user_feedback: int,
                             llm_reward: int | None = None) -> float:
    user_r = 1.0 if user_feedback == 1 else -1.0
    if llm_reward is not None:
        return (user_r + float(llm_reward)) / 2.0
    return user_r

def reward_to_weight(reward: float) -> float:
    # Feedback samples are weighted 50% higher than baseline
    return 1.5 if reward != 0.0 else 1.0
```

#### 4.5.7 JWT + bcrypt Authentication

```python
# security/auth.py

def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

def verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))

def create_jwt_token(data: dict, expires_delta=None) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(minutes=JWT_EXPIRATION_MINUTES)
    )
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, JWT_SECRET_KEY, algorithm="HS256")
```

#### 4.5.8 Fernet Encryption of Stress History

```python
# security/auth.py

def encrypt_data(data: Any) -> str:
    plaintext = json.dumps(data).encode("utf-8")
    return _fernet.encrypt(plaintext).decode("utf-8")

def decrypt_data(encrypted: str) -> Any:
    try:
        plaintext = _fernet.decrypt(encrypted.encode("utf-8"))
        return json.loads(plaintext.decode("utf-8"))
    except InvalidToken:
        return None
```

---

### 4.6 Output Descriptions

> *Note: The following section describes the screens and outputs as rendered by the running application.*

#### 4.6.1 Login / Registration Page

When the user first opens the dashboard, they are presented with a two-tab panel:
- **Register** — username and password fields with a "Create Account" button. On success, the token is stored in session state and the user is redirected to the Dashboard page.
- **Login** — Same fields with a "Sign In" button.

The page uses the warm limestone colour palette (parchment background, terracotta accents) and Playfair Display / DM Sans fonts.

#### 4.6.2 Dashboard — Stress Check-In

The main page contains:

1. **Text Input Area** — A wide `st.text_area` labelled "How are you feeling today?" accepting free-form input of unlimited length.

2. **Analyse Button** — Triggers `POST /analyze`. A spinner is shown during the API call.

3. **Stress Gauge** — A Plotly gauge chart (green/amber/red) displaying the stress score in [0, 1] with a needle and coloured arcs. The score is shown as a large centre label.

4. **Stress Level Card** — A coloured card displaying the level ("Low", "Moderate", "High", "Uncertain"), confidence percentage, and uncertainty value from the MC-Dropout ensemble.

5. **Temporal Context** — Shows:
   - Stress velocity (rising ↑, falling ↓, or stable ↔)
   - Current adaptive threshold
   - Volatile flag (⚡ if triggered)

6. **Intervention Panel** — Collapsible cards for each recommended intervention, colour-coded by category (emergency = red, breathing = blue, grounding = green, cognitive = amber, resource = teal). Each card shows the title, description, and a category badge.

7. **Word Attention Heatmap** — An HTML/CSS heatmap rendered via `st.components.v1.html` showing each token from the input coloured from white (low attention) to deep indigo (high attention). This allows the user to see which words drove the prediction.

8. **Breathing Animation** — An animated CSS circle that expands and contracts on a 4-7-8 breathing cycle, shown when stress level is high or moderate.

9. **Feedback Buttons** — Two buttons: "✓ Correct" and "✗ Incorrect", which call `POST /feedback` and store the reward signal.

#### 4.6.3 History & Analytics Page

Three sub-tabs:

1. **Timeline** — A Plotly line chart of all historical stress scores on the x-axis (datetime) with a horizontal dashed line at the current adaptive threshold. Points are coloured by stress level (green = low, amber = moderate, red = high). Hovering a point shows the session timestamp, score, and matched triggers.

2. **Distribution** — A Plotly donut chart showing the proportion of sessions by stress level (Low / Moderate / High / Uncertain).

3. **Session Log** — A `st.dataframe` table listing all sessions with columns for timestamp, stress label, score, triggers matched, and whether a crisis was detected. A "Download CSV" button allows exporting the full history.

#### 4.6.4 Settings Page

- **Account info** — Username and registration date.
- **Model info** — Live data from `GET /model/info`: model type, vocabulary size, decision threshold.
- **Feedback stats** — Summary from `GET /feedback/stats`: total feedbacks submitted, accuracy rate.
- **Sign Out** — Clears the JWT from session state and redirects to the login page.

#### 4.6.5 API Swagger Docs

The FastAPI `/docs` endpoint renders an auto-generated Swagger UI listing all endpoints with request schemas, example responses, and a "Try it out" interface. All authenticated endpoints show the Bearer token lock icon.

#### 4.6.6 Crisis Response Screen

When crisis keywords are detected, the Dashboard replaces the standard gauge and interventions with a full-width red alert card:

```
⚠️  We noticed language that suggests you may be in crisis.
    You are not alone. Please reach out to a professional:

    📞  988 Suicide & Crisis Lifeline — Call or text 988 (US)
    📞  Crisis Text Line — Text HOME to 741741
    🌐  International Association for Suicide Prevention
```

No stress score or other analysis is shown — the entire UI pivots to safety resources.

---

## 5. Conclusion

### 5.1 Summary & Achievement of Objectives

| Objective | Status | Notes |
|-----------|--------|-------|
| O1 — Accurate stress detection | ✅ Achieved | CNN with multi-head attention, trained on unified multi-domain corpus; F1-calibrated threshold |
| O2 — Uncertainty-aware inference | ✅ Achieved | MC-Dropout ensemble with 3 passes, uncertainty flagging at σ > 0.08 |
| O3 — Temporal stress tracking | ✅ Achieved | Velocity (linear regression), adaptive threshold (μ+1.5σ), volatility (σ of last 5) |
| O4 — Personalised interventions | ✅ Achieved | Eight trigger categories, four intervention layers, sorted by priority |
| O5 — Safety / crisis detection | ✅ Achieved | Crisis circuit-breaker with 988 Lifeline, halts all other processing |
| O6 — Encrypted, authenticated persistence | ✅ Achieved | bcrypt passwords, JWT Bearer auth, Fernet AES-256 history |
| O7 — RL-style feedback loop | ✅ Achieved | User + LLM reward, experience-replay retraining |

The project successfully delivers a complete end-to-end pipeline from raw text input to personalised intervention, with production-ready security controls and a feedback mechanism that allows the model to improve over time.

### 5.2 Future Work and Recommendations

#### 5.2.1 Model Improvements

- **Larger transformer backbone** — Replacing MiniLM with DeBERTa-v3-Base or a domain-specific mental-health BERT (e.g. MentalBERT) would likely improve accuracy, especially on nuanced expressions.
- **Multi-label classification** — Instead of binary stress/no-stress, predict multiple co-occurring emotions (anxiety, sadness, frustration) simultaneously.
- **Calibration** — Implement temperature scaling as part of the standard training pipeline (currently optional) to reduce overconfident predictions.
- **Longer context** — Replace the sliding-window average with a hierarchical model that explicitly models chunk relationships (e.g. a second-stage LSTM or Transformer over chunk embeddings).
- **Active learning** — Use uncertainty scores to surface the most informative samples for human labelling rather than relying on random user feedback.

#### 5.2.2 System Enhancements

- **Multi-language support** — Integrate multilingual models (e.g. XLM-RoBERTa) to serve non-English-speaking users.
- **Voice input** — Add Whisper-based speech-to-text as a preprocessing step, enabling hands-free check-ins.
- **Wearable integration** — Fuse physiological signals (heart-rate variability, skin conductance) with text features for a multimodal stress estimator.
- **Push notifications** — Alert the user when temporal trends suggest deteriorating mental health even without an explicit check-in.
- **Structured journalling** — Guided prompts that elicit richer text input, improving model accuracy and user engagement.

#### 5.2.3 Privacy and Compliance

- **GDPR / HIPAA readiness** — Implement data residency controls, right-to-erasure workflows, and audit logs.
- **On-device inference** — Quantise and export the CNN as an ONNX / TFLite model for browser or mobile inference, keeping sensitive text on-device.
- **Differential privacy** — Add differential privacy noise during federated retraining to protect individual user data.
- **Key rotation** — Implement Fernet multi-key rotation support so that encrypted history can be safely migrated when keys change.

#### 5.2.4 Clinical Validation

- Conduct a prospective study comparing system predictions against validated clinical instruments (PHQ-9, DASS-21) to establish ground-truth validity.
- Partner with clinical psychologists to curate a high-quality gold-standard benchmark and audit the intervention library for clinical accuracy.
- Apply for IRB approval before any deployment involving human subjects in a clinical context.

#### 5.2.5 Deployment and Scalability

- **PostgreSQL migration** — Replace SQLite with PostgreSQL for concurrent multi-user production deployments.
- **Async inference** — Replace the threading lock with a task queue (Celery + Redis) to handle burst load without blocking API workers.
- **Kubernetes** — Containerise into a Helm chart for autoscaling deployment on cloud providers.
- **Model versioning** — Integrate MLflow or Weights & Biases for experiment tracking and A/B model deployment.

---

## 6. References

1. **Vaswani, A., Shazeer, N., Parmar, N., et al.** (2017). *Attention Is All You Need*. Advances in Neural Information Processing Systems (NeurIPS), 30. https://arxiv.org/abs/1706.03762

2. **Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q.** (2017). *On Calibration of Modern Neural Networks*. Proceedings of the 34th International Conference on Machine Learning (ICML). https://arxiv.org/abs/1706.04599

3. **Gal, Y., & Ghahramani, Z.** (2016). *Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning*. Proceedings of the 33rd ICML. https://arxiv.org/abs/1506.02142

4. **Kim, Y.** (2014). *Convolutional Neural Networks for Sentence Classification*. Proceedings of the 2014 Conference on Empirical Methods in NLP (EMNLP). https://arxiv.org/abs/1408.5882

5. **He, P., Liu, X., Gao, J., & Chen, W.** (2021). *DeBERTa: Decoding-enhanced BERT with Disentangled Attention*. ICLR 2021. https://arxiv.org/abs/2006.03654

6. **Wang, W., Wei, F., Dong, L., et al.** (2020). *MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers*. NeurIPS 2020. https://arxiv.org/abs/2002.10957

7. **Turcan, E., & McKeown, K.** (2019). *Dreaddit: A Reddit Dataset for Stress Analysis in Social Media*. Proceedings of the 10th International Workshop on Health Text Mining (LOUHI@EMNLP). https://arxiv.org/abs/1911.00133

8. **Harrigian, K., Aguirre, C., & Dredze, M.** (2020). *Do Models of Mental Health Based on Social Media Data Generalize?* Findings of the Association for Computational Linguistics: EMNLP 2020. https://arxiv.org/abs/2011.09267

9. **World Health Organization.** (2022). *World Mental Health Report: Transforming Mental Health for All*. WHO. https://www.who.int/publications/i/item/9789240049338

10. **Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K.** (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*. NAACL-HLT 2019. https://arxiv.org/abs/1810.04805

11. **Paszke, A., Gross, S., Massa, F., et al.** (2019). *PyTorch: An Imperative Style, High-Performance Deep Learning Library*. NeurIPS 2019. https://arxiv.org/abs/1912.01703

12. **Wolf, T., Debut, L., Sanh, V., et al.** (2020). *Transformers: State-of-the-Art Natural Language Processing*. Proceedings of the 2020 Conference on EMNLP (Systems Demonstrations). https://arxiv.org/abs/1910.03771

13. **FastAPI Documentation.** Sebastián Ramírez, 2023. https://fastapi.tiangolo.com

14. **Streamlit Documentation.** Streamlit Inc., 2024. https://docs.streamlit.io

15. **Python Jose.** (2022). *python-jose: JSON Web Signatures and Tokens*. https://python-jose.readthedocs.io

16. **Fernet (Cryptography Library).** Python Cryptographic Authority, 2024. https://cryptography.io/en/latest/fernet/

17. **bcrypt.** *bcrypt: Good Password Hashing for Your Software and Your Server*. https://pypi.org/project/bcrypt/

18. **SAMHSA.** (2023). *988 Suicide and Crisis Lifeline*. Substance Abuse and Mental Health Services Administration. https://988lifeline.org

19. **International Association for Suicide Prevention.** (2024). *Crisis Centres*. https://www.iasp.info/resources/Crisis_Centres/

20. **Plotly Technologies Inc.** (2015). *Collaborative data science*. Plotly. https://plotly.com

---

*This report is intended for research and educational use. StressDetect — MindView is not a medical device and does not replace professional diagnosis, clinical assessment, or emergency care. If you or someone you know is in crisis, please call or text 988 (US) or contact your local emergency services.*
