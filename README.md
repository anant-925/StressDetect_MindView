---
title: StressDetect
emoji: 🧠
colorFrom: indigo
colorTo: purple
sdk: docker
pinned: false
license: mit
---

# 🧠 StressDetect

A full-stack mental health application for real-time stress detection with
personalized interventions — built as an end-to-end ML systems project.

## Architecture

```
┌─────────────────────────────────────────────────┐
│                  Streamlit UI                   │  ← port 7860 (public)
│  Dashboard · History & Analytics · Settings     │
└──────────────────┬──────────────────────────────┘
                   │ REST (localhost)
┌──────────────────▼──────────────────────────────┐
│               FastAPI Backend                   │  ← port 8000 (internal)
│  /analyze · /history · /feedback · /personalize │
└──────────────────┬──────────────────────────────┘
                   │
       ┌───────────┼───────────┐
       ▼           ▼           ▼
   CNN Model   SQLite DB   Intervention
  (MC-Dropout) (encrypted)   Engine
```

## Model

- **Multichannel 1D CNN** with multi-head self-attention (4 heads)
- **MC-Dropout ensemble** (3 passes) for uncertainty estimation
- **FPR-constrained threshold calibration** (max FPR 20%)
- **Focal loss** + cosine LR warmup + early stopping
- Checkpoint: [`Ace-119/stress-detection-cnn`](https://huggingface.co/Ace-119/stress-detection-cnn)

## Safety

- **4-layer intervention engine** with 988 crisis circuit breaker
- Crisis keywords (suicide/self-harm) → immediate 988 lifeline, pipeline halts
- 8 trigger categories: sleep, work, exam, money, relationship, health, grief, loneliness
- Escalation tracker: 3+ consecutive high-stress sessions → professional referral

## Features

| Feature | Detail |
|---|---|
| Stress scoring | CNN probability + MC-Dropout uncertainty |
| Temporal profiling | Adaptive threshold, velocity, volatility |
| Interventions | Progressive step-by-step guided flow |
| RL feedback loop | User + LLM-as-judge reward signal |
| Personalization | Per-user score bias from feedback history |
| Analytics | Timeline, calendar heatmap, polar chart, trigger frequency |
| Security | JWT auth, bcrypt passwords, AES-256 history encryption |

## Quick Start (local)

```bash
git clone https://github.com/Ace-119/StressDetection
cd StressDetection
pip install -r requirements-train.txt   # full deps including training
python scripts/download_model.py        # pulls model.pt from HF Hub
make dev                                # FastAPI :8000 + Streamlit :8501
```

## Crisis Resources

This app always surfaces crisis resources when needed.

**988 Suicide & Crisis Lifeline** — Call or text **988** (US)  
**Crisis Text Line** — Text HOME to **741741**  
**SAMHSA Helpline** — 1-800-662-4357 (free, confidential, 24/7)
