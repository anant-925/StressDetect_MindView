---
title: StressDetect
emoji: 🧠
colorFrom: indigo
colorTo: purple
sdk: docker
pinned: false
license: mit
---
# StressDetect - MindView

An end-to-end stress detection platform with:

- A secure FastAPI backend for analysis, history, and personalization
- A Streamlit dashboard for check-ins, trends, and interventions
- A deep learning pipeline for training and retraining text classifiers
- A safety-first intervention engine with crisis escalation pathways

This repository is designed as both a usable application and a complete ML systems project.

## What This Project Does

Given short or long free-text input, the system:

1. Cleans and processes text
2. Predicts stress probability with uncertainty-aware inference
3. Assigns a human-readable stress level
4. Applies temporal modeling from user history
5. Returns personalized intervention guidance
6. Stores encrypted session history and feedback

## Key Features

- Model inference with MC-Dropout uncertainty estimation
- Adaptive thresholding and temporal stress profiling
- Rule-based intervention engine with trigger categories
- Crisis keyword circuit breaker with immediate safety messaging
- JWT authentication and bcrypt password hashing
- Encrypted stress history at rest
- RL-style feedback loop from user and LLM reward signals

## Architecture

```text
Streamlit UI (port 7860)
    |
    | REST
    v
FastAPI Backend (port 8000)
    |
    +-- Model Inference (CNN/Transformer)
    +-- Temporal Model
    +-- Intervention Engine
    +-- SQLite Database (users, sessions, feedback)
```

Main runtime entry points:

- `app.py`: boots FastAPI in-process and starts Streamlit (useful for Spaces)
- `api/main.py`: API routes, auth, inference, persistence
- `ui/app.py`: dashboard UI

## Project Layout

```text
StressDetection/
|- api/                  FastAPI app and request handling
|- checkpoints/          Trained model checkpoint (model.pt)
|- data/                 Dataset and evaluation files
|- database/             SQLite access and schema logic
|- intervention/         Recommendation + temporal modeling
|- models/               Neural network architectures
|- scripts/              Data prep and model download utilities
|- security/             JWT, password hashing, encryption
|- tests/                Unit and integration tests
|- training/             Train and retrain scripts
|- ui/                   Streamlit frontend
|- utils/                Shared helpers
|- Dockerfile            Container build
|- supervisord.conf      Process supervision
|- requirements.txt      Runtime dependencies
|- requirements-train.txt Training and testing dependencies
```

## Quick Start (Local)

### 1. Create Environment and Install

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Download Model Checkpoint

```bash
python scripts/download_model.py
```

### 3. Run Backend API

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Run Dashboard

In a second terminal:

```bash
streamlit run ui/app.py
```

Open:

- API docs: http://localhost:8000/docs
- Dashboard: http://localhost:8501

## Makefile Commands

Common workflows:

- `make install` install runtime deps
- `make download` download checkpoint
- `make run` run FastAPI
- `make dashboard` run Streamlit
- `make preprocess` prepare dataset
- `make train` train model
- `make retrain` retrain with feedback
- `make test` run tests
- `make lint` run Ruff lint fixes

Note: the file is named `MakeFile` in this repo.

## API Endpoints

Public:

- `GET /health` service heartbeat
- `GET /model/info` model metadata
- `POST /register` create account and return token
- `POST /login` authenticate and return token

Authenticated (Bearer JWT):

- `POST /analyze` run stress analysis + interventions
- `GET /history` fetch previous sessions
- `POST /feedback` submit user feedback for replay/reward
- `GET /feedback/stats` feedback summary
- `GET /personalization` personalization status

## Training Workflow

### 1. Prepare Data

```bash
python scripts/data_preprocessing.py
```

Expected core training file:

- `data/processed/unified_stress.csv`

### 2. Train Model

```bash
python training/train.py \
  --model cnn \
  --epochs 15 \
  --batch-size 64 \
  --lr 1e-3 \
  --data data/processed/unified_stress.csv \
  --eval-set data/eval/happy_neutral_eval.csv \
  --output checkpoints/model.pt
```

Supported model options include `cnn`, `deberta`, and `minilm`.

### 3. Retrain from Feedback (Optional)

```bash
python training/retrain.py
```

## Testing

Run the full test suite:

```bash
pytest tests/ -v
```

Or with make:

```bash
make test
```

## Security and Safety

Security:

- Passwords hashed with bcrypt
- JWT-based authenticated access
- Encrypted session history in SQLite-backed storage

Safety behavior:

- Crisis phrase detection triggers immediate escalation messaging
- Multi-level intervention outputs aligned to stress severity
- Temporal escalation tracking across sessions

## Docker and Deployment

Containerized deployment is available through `Dockerfile`.

For single-process hosting environments, `app.py` starts FastAPI and Streamlit together.

## Troubleshooting

- API cannot connect from UI:
  - Ensure backend is running on port 8000
  - Set `API_URL` if backend is remote
- Missing checkpoint:
  - Run `python scripts/download_model.py`
- Import or dependency errors:
  - Reinstall with `pip install -r requirements.txt`
  - For training extras use `requirements-train.txt`

## License and Use

This project is intended for research and educational use. It is not a medical device and does not replace professional diagnosis or emergency care.