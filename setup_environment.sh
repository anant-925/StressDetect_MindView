#!/usr/bin/env bash
# setup_environment.sh
# Sets up the Python virtual environment and project directory structure.
# Usage: bash setup_environment.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Resilience-First AI: Environment Setup ==="

# 1. Create Python virtual environment (NO CONDA)
echo "[1/4] Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# 2. Upgrade pip
echo "[2/4] Upgrading pip..."
pip install --upgrade pip

# 3. Install requirements
echo "[3/4] Installing requirements..."
pip install -r requirements.txt

# 4. Create directory structure
echo "[4/4] Creating project directory structure..."
mkdir -p data/raw data/processed
mkdir -p models/saved_models
mkdir -p training
mkdir -p api
mkdir -p ui
mkdir -p utils
mkdir -p tests
mkdir -p security
mkdir -p intervention

echo ""
echo "=== Setup Complete ==="
echo "Directory structure:"
echo "  data/raw/          - Place your raw datasets here"
echo "  data/processed/    - Preprocessed unified CSV output"
echo "  models/            - Model architecture definitions"
echo "  models/saved_models/ - Trained model checkpoints"
echo "  training/          - Training scripts and configs"
echo "  api/               - FastAPI inference server"
echo "  ui/                - Streamlit UI application"
echo "  utils/             - Shared utilities"
echo "  security/          - Auth, JWT, encryption modules"
echo "  intervention/      - Recommendation engine, temporal model"
echo "  tests/             - Unit and integration tests"
echo ""
echo "Next step: Place your 5 dataset files in data/raw/ and run:"
echo "  python data_preprocessing.py"
