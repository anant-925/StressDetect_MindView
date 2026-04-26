# StressDetection

Resilience-First AI System for Cross-Platform Stress Detection.

## Project Structure

```
StressDetection/
├── data/
│   ├── raw/              # Place raw dataset files here
│   ├── eval/             # Evaluation datasets (happy_neutral_eval.csv)
│   └── processed/        # Preprocessed unified CSV output
├── models/               # Model architecture definitions
│   └── saved_models/     # Trained model checkpoints
├── notebooks/            # Google Colab training notebook
├── training/             # Training scripts and configs
├── api/                  # Inference server
├── ui/                   # Streamlit UI application
├── database/             # SQLite database manager
├── security/             # Auth, JWT, encryption modules
├── intervention/         # Recommendation engine, temporal model
├── utils/                # Shared utilities
├── tests/                # Unit and integration tests
├── data_preprocessing.py # Multi-dataset merge script
├── requirements.txt      # Python dependencies
├── setup_environment.sh  # Environment setup script (Linux/macOS)
└── run_windows.bat       # One-click launcher for Windows 11
```

---

## Google Colab — Complete Step-by-Step Guide

> **What you'll have at the end:** a trained stress-detection model running as a
> live web application accessible via a public URL — all from inside Colab,
> no local machine required.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/anant-925/StressDetection/blob/main/notebooks/stress_detection_colab.ipynb)

### Before You Start

1. **Switch to a GPU runtime** — free T4 GPU cuts training time from hours to minutes.
   In Colab: **Runtime → Change runtime type → T4 GPU → Save**.

2. **Get a free ngrok account** — needed only if you want to run the live web UI
   from Colab (optional, Steps 8–10).
   Sign up free at <https://dashboard.ngrok.com/signup>, then copy your
   **Authtoken** from <https://dashboard.ngrok.com/get-started/your-authtoken>.

---

### Cell 1 — Mount Google Drive & Clone the Repository

> Everything is saved to Google Drive so your work survives session disconnects.

```python
# ── Cell 1 ──────────────────────────────────────────────────────────────────
from google.colab import drive
drive.mount('/content/drive')

import os

# Persistent directories on Google Drive
DRIVE_BASE = '/content/drive/MyDrive/StressDetection'
for d in ['data/raw', 'data/processed', 'checkpoints', 'logs']:
    os.makedirs(os.path.join(DRIVE_BASE, d), exist_ok=True)

# Clone the repository (skips if already cloned)
if not os.path.isdir('/content/StressDetection'):
    !git clone https://github.com/anant-925/StressDetection.git /content/StressDetection

%cd /content/StressDetection
print("Ready — working directory:", os.getcwd())
```

---

### Cell 2 — Install Dependencies

```python
# ── Cell 2 ──────────────────────────────────────────────────────────────────
!pip install -q -r requirements.txt
print("All dependencies installed.")
```

> This takes about 2 minutes on the first run. Subsequent runs are faster
> because Colab caches packages within the same session.

---

### Cell 3 — Upload Raw Datasets (saved to Google Drive)

Upload your dataset files **once**. They stay on Google Drive forever.

```python
# ── Cell 3 ──────────────────────────────────────────────────────────────────
import shutil, os
from google.colab import files

DRIVE_RAW  = '/content/drive/MyDrive/StressDetection/data/raw'
LOCAL_RAW  = '/content/StressDetection/data/raw'
os.makedirs(LOCAL_RAW, exist_ok=True)

# ── Upload files from your computer ──
print("Select one or more dataset files to upload:")
uploaded = files.upload()            # Opens a file-picker dialog
for filename in uploaded:
    # Save to Google Drive (persists across sessions)
    shutil.move(filename, os.path.join(DRIVE_RAW, filename))
    print(f"  Saved to Drive: {filename}")

# ── Sync Drive → local workspace ──
for f in os.listdir(DRIVE_RAW):
    shutil.copy2(os.path.join(DRIVE_RAW, f), os.path.join(LOCAL_RAW, f))
print(f"\nDatasets available locally in {LOCAL_RAW}:")
print('\n'.join(f'  {f}' for f in os.listdir(LOCAL_RAW)))
```

**Required / optional dataset files** (place in `data/raw/`):

| File | Domain | Notes |
|------|--------|-------|
| `dreaddit-train.csv` (or `.csv.zip`) | Reddit Long | |
| `Reddit_Combi.csv` (or `.xlsx`) | Reddit Long | |
| `Reddit_Title.csv` (or `.xlsx`) | Reddit Short | |
| `Twitter_Full.csv` (or `.xlsx`) | Twitter Short | |
| `Stressed_Tweets.csv` | Twitter Short | implicit label = 1 |
| `Happy_Neutral.csv` | Optional negatives | implicit label = 0 |

> **Already uploaded before?** Skip the `files.upload()` call — the second
> block (Drive → local sync) is enough to restore them.

---

### Cell 4 — Preprocess Data

```python
# ── Cell 4 ──────────────────────────────────────────────────────────────────
# Run only if the processed CSV isn't already on Drive
PROCESSED_DRIVE = '/content/drive/MyDrive/StressDetection/data/processed/unified_stress.csv'
PROCESSED_LOCAL = 'data/processed/unified_stress.csv'

if os.path.isfile(PROCESSED_DRIVE):
    # Restore from Drive (fast — no reprocessing needed)
    os.makedirs('data/processed', exist_ok=True)
    shutil.copy2(PROCESSED_DRIVE, PROCESSED_LOCAL)
    print(f"Restored processed data from Drive: {PROCESSED_LOCAL}")
else:
    # First run — build the unified CSV from raw files
    !python data_preprocessing.py
    shutil.copy2(PROCESSED_LOCAL, PROCESSED_DRIVE)
    print("Processed data saved to Google Drive.")

import pandas as pd
df = pd.read_csv(PROCESSED_LOCAL)
print(f"\nRows: {len(df):,}  |  Label balance:")
print(df['label'].value_counts().to_string())
```

The script produces `data/processed/unified_stress.csv` with columns
`text`, `label`, `domain` plus any numeric features from Dreaddit.

---

### Cell 5 — Train the Model

```python
# ── Cell 5 ──────────────────────────────────────────────────────────────────
# Recommended: CNN is fastest and works well out of the box.
# Change --model to 'deberta' or 'minilm' for transformer-based models
# (requires more VRAM and ~3× longer training time).

!python training/train.py \
    --model cnn \
    --epochs 15 \
    --batch-size 64 \
    --lr 1e-3 \
    --weight-decay 1e-4 \
    --dropout 0.3 \
    --label-smoothing 0.1 \
    --class-weighted \
    --patience 3 \
    --device cuda \
    --data data/processed/unified_stress.csv \
    --eval-set data/eval/happy_neutral_eval.csv \
    --output checkpoints/model.pt
```

**All training flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `cnn` | `cnn`, `deberta`, or `minilm` |
| `--epochs` | `10` | Max training epochs |
| `--batch-size` | `64` | Mini-batch size |
| `--lr` | `1e-3` | Learning rate (CNN); transformers use `2e-5` internally |
| `--weight-decay` | `1e-4` | AdamW weight decay |
| `--dropout` | `0.3` | Dropout rate |
| `--label-smoothing` | `0.0` | Label smoothing for cross-entropy |
| `--class-weighted` | off | Use inverse-frequency class weights (recommended for imbalanced data) |
| `--patience` | `3` | Early-stopping patience (epochs without F1 improvement) |
| `--device` | `cuda` | `cuda` or `cpu` |
| `--data` | `data/processed/unified_stress.csv` | Preprocessed CSV |
| `--output` | `checkpoints/model.pt` | Checkpoint save path |
| `--eval-set` | `data/eval/happy_neutral_eval.csv` | Fixed evaluation set for false-positive monitoring |
| `--max-length` | `256` | Transformer max token length |

Training prints a live progress bar and saves the **best checkpoint by validation F1**
to `checkpoints/model.pt`.  The threshold calibrated during training is embedded
in the checkpoint and loaded automatically by the API.

---

### Cell 6 — Save Checkpoint to Google Drive & Download

```python
# ── Cell 6 ──────────────────────────────────────────────────────────────────
import shutil
from google.colab import files

CKPT_LOCAL = 'checkpoints/model.pt'
CKPT_DRIVE = '/content/drive/MyDrive/StressDetection/checkpoints/model.pt'

# Persist to Google Drive (survives session resets)
shutil.copy2(CKPT_LOCAL, CKPT_DRIVE)
print(f"Checkpoint saved to Google Drive: {CKPT_DRIVE}")

# Download to your computer (for Windows / local deployment)
files.download(CKPT_LOCAL)
print("Download started — check your browser's downloads folder.")
```

> **That's all you need to transfer:** a single `model.pt` file.
> No vocabulary file is required — the tokenizer is hash-based and
> fully deterministic across all platforms.

---

### Cell 7 — Quick Inference Test (no server needed)

Verify the model works before spinning up the full application:

```python
# ── Cell 7 ──────────────────────────────────────────────────────────────────
import sys, hashlib, torch
sys.path.insert(0, '/content/StressDetection')

from models.architecture import OptimizedMultichannelCNN

VOCAB_SIZE   = 10_000
MAX_LEN      = 256
CKPT_LOCAL   = 'checkpoints/model.pt'

# ── Load checkpoint ──
checkpoint = torch.load(CKPT_LOCAL, map_location='cpu', weights_only=True)
model_type  = checkpoint.get('model_type', 'cnn')
threshold   = float(checkpoint.get('decision_threshold', 0.5))

if model_type != 'cnn':
    print(f"Model type is '{model_type}'. Use the transformer path below.")
else:
    dropout    = float(checkpoint.get('dropout', 0.3))
    feature_dim = int(checkpoint.get('feature_dim', 0))
    model = OptimizedMultichannelCNN(
        vocab_size=VOCAB_SIZE, embed_dim=128, num_filters=64,
        kernel_sizes=(2, 3, 5), num_classes=2,
        dropout=dropout, aux_dim=feature_dim,
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # ── Simple hash-based tokenizer (matches API behaviour exactly) ──
    def tokenize(text, vocab_size=VOCAB_SIZE, max_len=MAX_LEN):
        tokens = text.lower().split()[:max_len]
        ids = [
            int(hashlib.md5(t.encode()).hexdigest(), 16) % vocab_size
            for t in tokens
        ]
        if len(ids) < max_len:
            ids += [0] * (max_len - len(ids))
        return ids

    # ── Run inference ──
    test_sentences = [
        "I can't sleep, my mind won't stop racing",
        "Had an amazing day with family, feeling blessed",
        "Overwhelmed with deadlines, barely keeping up",
        "Just finished a great workout, feeling strong",
        "Everything is going wrong and I don't know what to do",
    ]

    print(f"Decision threshold: {threshold:.3f}\n")
    print(f"{'Text':<55} {'Score':>6}  {'Label'}")
    print('-' * 72)
    for text in test_sentences:
        ids    = tokenize(text)
        tensor = torch.tensor([ids], dtype=torch.long)
        with torch.no_grad():
            out = model(tensor)
        prob  = float(torch.softmax(out['logits'], dim=-1)[0, 1])
        label = 'STRESS' if prob >= threshold else 'no stress'
        print(f"{text[:54]:<55} {prob:>6.3f}  {label}")
```

Expected output — the model should clearly distinguish stressed from calm text.

---

### Cells 8–10 — Run the Full Application from Colab (Optional)

> These cells start the FastAPI backend and Streamlit UI inside Colab and
> expose them via **ngrok** public URLs so you can open the dashboard in
> any browser.  Requires a **free ngrok account** (see *Before You Start*).

#### Cell 8 — Authenticate ngrok

```python
# ── Cell 8 ──────────────────────────────────────────────────────────────────
!pip install -q pyngrok

from pyngrok import ngrok

# Paste your authtoken from https://dashboard.ngrok.com/get-started/your-authtoken
NGROK_TOKEN = "PASTE_YOUR_NGROK_AUTHTOKEN_HERE"
ngrok.set_auth_token(NGROK_TOKEN)
print("ngrok authenticated.")
```

#### Cell 9 — Start the FastAPI Backend

```python
# ── Cell 9 ──────────────────────────────────────────────────────────────────
import subprocess, time, os

os.makedirs('checkpoints', exist_ok=True)

# Set security keys (auto-generated dev keys are fine for personal testing)
os.environ.setdefault('JWT_SECRET_KEY', 'colab-dev-secret-key-change-for-production')

# Generate and set a Fernet key if not already set
if 'FERNET_KEY' not in os.environ:
    from cryptography.fernet import Fernet
    os.environ['FERNET_KEY'] = Fernet.generate_key().decode()

# Start FastAPI in the background
api_proc = subprocess.Popen(
    ['uvicorn', 'api.main:app', '--host', '0.0.0.0', '--port', '8000'],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT
)
time.sleep(4)          # Wait for the server to finish loading

# Create a public ngrok tunnel to port 8000
api_tunnel = ngrok.connect(8000)
API_URL    = str(api_tunnel.public_url)
print(f"FastAPI is live at:  {API_URL}")
print(f"Interactive API docs: {API_URL}/docs")
```

> Open **`{API_URL}/docs`** in your browser to explore all endpoints
> interactively (register, login, analyze) with the built-in Swagger UI.

#### Cell 10 — Start the Streamlit UI

```python
# ── Cell 10 ─────────────────────────────────────────────────────────────────
import subprocess, time, os

# Tell the UI where the API lives
os.environ['API_URL'] = API_URL

# Start Streamlit in the background
ui_proc = subprocess.Popen(
    ['streamlit', 'run', 'ui/app.py',
     '--server.port', '8501',
     '--server.headless', 'true'],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    env={**os.environ}
)
time.sleep(6)          # Wait for Streamlit to compile

# Create a public ngrok tunnel to port 8501
ui_tunnel = ngrok.connect(8501)
UI_URL    = str(ui_tunnel.public_url)
print(f"Streamlit UI is live at: {UI_URL}")
print("Open the link above in your browser to use the dashboard.")
```

> Click the **Streamlit URL**, register an account, and start analysing text.

#### Stopping the Services

```python
# Run this cell when you are done
api_proc.terminate()
ui_proc.terminate()
ngrok.kill()
print("All services stopped.")
```

---

### Reconnecting After a Disconnect

If your Colab session timed out or you closed the browser tab:

| Cell | Action | Skip if… |
|------|--------|---------|
| **Cell 1** | Mount Drive + clone repo | — always re-run |
| **Cell 2** | Install dependencies | — always re-run |
| **Cell 3** | Upload datasets | Files already in Drive — run only the sync block |
| **Cell 4** | Preprocess data | Processed CSV is on Drive — the cell auto-restores it |
| **Cell 5** | Train | Checkpoint already on Drive — skip, go to Cell 6 |
| **Cell 6** | Save checkpoint | Restore from Drive: `shutil.copy2(CKPT_DRIVE, CKPT_LOCAL)` |
| **Cells 7–10** | Test / run app | — re-run as needed |

**Your datasets, processed CSV, and trained checkpoint are all safe on Google Drive.**

---

### Quick-Reference Table

| Step | Where | Command / Action |
|------|-------|-----------------|
| 1 | Colab | Switch runtime to T4 GPU |
| 2 | Colab | Cell 1 — mount Drive, clone repo |
| 3 | Colab | Cell 2 — install dependencies |
| 4 | Colab | Cell 3 — upload datasets to Drive |
| 5 | Colab | Cell 4 — preprocess → `unified_stress.csv` |
| 6 | Colab | Cell 5 — train the model |
| 7 | Colab | Cell 6 — save `model.pt` to Drive; download to PC |
| 8 | Colab | Cell 7 — quick inference test (no server needed) |
| 9 | Colab | Cells 8–10 — live app via ngrok *(optional)* |
| 10 | Windows | Download `model.pt`; run `run_windows.bat` |

---

## Part B — Running the Application on Windows 11

#### Quick Start (One-Click Launcher)

After cloning the repo and placing `model.pt`, simply double-click:

```
run_windows.bat
```

This automatically creates the virtual environment, installs dependencies,
starts both the FastAPI backend and Streamlit UI in separate windows, and
opens your browser to the dashboard.

#### Manual Setup

##### Step 1 — Clone the Repository

```cmd
git clone https://github.com/anant-925/StressDetection.git
cd StressDetection
```

##### Step 2 — Set Up the Python Environment

```cmd
python -m venv venv
venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

> Use **`venv` + `pip`** — do not use Conda.
> Requires **Python 3.10 or newer**.

##### Step 3 — Place the Trained Checkpoint

Download `model.pt` from Google Drive (`MyDrive/StressDetection/checkpoints/model.pt`)
or from your Colab download:

```cmd
mkdir checkpoints
copy %USERPROFILE%\Downloads\model.pt checkpoints\model.pt
```

The API server looks for `checkpoints/model.pt` by default.
Override via the `STRESS_MODEL_CHECKPOINT` environment variable if needed.

##### Step 4 — Set Security Keys *(optional for local use)*

```cmd
set JWT_SECRET_KEY=your-random-secret-key-here
set FERNET_KEY=your-base64-fernet-key-here
```

Generate a Fernet key (requires Step 2 — `pip install -r requirements.txt` — to be done first):

```cmd
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

If omitted, auto-generated development keys are used (fine for local testing,
not recommended for production).

##### Step 5 — Start the FastAPI Backend *(Terminal 1)*

```cmd
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

The model checkpoint is loaded on the first `/analyze` request.
API docs are available at <http://localhost:8000/docs>.

##### Step 6 — Start the Streamlit UI *(Terminal 2)*

```cmd
venv\Scripts\activate
streamlit run ui/app.py
```

Opens the dashboard at <http://localhost:8501>.
The UI connects to the FastAPI backend at `http://localhost:8000` by default
(override with the `API_URL` environment variable).

##### Step 7 — Use the Application

1. Open <http://localhost:8501> in your browser.
2. **Register** a new account (username ≥ 3 chars, password ≥ 8 chars).
3. Type text describing how you're feeling and click **Analyse**.
4. Results include:
   - **Stress score** (0–100%)
   - **Stress velocity** (trend direction over your history)
   - **Attention heatmap** (which words influenced the prediction)
   - **Recommended interventions** (breathing, grounding, cognitive reframes)
   - **Crisis detection** — if crisis language is detected the app shows
     the 988 Suicide & Crisis Lifeline and stops further processing.
5. Your stress history chart grows with each analysis during the session.

---

## Quick Start (local Linux/macOS)

```bash
# 1. Set up the environment
bash setup_environment.sh

# 2. Place your dataset files in data/raw/
# 3. Run preprocessing
python data_preprocessing.py

# 4. Train
python training/train.py

# 5. Start API + UI
uvicorn api.main:app --host 0.0.0.0 --port 8000 &
streamlit run ui/app.py
```

## Datasets

Place these files in `data/raw/`:

| File | Domain |
|------|--------|
| `dreaddit-train.csv` (or `.csv.zip`) | Reddit Long |
| `Reddit_Combi.csv` (or `.xlsx`) | Reddit Long |
| `Reddit_Title.csv` (or `.xlsx`) | Reddit Short |
| `Twitter_Full.csv` (or `.xlsx`) | Twitter Short |
| `Stressed_Tweets.csv` | Twitter Short (implicit label=1) |
| `Happy_Neutral.csv` | Optional negatives (implicit label=0) |

### Happy/Neutral Evaluation Set

The repository includes `data/eval/happy_neutral_eval.csv`, a small fixed
set of happy/neutral sentences used to monitor false positives during training.
You can extend it with your own examples (keep the `text,label` columns).

## Testing

```bash
python -m pytest tests/ -v
```
