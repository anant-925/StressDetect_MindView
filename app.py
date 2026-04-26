#!/usr/bin/env python3
"""
app.py  (HuggingFace Spaces entry point)
=========================================
Boots the FastAPI backend in a background thread, then hands control
to the Streamlit UI — both run inside a single Spaces process.

HuggingFace Spaces runs one process and exposes port 7860.
Streamlit listens on 7860; FastAPI listens on 8000 (internal only).
The Streamlit UI talks to FastAPI at http://localhost:8000.

Usage (locally)
---------------
    streamlit run app.py --server.port 7860

Usage (Spaces) — Spaces calls `streamlit run app.py` automatically
when SDK: streamlit is set in README.md.
"""

from __future__ import annotations

import os
import sys
import threading
import time

# ── Point the UI at the in-process FastAPI ──
os.environ.setdefault("API_URL", "http://localhost:8000")

# ── Download model checkpoint if not present ──
_CHECKPOINT = os.path.join(os.path.dirname(__file__), "checkpoints", "model.pt")
if not os.path.isfile(_CHECKPOINT):
    print("[startup] Checkpoint not found — downloading from HuggingFace Hub …")
    import subprocess
    subprocess.run(
        [sys.executable, "scripts/download_model.py"],
        check=False,
    )


def _run_fastapi() -> None:
    """Start uvicorn in a daemon thread."""
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        log_level="warning",
    )


# Start FastAPI in the background
_api_thread = threading.Thread(target=_run_fastapi, daemon=True)
_api_thread.start()

# Give FastAPI a moment to bind before Streamlit starts hitting /health
time.sleep(3)

# ── Hand off to the Streamlit UI ──
# Streamlit's entry point is ui/app.py — we import and call main() directly
# so Spaces only needs to run `streamlit run app.py`.
sys.path.insert(0, os.path.dirname(__file__))
from ui.app import main  # noqa: E402
main()