#!/usr/bin/env python3
from __future__ import annotations

import os
import sys

REPO_ID    = "Ace-119/stress-detection-cnn"
FILENAME   = "model.pt"
LOCAL_DIR  = os.path.join(os.path.dirname(os.path.dirname(__file__)), "checkpoints")
LOCAL_PATH = os.path.join(LOCAL_DIR, FILENAME)


def download() -> None:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("ERROR: huggingface_hub is not installed.")
        print("Run:  pip install huggingface-hub")
        sys.exit(1)

    os.makedirs(LOCAL_DIR, exist_ok=True)

    if os.path.isfile(LOCAL_PATH):
        size_mb = os.path.getsize(LOCAL_PATH) / 1_048_576
        print(f"Checkpoint already present: {LOCAL_PATH} ({size_mb:.1f} MB)")
        return

    token = os.environ.get("HF_TOKEN") or None
    print(f"Downloading {REPO_ID}/{FILENAME} → {LOCAL_PATH} ...")

    try:
        path = hf_hub_download(
            repo_id=REPO_ID,
            filename=FILENAME,
            local_dir=LOCAL_DIR,
            token=token,
        )
        size_mb = os.path.getsize(path) / 1_048_576
        print(f"Done. Saved to {path} ({size_mb:.1f} MB)")
    except Exception as exc:
        print(f"ERROR: Could not download model: {exc}")
        print(
            "If the repo is private, set HF_TOKEN:\n"
            "  export HF_TOKEN=hf_your_token_here"
        )
        sys.exit(1)


if __name__ == "__main__":
    download()