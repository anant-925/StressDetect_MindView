# ── Base image ──────────────────────────────────────────────────────────────
# Python 3.11 slim keeps the image lean while matching the dev environment.
FROM python:3.11-slim

# ── System dependencies ──────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ────────────────────────────────────────────────────────
WORKDIR /app

# ── Python dependencies ──────────────────────────────────────────────────────
# Copy requirements first so Docker caches this layer unless deps change.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# ── Application code ─────────────────────────────────────────────────────────
COPY . .

# ── Download model checkpoint from HuggingFace Hub ───────────────────────────
# Runs at build time so the container starts instantly (no cold-download delay).
# If HF_TOKEN is set as a build secret the download works for private repos too.
RUN python scripts/download_model.py

# ── Environment ──────────────────────────────────────────────────────────────
# Tell the Streamlit UI where to find the FastAPI backend (same container).
ENV API_URL=http://localhost:8000
# Streamlit must listen on 0.0.0.0:7860 for Spaces to route traffic correctly.
ENV STREAMLIT_SERVER_PORT=7860
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
# Disable Streamlit's browser-open behaviour (no browser inside a container).
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
# Keep Python output unbuffered so logs appear in real time.
ENV PYTHONUNBUFFERED=1
# Use /app/data as the SQLite database location (writable inside the container).
ENV STRESS_DB_PATH=/app/stress_detection.db

# ── Expose Streamlit port ────────────────────────────────────────────────────
# HuggingFace Spaces routes all external traffic to port 7860.
EXPOSE 7860

# ── Process manager ──────────────────────────────────────────────────────────
# supervisord runs FastAPI (port 8000) and Streamlit (port 7860) as two
# supervised processes inside one container — the standard pattern for
# HuggingFace Spaces Docker deployments that need a backend + frontend.
RUN pip install --no-cache-dir supervisor

COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

CMD ["/usr/local/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
