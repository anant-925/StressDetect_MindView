# ── Base image ──────────────────────────────────────────────────────────────
FROM python:3.11-slim

# ── Create non-root user (HF best practice) ─────────────────────────────────
RUN useradd -m -u 1000 user

# ── System dependencies ──────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ────────────────────────────────────────────────────────
WORKDIR /app

# ── Python dependencies ──────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir supervisor

# ── Application code ─────────────────────────────────────────────────────────
COPY . .

# ── Download model checkpoint ────────────────────────────────────────────────
RUN python scripts/download_model.py

# ── Permissions (IMPORTANT for HF) ───────────────────────────────────────────
RUN chown -R user:user /app

# Switch to non-root AFTER setup
USER user

# ── Environment ──────────────────────────────────────────────────────────────
ENV API_URL=http://localhost:8000
ENV STREAMLIT_SERVER_PORT=7860
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
ENV PYTHONUNBUFFERED=1
# ENV STRESS_DB_PATH=/app/stress_detection.db
ENV STRESS_DB_PATH=/data/stress_detection.db
# ── Expose port ──────────────────────────────────────────────────────────────
EXPOSE 7860

# ── Supervisor config ────────────────────────────────────────────────────────
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# ── Start services ───────────────────────────────────────────────────────────
CMD ["/usr/local/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]