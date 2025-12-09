#!/usr/bin/env bash
set -e

# Download models if they don't exist (idempotent)
python /app/scripts/download_models.py

# Ensure permissions
mkdir -p /app/models /app/faiss_index
chown -R root:root /app/models /app/faiss_index || true

# Launch app (Gradio)
python /app/app.py --server-name 0.0.0.0 --server-port ${PORT:-7860}
