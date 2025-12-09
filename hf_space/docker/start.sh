#!/usr/bin/env bash
set -e

# ensure models exist
python /app/scripts/download_models.py

# launch the gradio app
python /app/hf_space/app.py
