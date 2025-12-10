# hf_space/docker/Dockerfile
FROM python:3.10-slim

# System deps
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential cmake wget git ffmpeg poppler-utils && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only what Space needs (copy entire repo is fine, but keep models excluded by .gitignore)
COPY . /app

# Install python deps (ensure requirements.txt exists and is minimal)
RUN pip install --no-cache-dir -r requirements.txt

# Create models directory (container will download models on startup)
RUN mkdir -p /app/models && mkdir -p /app/faiss_index

# HF Spaces expects the app to listen on PORT 7860 (Gradio default). We'll run Uvicorn on 7860.
ENV PORT=7860
EXPOSE 7860

# Entrypoint - start the app (this will run download_models at startup if you kept that call)
CMD ["python", "hf_space/app.py"]
