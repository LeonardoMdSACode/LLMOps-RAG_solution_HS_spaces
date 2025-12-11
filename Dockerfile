FROM python:3.10-slim

# System deps
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential cmake wget git ffmpeg poppler-utils && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only what HF Space needs (exclude localhost folder)
COPY multi_doc_chat ./multi_doc_chat
COPY templates ./templates
COPY static ./static
COPY scripts ./scripts
COPY configs ./configs
COPY requirements.txt ./requirements.txt
COPY app.py ./app.py

# Install python deps
RUN pip install --no-cache-dir -r requirements.txt

# Create folders for models/index
RUN mkdir -p /app/models /app/faiss_index

# HF Spaces port
ENV PORT=7860
EXPOSE 7860

# Entrypoint
CMD ["python", "app.py"]
