# Use a Debian slim base for size control
FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# Install system deps required for llama-cpp-python, faiss, pdf parsing, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git libgomp1 libopenblas-dev wget curl unzip \
    poppler-utils \
 && rm -rf /var/lib/apt/lists/*

# Copy project files (we'll download models at runtime to avoid huge images)
COPY ../app.py /app/app.py
COPY ../requirements-hf.txt /app/requirements-hf.txt
COPY ../scripts /app/scripts
COPY ../multi_doc_chat /app/multi_doc_chat
COPY ../configs /app/configs

# Install Python deps
RUN python -m pip install --upgrade pip
RUN pip install --no-cache-dir -r /app/requirements-hf.txt

# Create models and faiss directories (writeable)
RUN mkdir -p /app/models /app/faiss_index

# Copy startup script and grant execute
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

EXPOSE 7860
CMD [ "/app/start.sh" ]
