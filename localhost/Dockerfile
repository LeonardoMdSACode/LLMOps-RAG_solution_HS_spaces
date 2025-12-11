FROM python:3.10-slim

WORKDIR /app

COPY localhost/requirements.txt ./requirements.txt
COPY localhost/main.py ./main.py
COPY multi_doc_chat ./multi_doc_chat
COPY templates ./templates
COPY static ./static
COPY scripts ./scripts
COPY configs ./configs
COPY models ./models
COPY faiss_index ./faiss_index

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential cmake wget git && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -r requirements.txt

# Ensure PYTHONPATH includes /app
ENV PYTHONPATH=/app

EXPOSE 8000
CMD ["python", "main.py"]
