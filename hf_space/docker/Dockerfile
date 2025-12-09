FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git libgomp1 libopenblas-dev wget curl unzip poppler-utils \
 && rm -rf /var/lib/apt/lists/*

COPY . /app

RUN python -m pip install --upgrade pip
RUN pip install --no-cache-dir -r /app/hf_space/docker/requirements-hf.txt

RUN mkdir -p /app/models /app/faiss_index

COPY hf_space/docker/start.sh /app/start.sh
RUN chmod +x /app/start.sh

EXPOSE 7860
CMD ["/app/start.sh"]
