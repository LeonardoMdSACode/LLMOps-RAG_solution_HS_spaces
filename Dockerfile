FROM python:3.10-slim

# Install dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential cmake wget git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy everything
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

# Default command for local testing
CMD ["python", "main.py"]
