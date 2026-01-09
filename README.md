---
title: LLMOps RAG Solution HS Spaces
emoji: 🧠
colorFrom: blue
colorTo: red
sdk: docker
app_file: Dockerfile
pinned: false
license: mit
---

# Multi-Document Chat (LLMOps / RAG)

This repository supports **two parallel execution modes** for the same application:

1. **Local development via Docker Compose (http://localhost:8000/)**
2. **Production-style deployment on Hugging Face Spaces (https://huggingface.co/spaces/LeonardoMdSA/LLMOps-RAG_solution-HS_spaces)**

Both modes run the application inside containers, but they are intentionally **separated** to avoid coupling local tooling with Hugging Face constraints.

---

## Repository Structure

```text
.
├── localhost/
│   ├── docker-compose.yml
│   ├── Dockerfile
│   └── ... (local-only configs, volumes, helpers)
│
├── app.py                # HF Spaces entrypoint
├── Dockerfile            # HF Spaces Docker image
├── requirements.txt
├── README.md
├── LICENSE
└── ... (shared application code)
```

### Key Design Decision

* **`localhost/`** is exclusively for local development and testing
* **Repository root** is optimized for **Hugging Face Spaces**

They serve different runtimes and must not be merged.

---

## Local Development (Docker Compose)

The `localhost/` folder allows you to run the full application locally inside Docker using **docker-compose**.

### Characteristics

* Uses `docker-compose.yml`
* Can spin up multiple services (API, vector DB, tools, etc.)
* Suitable for:

  * Rapid iteration
  * Debugging
  * Volume mounting
  * Local experimentation

### Run Locally

```bash
docker-compose up --build
```

This mode is **not** used by Hugging Face Spaces and is ignored during HF deployment.

---

## Hugging Face Spaces Deployment (Docker)

Hugging Face Spaces builds and runs the application using **only the repository root**.

### Required Files (Root)

* `app.py` → **mandatory entrypoint**
* `Dockerfile` → defines the HF Space image
* `requirements.txt` → Python dependencies

HF Spaces:

* Does **not** use `docker-compose`
* Builds a **single container**
* Exposes the app based on what `app.py` launches

### Deployment Flow

1. Push repository to Hugging Face Spaces
2. HF detects `Dockerfile`
3. Image is built automatically
4. `app.py` is executed inside the container

No local-only files are required or read.

---

## Testing

This repository includes a dedicated `tests/` folder for automated testing.

### Test Structure

```text
tests/
├── __init__.py
├── test_*.py
└── ... (unit and integration tests)
```

Tests are written using **pytest** and are designed to validate:

* Core application logic
* RAG / retrieval components
* Data processing and utility functions

### Run Tests Locally

From the repository root:

```bash
pytest
```

Or explicitly:

```bash
python -m pytest
```

Tests are intended to be executed **locally or in CI**. They are not required for Hugging Face Spaces runtime execution and are not invoked during HF builds unless you explicitly add them to the Dockerfile.

---

## Technology Stack

This project implements a production-style **Retrieval-Augmented Generation (RAG)** system using a fully custom pipeline (no LangChain).

---

### Language & Runtime
- **Python 3.13**
  - Local development
  - Hugging Face Spaces runtime compatibility

---

### API & Web Server
- **FastAPI**
  - REST API for document upload and chat
  - Async request handling
- **Uvicorn**
  - ASGI server
  - Configured for port `7860` (HF Spaces requirement)

---

### Frontend
- **HTML / CSS**
- **Jinja2 Templates**
- **FastAPI StaticFiles**
- No Streamlit
- No JavaScript framework (React/Vue)

---

### Retrieval-Augmented Generation (RAG)

#### Embeddings
- **sentence-transformers**
  - Model: `sentence-transformers/all-MiniLM-L6-v2`
  - Used exclusively for document embeddings
  - Loaded at application startup

#### Vector Store
- **FAISS**
  - In-memory vector index with filesystem persistence
  - Index directory: `faiss_index/`
  - Ephemeral on Hugging Face Spaces (resets when container sleeps)

#### Retrieval Logic
- Fully custom implementation
- Manual chunking, embedding, indexing, and similarity search
- No LangChain or LlamaIndex

---

### Large Language Model (LLM)

- **GGUF-format local LLM**
  - Example: `Qwen2.5-0.5B-Instruct (Q4_0)`
- **llama.cpp-compatible runtime**
- Models downloaded at startup into `models/`
- Used for generation only (not embeddings)

---

### Document Ingestion

- **Supported formats**
  - `.txt`
  - `.pdf`
- **Libraries**
  - `PyPDF2` (functional; noted as deprecated)
- Async ingestion pipeline
- Chunking and embedding performed during upload

---

### Deployment
- **Hugging Face Spaces (Docker-based)**
  - Stateless container
  - No persistent volume
  - FAISS index and uploaded documents reset when container sleeps
- Local Docker-compatible structure

---

### Testing
- **pytest**
- **pytest-asyncio**
- Unit tests
- Integration tests
- External dependencies (FAISS, embedder) mocked where required

---

### Utilities & Tooling
- **requests** – model downloads
- **tqdm** – download progress visualization
- **logging (stdlib)** – centralized logging in `app.py`
- **threading / uuid** – session and state management

---

### Explicitly Not Used
- LangChain
- LlamaIndex
- Streamlit
- Cloud-hosted LLM APIs
- Managed vector databases (Pinecone, Weaviate, etc.)
- Persistent storage

---

## Important Notes

* Keep **HF-specific files at repository root**
* Keep **local-only tooling inside `localhost/`**
* Do not move `app.py` or root `Dockerfile`
* Do not introduce `docker-compose.yml` at root

This separation is intentional and correct.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
