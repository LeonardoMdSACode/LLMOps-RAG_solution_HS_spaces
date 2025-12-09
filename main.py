# main.py
from __future__ import annotations
import os
from pathlib import Path
from typing import Dict, List
import uuid
import threading
from io import BytesIO

from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from PyPDF2 import PdfReader

# Ensure models are downloaded (idempotent; will skip if already present)
from scripts.download_models import main as download_models_main
download_models_main()

# Initialize free-tier RAGService (loads faiss index if present)
from multi_doc_chat.rag_service import create_rag_service
rag_service = create_rag_service()

from multi_doc_chat.exception.custom_exception import DocumentPortalException

# ----------------------------
# FastAPI initialization
# ----------------------------
app = FastAPI(title="MultiDocChat", version="0.1.0")

# CORS (optional for local dev)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static and templates
BASE_DIR = Path(__file__).resolve().parent
static_dir = BASE_DIR / "static"
templates_dir = BASE_DIR / "templates"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
templates = Jinja2Templates(directory=str(templates_dir))

# ----------------------------
# Simple in-memory chat history and lock
# ----------------------------
STATE_LOCK = threading.Lock()       # protects FAISS index operations & consistent query behavior
SESSIONS: Dict[str, List[dict]] = {}

# ----------------------------
# Adapters (optional)
# ----------------------------
class FastAPIFileAdapter:
    """Adapt FastAPI UploadFile to a simple object with .name and .getbuffer()."""
    def __init__(self, uf: UploadFile):
        self._uf = uf
        self.name = uf.filename or "file"

    def getbuffer(self) -> bytes:
        self._uf.file.seek(0)
        return self._uf.file.read()


# ----------------------------
# Models
# ----------------------------
class UploadResponse(BaseModel):
    session_id: str
    indexed: bool
    message: str | None = None


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    answer: str


# ----------------------------
# Utilities: PDF -> text, simple chunker
# ----------------------------
def pdf_to_text(file_obj) -> str:
    """
    Safely extract text from an UploadFile-like object using PdfReader and BytesIO.
    Returns full text (concatenated pages) or raises an Exception.
    """
    try:
        data = BytesIO(file_obj.read())
        reader = PdfReader(data)
        pages = []
        for p in reader.pages:
            txt = p.extract_text() or ""
            pages.append(txt)
        return "\n".join(pages)
    except Exception as e:
        raise RuntimeError(f"PDF extraction failed: {e}")


def read_txt_file(file_obj) -> str:
    file_obj.file.seek(0)
    raw = file_obj.file.read()
    if isinstance(raw, bytes):
        return raw.decode("utf-8", errors="ignore")
    return str(raw)


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """
    Naive chunker: splits text into overlapping chunks for better retrieval.
    chunk_size and overlap are character counts.
    """
    if not text:
        return []
    chunks = []
    start = 0
    L = len(text)
    while start < L:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = max(end - overlap, end)  # move forward; if overlap >= chunk_size, avoid infinite loop
    return chunks


# ----------------------------
# Routes
# ----------------------------
@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def home(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/upload", response_model=UploadResponse)
async def upload(files: List[UploadFile] = File(...)) -> UploadResponse:
    """
    Accept multiple PDF or TXT files, extract text, chunk them, and ingest into FAISS via rag_service.
    Returns a generated session_id for subsequent /chat calls.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    try:
        all_chunks: List[str] = []

        for f in files:
            fname = (f.filename or "").lower()
            # reset file pointer
            f.file.seek(0)

            if fname.endswith(".pdf"):
                try:
                    text = pdf_to_text(f)
                except Exception as e:
                    raise HTTPException(status_code=400, detail=f"PDF parse error for {f.filename}: {e}")
            else:
                # treat as text by default (txt, md, etc.)
                try:
                    text = read_txt_file(f)
                except Exception as e:
                    raise HTTPException(status_code=400, detail=f"Text read error for {f.filename}: {e}")

            # chunk the text to smaller pieces for better retrieval quality
            chunks = chunk_text(text)
            if not chunks:
                # fallback to full text if chunker failed or text too short
                chunks = [text]
            all_chunks.extend(chunks)

        if not all_chunks:
            raise HTTPException(status_code=400, detail="No readable text extracted from uploaded files")

        # ingest with thread-safety (protect FAISS)
        with STATE_LOCK:
            rag_service.ingest_documents(all_chunks)

        # Generate unique session id and initialize history
        session_id = str(uuid.uuid4())
        SESSIONS[session_id] = []

        return UploadResponse(session_id=session_id, indexed=True, message=f"Ingested {len(all_chunks)} chunks")
    except DocumentPortalException as e:
        raise HTTPException(status_code=500, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {e}")


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    """
    Query the RAG system. Client must supply session_id returned from /upload.
    This endpoint appends user/assistant turns to an in-memory session history.
    """
    session_id = req.session_id or ""
    if not session_id or session_id not in SESSIONS:
        raise HTTPException(status_code=400, detail="Invalid or expired session_id. Re-upload documents.")

    message = (req.message or "").strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    try:
        # Optionally include history into the prompt — currently rag_service.query only needs question.
        with STATE_LOCK:
            # Use lock to prevent index writes while queries are running.
            answer = rag_service.query(message)

        # update history (no heavy locking needed since single threaded here, but we use STATE_LOCK for safety)
        with STATE_LOCK:
            SESSIONS[session_id].append({"role": "user", "content": message})
            SESSIONS[session_id].append({"role": "assistant", "content": answer})

        return ChatResponse(answer=answer)
    except DocumentPortalException as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat failed: {e}")


# Optional: list active sessions (debug)
@app.get("/sessions")
def list_sessions():
    return {"sessions": list(SESSIONS.keys())}


# Uvicorn entrypoint for `python main.py`
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=True)
