# main.py

from __future__ import annotations
import os
from pathlib import Path
from typing import Dict, List
import uuid
import threading
import traceback
import logging

from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from multi_doc_chat.src.document_ingestion.data_ingestion import ingest_upload_files
from multi_doc_chat.rag_service import create_rag_service

# --- Configure logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# --- Ensure models downloaded ---
from scripts.download_models import main as download_models_main
try:
    download_models_main()
except Exception as e:
    logger.warning("download_models_main() failed: %s\n%s", e, traceback.format_exc())

# --- ROOT paths for FAISS and models ---
ROOT_DIR = Path(__file__).resolve().parent
FAISS_DIR = ROOT_DIR / "faiss_index"
MODELS_DIR = ROOT_DIR / "models"

# --- Create RAGService ---
rag_service = create_rag_service(faiss_dir=str(FAISS_DIR))

# --- Startup logging for monitoring ---
logger.info("RAGService initialized at %s", FAISS_DIR)
logger.info("Number of documents in FAISS index: %d", len(rag_service.documents))
if rag_service.loader.embedder:
    logger.info("Embedding model loaded successfully")
else:
    logger.warning("Embedding model not loaded")

# --- FastAPI app ---
app = FastAPI(title="RAG Solution", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

static_dir = ROOT_DIR / "static"
templates_dir = ROOT_DIR / "templates"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
templates = Jinja2Templates(directory=str(templates_dir))

STATE_LOCK = threading.Lock()
SESSIONS: Dict[str, List[dict]] = {}

# --- Pydantic models ---
class UploadResponse(BaseModel):
    session_id: str
    indexed: bool
    message: str | None = None

class ChatRequest(BaseModel):
    session_id: str
    message: str

class ChatResponse(BaseModel):
    answer: str

# --- Health endpoint ---
@app.get("/health")
def health():
    return {
        "status": "ok",
        "faiss_docs": len(rag_service.documents),
        "model_loaded": rag_service.loader.embedder is not None
    }

# --- Home endpoint ---
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# --- Upload endpoint ---
@app.post("/upload", response_model=UploadResponse)
async def upload(files: List[UploadFile] = File(...)) -> UploadResponse:
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    session_id = str(uuid.uuid4())
    SESSIONS[session_id] = []

    try:
        logger.info("Files received: %s", [f.filename for f in files])

        # ingest the files using the shared rag_service
        with STATE_LOCK:
            await ingest_upload_files(files, rag_service)

        logger.info("Files ingested successfully for session %s", session_id)
        return UploadResponse(
            session_id=session_id,
            indexed=True,
            message=f"Ingested under session {session_id}"
        )

    except Exception as e:
        tb = traceback.format_exc()
        logger.error("Upload failed: %s\n%s", e, tb)
        raise HTTPException(status_code=500, detail=f"Upload failed: {e}")

# --- Chat endpoint ---
@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    session_id = req.session_id or ""
    if not session_id or session_id not in SESSIONS:
        raise HTTPException(status_code=400, detail="Invalid or expired session_id. Re-upload documents.")
    message = (req.message or "").strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    try:
        logger.info("Chat request received: session=%s, message=%s", session_id, message)
        with STATE_LOCK:
            answer = rag_service.query(message)
        SESSIONS[session_id].append({"role": "user", "content": message})
        SESSIONS[session_id].append({"role": "assistant", "content": answer})
        logger.info("Chat response sent for session %s", session_id)
        return ChatResponse(answer=answer)

    except Exception as e:
        tb = traceback.format_exc()
        logger.error("Chat failed: %s\n%s", e, tb)
        raise HTTPException(status_code=500, detail=f"Chat failed: {e}")

# --- List sessions ---
@app.get("/sessions")
def list_sessions():
    return {"sessions": list(SESSIONS.keys())}

# --- Run server ---
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
