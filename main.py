# main.py  (updated)
from __future__ import annotations
import os
from pathlib import Path
from typing import Dict, List
import uuid
import threading

from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from multi_doc_chat.src.document_ingestion.data_ingestion import ingest_upload_files
from multi_doc_chat.rag_service import create_rag_service

# Ensure models downloaded
from scripts.download_models import main as download_models_main
download_models_main()


rag_service = create_rag_service()

app = FastAPI(title="MultiDocChat", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
static_dir = BASE_DIR / "static"
templates_dir = BASE_DIR / "templates"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
templates = Jinja2Templates(directory=str(templates_dir))

STATE_LOCK = threading.Lock()
SESSIONS: Dict[str, List[dict]] = {}

class UploadResponse(BaseModel):
    session_id: str
    indexed: bool
    message: str | None = None

class ChatRequest(BaseModel):
    session_id: str
    message: str

class ChatResponse(BaseModel):
    answer: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload", response_model=UploadResponse)
async def upload(files: List[UploadFile] = File(...)) -> UploadResponse:
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    try:
        print("Files received:", [f.filename for f in files])

        # create a session ID FIRST
        session_id = str(uuid.uuid4())
        SESSIONS[session_id] = []

        # ingest the files (this updates the global index)
        with STATE_LOCK:
            await ingest_upload_files(files)

        return UploadResponse(
            session_id=session_id,
            indexed=True,
            message=f"Ingested under session {session_id}"
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Upload failed: {e}")


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    print("CHAT REQUEST:", req)
    session_id = req.session_id or ""
    if not session_id or session_id not in SESSIONS:
        raise HTTPException(status_code=400, detail="Invalid or expired session_id. Re-upload documents.")
    message = (req.message or "").strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    try:
        # use RAGService to answer (lock around index access)
        with STATE_LOCK:
            answer = rag_service.query(message)
        SESSIONS[session_id].append({"role":"user","content":message})
        SESSIONS[session_id].append({"role":"assistant","content":answer})
        return ChatResponse(answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat failed: {e}")

@app.get("/sessions")
def list_sessions():
    return {"sessions": list(SESSIONS.keys())}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=True)
