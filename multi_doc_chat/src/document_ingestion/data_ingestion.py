"""
data_ingestion.py
Wrapper to accept uploaded files, extract text and chunk, and call RAGService.ingest_documents
"""

from pathlib import Path
from typing import List
import uuid

from ...rag_service import create_rag_service
from ...utils.document_ops import pdf_to_text_fileobj, read_text_fileobj, chunk_text
from multi_doc_chat.utils.document_ops import pdf_to_text_fileobj

RAG = create_rag_service()

async def ingest_upload_files(upload_files: List) -> str:
    """
    Accepts UploadFile-like objects (FastAPI UploadFile or Gradio file objects)
    Returns session_id.
    """
    all_chunks = []
    for f in upload_files:
        fname = (getattr(f, "filename", None) or getattr(f, "name", "")).lower()
        # PDF
        if fname.endswith(".pdf"):
            text = await pdf_to_text_fileobj(f)
        else:
            text = await read_text_fileobj(f)
        chunks = chunk_text(text)
        if not chunks:
            chunks = [text]
        all_chunks.extend(chunks)

    if not all_chunks:
        raise ValueError("No readable text extracted.")

    # ingest (RAG persists index)
    RAG.ingest_documents(all_chunks)

    # return a session id that maps to the global faiss_index (simple approach)
    # here we just keep single global index; session tracking is done by API layer.
    return "default"
