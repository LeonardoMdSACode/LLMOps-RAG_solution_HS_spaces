from typing import List
from ...utils.document_ops import pdf_to_text_fileobj, read_text_fileobj, chunk_text

async def ingest_upload_files(upload_files: List, rag_service) -> str:
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

    # ingest (update the shared RAG instance)
    rag_service.ingest_documents(all_chunks)

    return "default"
