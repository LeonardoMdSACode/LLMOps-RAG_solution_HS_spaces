"""
document_ops.py
Utilities for reading PDFs/TXT and chunking text.
"""

from io import BytesIO
from pathlib import Path
from typing import List
from PyPDF2 import PdfReader

async def pdf_to_text_fileobj(fileobj) -> str:
    data = BytesIO(await fileobj.read())
    reader = PdfReader(data)
    pages = []
    for p in reader.pages:
        pages.append(p.extract_text() or "")
    return "\n".join(pages)

def read_text_fileobj(fileobj) -> str:
    fileobj.file.seek(0)
    b = fileobj.file.read()
    if isinstance(b, bytes):
        return b.decode("utf-8", errors="ignore")
    return str(b)

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    if not text:
        return []
    chunks = []
    start = 0
    L = len(text)
    while start < L:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = max(end - overlap, end)
    return chunks
