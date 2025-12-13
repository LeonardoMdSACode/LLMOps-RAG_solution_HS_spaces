import pytest
import asyncio
import numpy as np
from io import BytesIO
from unittest.mock import MagicMock, AsyncMock
import faiss

from multi_doc_chat.rag_service import create_rag_service
from multi_doc_chat.src.document_ingestion import data_ingestion as di


# -------------------------
# Proper async UploadFile mock
# -------------------------
class DummyUploadFile:
    def __init__(self, filename: str, content: str):
        self.filename = filename
        self._content = content.encode("utf-8")

    async def read(self) -> bytes:
        return self._content


# -------------------------
# Fake embedder (fast + deterministic)
# -------------------------
class FakeEmbedder:
    def encode(self, texts, show_progress_bar=False):
        return np.zeros((len(texts), 384), dtype="float32")


@pytest.mark.asyncio
async def test_ingest_txt_file(monkeypatch, tmp_path):
    """
    Unit test:
    - no real FAISS
    - no real disk
    - no real models
    - no blocking calls
    """

    # --- Create service with temp dir ---
    rag_service = create_rag_service(faiss_dir=str(tmp_path))

    # --- Patch embedder ---
    rag_service.loader.embedder = FakeEmbedder()

    # --- Patch FAISS completely ---
    rag_service.index = MagicMock()
    rag_service.index.add = MagicMock()

    # --- Patch document ops to avoid real file parsing ---
    async def fake_read_text_fileobj(fileobj):
        return "Hello world! This is a test."

    monkeypatch.setattr(
        di,
        "read_text_fileobj",
        fake_read_text_fileobj,
    )

    # --- Input ---
    upload = DummyUploadFile("example.txt", "Hello world! This is a test.")
    monkeypatch.setattr(faiss, "write_index", lambda *args, **kwargs: None)

    # --- Act ---
    session_id = await di.ingest_upload_files([upload], rag_service)

    # --- Assert ---
    assert session_id == "default"
    assert len(rag_service.documents) > 0
    assert any("Hello world" in doc for doc in rag_service.documents)

    # --- Ensure FAISS was called ---
    rag_service.index.add.assert_called_once()

@pytest.fixture(autouse=True)
def no_threads(monkeypatch):
    async def fake_to_thread(func, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", fake_to_thread)
