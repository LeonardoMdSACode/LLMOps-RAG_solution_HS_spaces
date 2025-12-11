import pytest
from multi_doc_chat.rag_service import create_rag_service
from multi_doc_chat.src.document_ingestion import data_ingestion as di
from io import BytesIO
import asyncio
from unittest.mock import MagicMock
import numpy as np

class DummyUploadFile:
    def __init__(self, name, content):
        self.filename = name
        self.file = BytesIO(content.encode("utf-8"))
        self._content = content.encode("utf-8")
    def read(self):
        return self._content

# Patch async file reading
import multi_doc_chat.utils.document_ops as doc_ops
async def async_read_text_fileobj(fileobj):
    return await asyncio.to_thread(doc_ops.read_text_fileobj, fileobj)
di.read_text_fileobj = async_read_text_fileobj

class FakeEmbedder:
    def encode(self, texts, show_progress_bar=False):
        return np.zeros((len(texts), 768), dtype="float32")

@pytest.mark.asyncio
async def test_ingest_txt_file():
    rag_service = create_rag_service(faiss_dir="tests/faiss_test_index")

    # patch embedder
    rag_service.loader.embedder = FakeEmbedder()

    # patch FAISS completely
    rag_service.index = MagicMock()
    rag_service.index.add = MagicMock()
    
    txt_file = DummyUploadFile("example.txt", "Hello world! This is a test.")

    session_id = await di.ingest_upload_files([txt_file], rag_service)

    assert session_id == "default"
    assert len(rag_service.documents) > 0
    assert any("Hello world" in chunk for chunk in rag_service.documents)
