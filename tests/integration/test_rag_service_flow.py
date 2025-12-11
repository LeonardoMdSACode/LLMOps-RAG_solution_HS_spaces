import pytest
from multi_doc_chat.rag_service import create_rag_service
from unittest.mock import MagicMock
import numpy as np

class FakeEmbedder:
    def encode(self, texts, show_progress_bar=False):
        return np.zeros((len(texts), 768), dtype="float32")

@pytest.mark.asyncio
async def test_rag_service_basic_flow():
    rag = create_rag_service(faiss_dir="tests/faiss_test_index")

    # patch embedder
    rag.loader.embedder = FakeEmbedder()

    # patch FAISS index to fake but correct shapes
    class FakeIndex:
        def search(self, q_vec, top_k):
            # Return dummy distances and indices
            return np.zeros((1, top_k)), np.zeros((1, top_k), dtype=int)
    rag.index = FakeIndex()

    # add docs to memory
    rag.documents.extend(["Hello world", "Another chunk"])

    answer = rag.query("Hello?")
    assert isinstance(answer, str)
