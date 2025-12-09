"""
rag_service.py
-----------------------------------------
Free-tier RAG service for LLMOps project.
Uses only:
- Local LLM via llama-cpp-python
- SentenceTransformers embeddings
- FAISS vector store
No cloud, no API keys, fully free
"""

from pathlib import Path
from typing import List, Optional
import os
import pickle

from .model_loader import ModelLoader

# --------------------------------------------------------------
# RAG SERVICE CLASS
# --------------------------------------------------------------
class RAGService:
    """
    Handles:
      1. Document ingestion
      2. FAISS index management
      3. Querying and answer generation via ModelLoader
    """

    def __init__(self,
                 model_loader: Optional[ModelLoader] = None,
                 faiss_dir: str = "faiss_index"):
        self.faiss_dir = Path(faiss_dir)
        self.faiss_dir.mkdir(exist_ok=True)
        self.loader = model_loader or ModelLoader(faiss_dir=str(self.faiss_dir))
        self.documents = self.loader.documents  # original text chunks

    # --------------------------------------------------------------
    # INGEST DOCUMENTS
    # --------------------------------------------------------------
    def ingest_documents(self, texts: List[str]):
        """
        Split and store documents, generate embeddings, and update FAISS index.
        """
        if not texts:
            return

        # append new docs
        self.documents.extend(texts)

        if not self.loader.embedder:
            raise RuntimeError("Embedding model not loaded.")

        # compute embeddings
        embeddings = self.loader.embed(texts).astype("float32")

        # init / update FAISS index
        if self.loader.index is None:
            import faiss
            dim = embeddings.shape[1]
            self.loader.index = faiss.IndexFlatL2(dim)

        self.loader.index.add(embeddings)

        # save FAISS index and documents
        self._save_index()

    # --------------------------------------------------------------
    # QUERY / CHAT
    # --------------------------------------------------------------
    def query(self, question: str, top_k: int = 3, max_tokens: int = 256) -> str:
        """
        Run retrieval then generate answer using local LLM.
        """
        return self.loader.answer_from_rag(question, max_tokens=max_tokens)

    # --------------------------------------------------------------
    # INDEX SAVE / LOAD
    # --------------------------------------------------------------
    def _save_index(self):
        """
        Save FAISS index + document chunks to disk.
        """
        if self.loader.index is None:
            return

        index_path = self.faiss_dir / "index.bin"
        doc_path = self.faiss_dir / "docs.txt"

        import faiss
        faiss.write_index(self.loader.index, str(index_path))

        # save docs
        with open(doc_path, "w", encoding="utf-8") as f:
            for d in self.documents:
                f.write(d.replace("\n", " ") + "\n")

    def load_index(self):
        """
        Reload FAISS index and docs from disk.
        """
        index_path = self.faiss_dir / "index.bin"
        doc_path = self.faiss_dir / "docs.txt"

        if not index_path.exists():
            print("[INFO] FAISS index not found. Starting empty.")
            return

        import faiss
        self.loader.index = faiss.read_index(str(index_path))

        if doc_path.exists():
            with open(doc_path, "r", encoding="utf-8") as f:
                self.documents = [line.strip() for line in f.readlines()]

# --------------------------------------------------------------
# HELPER FUNCTIONS (optional)
# --------------------------------------------------------------
def create_rag_service(faiss_dir: str = "faiss_index") -> RAGService:
    """
    Factory for RAGService with default ModelLoader
    """
    loader = ModelLoader(faiss_dir=faiss_dir)
    service = RAGService(model_loader=loader, faiss_dir=faiss_dir)
    service.load_index()
    return service
