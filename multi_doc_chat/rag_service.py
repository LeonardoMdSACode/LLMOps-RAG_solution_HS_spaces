"""
multi_doc_chat/rag_service.py
RAG service using ModelLoader, FAISS, and local embeddings.
"""

from pathlib import Path
from typing import List, Optional
import numpy as np
import yaml

from .model_loader import ModelLoader

try:
    import faiss
except Exception:
    faiss = None

# load config
CFG_PATH = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"
if CFG_PATH.exists():
    with open(CFG_PATH, "r") as f:
        _CFG = yaml.safe_load(f)
else:
    _CFG = {"faiss_dir": "faiss_index"}

class RAGService:
    def __init__(self, model_loader: Optional[ModelLoader] = None, faiss_dir: Optional[str] = None):
        cfg_faiss = faiss_dir or _CFG.get("faiss_dir", "faiss_index")
        self.faiss_dir = Path(cfg_faiss)
        self.faiss_dir.mkdir(parents=True, exist_ok=True)
        self.loader = model_loader or ModelLoader(faiss_dir=str(self.faiss_dir))
        self.documents: List[str] = self.loader.documents or []
        self.index = None
        if faiss:
            self._try_load_index()

    def _try_load_index(self):
        idx_path = self.faiss_dir / "index.bin"
        docs_path = self.faiss_dir / "docs.txt"
        if idx_path.exists() and faiss:
            self.index = faiss.read_index(str(idx_path))
            if docs_path.exists():
                with open(docs_path, "r", encoding="utf-8") as f:
                    self.documents = [line.rstrip("\n") for line in f.readlines()]

    def ingest_documents(self, texts: List[str]):
        if not texts:
            return
        # extend docs
        start_index = len(self.documents)
        self.documents.extend(texts)

        if self.loader.embedder is None:
            raise RuntimeError("Embedding model not loaded.")

        embeddings = self.loader.embed(texts).astype("float32")
        if faiss is None:
            raise RuntimeError("faiss not available (install faiss-cpu)")

        if self.index is None:
            dim = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dim)

        self.index.add(embeddings)
        # persist
        idx_path = self.faiss_dir / "index.bin"
        docs_path = self.faiss_dir / "docs.txt"
        faiss.write_index(self.index, str(idx_path))
        with open(docs_path, "w", encoding="utf-8") as f:
            for d in self.documents:
                f.write(d.replace("\n", " ")+"\n")

    def search(self, query: str, top_k: int = 3):
        if self.index is None:
            return []
        q_vec = self.loader.embed([query]).astype("float32")
        distances, indices = self.index.search(q_vec, top_k)
        results = []
        for idx in indices[0]:
            if 0 <= idx < len(self.documents):
                results.append(self.documents[idx])
        return results

    def query(self, question: str, top_k: int = 3, max_tokens: int = 256) -> str:
        # retrieve
        chunks = self.search(question, top_k=top_k)
        context = "\n\n".join(chunks)
        prompt = f"You are an assistant. Use the context to answer the question.\n\nCONTEXT:\n{context}\n\nQUESTION: {question}\n\nANSWER:"
        return self.loader.answer_from_rag(prompt, max_tokens=max_tokens)


def create_rag_service(faiss_dir: str = "faiss_index") -> RAGService:
    loader = ModelLoader()
    rs = RAGService(model_loader=loader, faiss_dir=faiss_dir)
    return rs
