"""
model_loader.py
-----------------------------------------
Free-tier model loader for the LLMOps project.
Uses only:
- llama-cpp-python (local GGUF/GGML models)
- sentence-transformers for embeddings
- optional FAISS for vector search

No AWS, no OpenAI, no credit card required.
"""

import os
from pathlib import Path
from typing import List, Optional

import numpy as np

try:
    from llama_cpp import Llama
except Exception:
    Llama = None  # handled gracefully

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

try:
    import faiss
except Exception:
    faiss = None


# --------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------
DEFAULT_MODEL_PATH = "models/ggml-model-q4_0.bin"
DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_FAISS_DIR = "faiss_index"


# --------------------------------------------------------------
# MAIN CLASS
# --------------------------------------------------------------
class ModelLoader:
    """
    Loads:
      1. Local LLM via llama-cpp-python
      2. SentenceTransformers embedding model
      3. Optional FAISS index for RAG
    """

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        embed_model_name: str = DEFAULT_EMBED_MODEL,
        faiss_dir: Optional[str] = DEFAULT_FAISS_DIR,
        n_ctx: int = 2048,
    ):
        self.model_path = Path(model_path)
        self.embed_model_name = embed_model_name
        self.faiss_dir = Path(faiss_dir) if faiss_dir else None
        self.n_ctx = n_ctx

        self.llm = None
        self.embedder = None
        self.index = None
        self.documents = []  # store original texts

        self._load_all()

    # --------------------------------------------------------------
    # LOADERS
    # --------------------------------------------------------------
    def _load_llm(self):
        if not self.model_path.exists():
            print(f"[WARNING] LLM model file not found: {self.model_path}")
            print("→ Place a GGUF/GGML model file in /models/ and update configs/default.yaml")
            return None

        if Llama is None:
            print("[WARNING] llama-cpp-python is not installed.")
            return None

        print(f"[INFO] Loading local LLM from {self.model_path} ...")
        return Llama(
            model_path=str(self.model_path),
            n_ctx=self.n_ctx,
            n_threads=4,
            temperature=0.7,
        )

    def _load_embedder(self):
        if SentenceTransformer is None:
            print("[ERROR] sentence-transformers is not installed.")
            return None

        print(f"[INFO] Loading embedding model: {self.embed_model_name}")
        return SentenceTransformer(self.embed_model_name)

    def _load_faiss(self):
        if self.faiss_dir is None:
            return None

        index_path = self.faiss_dir / "index.bin"
        doc_path = self.faiss_dir / "docs.txt"

        if not faiss:
            print("[WARNING] Faiss not available; skipping retrieval.")
            return None

        if not index_path.exists():
            print("[INFO] No FAISS index found — starting empty.")
            return None

        print(f"[INFO] Loading FAISS index from {index_path}")
        index = faiss.read_index(str(index_path))

        # load accompanying documents
        if doc_path.exists():
            with open(doc_path, "r", encoding="utf-8") as f:
                self.documents = [line.strip() for line in f.readlines()]

        return index

    def _load_all(self):
        self.llm = self._load_llm()
        self.embedder = self._load_embedder()
        self.index = self._load_faiss()

    # --------------------------------------------------------------
    # EMBEDDING
    # --------------------------------------------------------------
    def embed(self, texts: List[str]):
        if self.embedder is None:
            raise RuntimeError("Embedding model missing.")
        return self.embedder.encode(texts, show_progress_bar=False)

    # --------------------------------------------------------------
    # RAG SEARCH
    # --------------------------------------------------------------
    def search(self, query: str, top_k: int = 3):
        if not self.index:
            return []

        query_vec = self.embed([query]).astype("float32")
        distances, indices = self.index.search(query_vec, top_k)

        results = []
        for idx in indices[0]:
            if idx < len(self.documents):
                results.append(self.documents[idx])

        return results

    # --------------------------------------------------------------
    # LLM GENERATION
    # --------------------------------------------------------------
    def chat(self, prompt: str, max_tokens: int = 256) -> str:
        if not self.llm:
            return "[Local LLM missing — place a .gguf/.bin model in /models/]"

        output = self.llm(
            prompt,
            max_tokens=max_tokens,
            stop=["</s>"],
        )

        return output["choices"][0]["text"]

    # --------------------------------------------------------------
    # RAG + LLM
    # --------------------------------------------------------------
    def answer_from_rag(self, query: str, max_tokens=256):
        """Runs retrieval then generates answer."""
        context_chunks = self.search(query)
        context_text = "\n".join(context_chunks)

        prompt = (
            "You are a helpful assistant using retrieved context.\n\n"
            f"CONTEXT:\n{context_text}\n\n"
            f"QUESTION: {query}\n\n"
            "ANSWER:"
        )

        return self.chat(prompt, max_tokens=max_tokens)
