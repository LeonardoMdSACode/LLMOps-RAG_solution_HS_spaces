from pathlib import Path
from typing import List, Optional
import yaml
import numpy as np

try:
    from llama_cpp import Llama
except Exception:
    Llama = None

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None


# Load config
CFG_PATH = Path(__file__).resolve().parent.parent.parent / "configs" / "default.yaml"
if CFG_PATH.exists():
    with open(CFG_PATH, "r") as f:
        _CFG = yaml.safe_load(f)
else:
    _CFG = {
        "model_path": "models/qwen2.5-0.5b-instruct-q4_0.gguf",
        "embed_model": "sentence-transformers/all-MiniLM-L6-v2",
        "faiss_dir": "faiss_index",
        "chunk_size": 1000,
        "chunk_overlap": 200
    }


class ModelLoader:
    def __init__(
        self,
        model_path: Optional[str] = None,
        embed_model_name: Optional[str] = None,
        faiss_dir: Optional[str] = None,
        n_ctx: int = 2048,  # 0.5B models cannot handle 4k context well
    ):
        self.model_path = Path(model_path or _CFG.get("model_path"))
        self.embed_model_name = embed_model_name or _CFG.get("embed_model")
        self.faiss_dir = Path(faiss_dir or _CFG.get("faiss_dir"))
        self.n_ctx = n_ctx

        self.llm = None
        self.embedder = None
        self.index = None
        self.documents: List[str] = []

        self._load_all()

    def _load_llm(self):
        if not self.model_path.exists():
            print(f"[WARN] LLM model not found: {self.model_path}")
            return None

        if Llama is None:
            print("[WARN] llama-cpp-python missing.")
            return None

        print(f"[INFO] Loading local LLM: {self.model_path}")

        return Llama(
            model_path=str(self.model_path),
            n_ctx=self.n_ctx,
            n_threads=4,
            n_gpu_layers=0
        )

    def _load_embedder(self):
        if SentenceTransformer is None:
            print("[WARN] sentence-transformers missing.")
            return None

        print(f"[INFO] Loading embedder: {self.embed_model_name}")
        return SentenceTransformer(self.embed_model_name)

    def _load_all(self):
        self.llm = self._load_llm()
        self.embedder = self._load_embedder()
        self.index = None

    def embed(self, texts: List[str]):
        if self.embedder is None:
            raise RuntimeError("Embedder is missing.")
        return self.embedder.encode(texts, show_progress_bar=False)

    def chat(self, prompt: str, max_tokens: int = 256) -> str:
        if not self.llm:
            return "[Local LLM missing — place a .gguf model inside models/]"

        out = self.llm(
            prompt,
            max_tokens=max_tokens,
            temperature=0.7,
            top_p=0.9,
            echo=False
        )

        try:
            return out["choices"][0]["text"].strip()
        except Exception:
            return str(out)

    def answer_from_rag(self, query: str, max_tokens: int = 256) -> str:
        return self.chat(query, max_tokens=max_tokens)
