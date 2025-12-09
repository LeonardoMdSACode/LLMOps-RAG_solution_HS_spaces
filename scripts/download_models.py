"""
download_models.py
-----------------------------------------
Downloads free, open-source LLM models into the `models/` folder
for the no-credit-card LLMOps RAG project.

- Uses Hugging Face `transformers` snapshot URLs or community GGUF builds
- Compatible with llama-cpp-python or gguf/ggml loaders
- Fully free-tier, no API keys required
"""

import os
from pathlib import Path
import requests
from tqdm import tqdm

# --------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

# Example models (small, quantized, free)
# Users can add more models here
MODEL_LIST = [
    {
        "name": "7B-q4_0",
        "filename": "ggml-model-q4_0.bin",
        "url": "https://huggingface.co/TheBloke/Llama-2-7B-GGML/resolve/main/ggml-model-q4_0.bin"
    },
    {
        "name": "7B-q4_1",
        "filename": "ggml-model-q4_1.bin",
        "url": "https://huggingface.co/TheBloke/Llama-2-7B-GGML/resolve/main/ggml-model-q4_1.bin"
    }
]

# --------------------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------------------
def download_file(url: str, dest: Path):
    """
    Download file with progress bar
    """
    if dest.exists():
        print(f"[INFO] File already exists: {dest}")
        return

    response = requests.get(url, stream=True)
    total_size = int(response.headers.get("content-length", 0))
    chunk_size = 1024 * 1024  # 1MB

    with open(dest, "wb") as f, tqdm(
        desc=f"Downloading {dest.name}",
        total=total_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                bar.update(len(chunk))

# --------------------------------------------------------------
# MAIN FUNCTION
# --------------------------------------------------------------
def main():
    print("[INFO] Starting model download for LLMOps RAG free-tier project...")
    for model in MODEL_LIST:
        dest_path = MODELS_DIR / model["filename"]
        print(f"[INFO] Downloading model '{model['name']}' to {dest_path}")
        try:
            download_file(model["url"], dest_path)
        except Exception as e:
            print(f"[ERROR] Failed to download {model['name']}: {e}")

    print("[INFO] All downloads completed.")
    print(f"[INFO] Models are saved under: {MODELS_DIR.resolve()}")

# --------------------------------------------------------------
# ENTRY POINT
# --------------------------------------------------------------
if __name__ == "__main__":
    main()
