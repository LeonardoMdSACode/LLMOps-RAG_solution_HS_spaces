from pathlib import Path
import requests
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")



MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

MODEL_LIST = [
    {
        "name": "qwen2.5-0.5b-instruct-q4_0",
        "filename": "qwen2.5-0.5b-instruct-q4_0.gguf",
        "url": "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_0.gguf"
    }
]

def download_file(url: str, dest: Path):
    if dest.exists():
        return
    resp = requests.get(url, stream=True)
    content_type = resp.headers.get("content-type", "")
    if "text/html" in content_type:
        raise ValueError(f"URL returned HTML, not a model file: {url}")
    total = int(resp.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as bar:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)
                bar.update(len(chunk))

def main():
    for m in MODEL_LIST:
        dest = MODELS_DIR / m["filename"]
        try:
            download_file(m["url"], dest)
        except Exception as e:
            print(f"Failed to download {m['name']}: {e}")

if __name__ == "__main__":
    main()
