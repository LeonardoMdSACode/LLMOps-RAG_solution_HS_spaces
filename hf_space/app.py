# hf_space/app.py
from __future__ import annotations
import os
from pathlib import Path

# ensure models can be downloaded at container start (idempotent)
try:
    # If your download script is in scripts/download_models.py and exposes main()
    from scripts.download_models import main as download_models_main
    # Attempt download (this will return quickly if models exist)
    try:
        download_models_main()
    except Exception as e:
        # don't crash the entire container if download fails; log and continue
        print(f"[WARN] download_models_main() failed: {e}")

except Exception as ex:
    print(f"[WARN] download script not available or failed to import: {ex}")

# Import the main FastAPI app from the project (assumes main.py defines 'app')
try:
    # If your repo root has main.py with FastAPI app named 'app'
    from main import app as fastapi_app
except Exception as e:
    print(f"[ERROR] Could not import main.app: {e}")
    raise

if __name__ == "__main__":
    # run uvicorn on PORT (HF Spaces commonly expects 7860)
    import uvicorn
    port = int(os.getenv("PORT", "7860"))
    uvicorn.run(fastapi_app, host="0.0.0.0", port=port)
