install:
    pip install -r requirements.txt

run:
    uvicorn main:app --host 0.0.0.0 --port 8000

embed:
    python scripts/embed_data.py

evaluate:
    python scripts/evaluate_local.py

hf-space-start:
    cd hf_space && python app.py
