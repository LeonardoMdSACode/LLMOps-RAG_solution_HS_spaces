# hf_space

This folder contains a Gradio app for deployment as a Hugging Face Space.

How to deploy:
1. Create a Hugging Face account (public Spaces do not require credit card).
2. Create a new Space with "Gradio" runtime (or Docker if you use the Dockerfile).
3. Push the contents of this folder to the HF Space repository (or copy-paste app.py).
4. Make sure to provide required model files under `models/` in the HF Space repo or use model download logic on startup.

Notes:
- Prefer small quantized models to keep startup time reasonable.
- If using Docker, place the Dockerfile contents under the Space repo root and set runtime to Docker.
