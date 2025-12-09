# models/

This directory should hold local model weights for running inference without cloud APIs.

Recommended:
- small quantized ggml/gguf models for llama-cpp (7B q4_0 or smaller) placed as:
  models/ggml-model-q4_0.bin

Where to get models:
- Search Hugging Face for "ggml" or "gguf" builds (TheBloke has many community builds).
- Always check license/usage terms before downloading.

Note: large model files should not be committed to the repo. Add them to `.gitignore`:

    models/*
    !models/README.md
