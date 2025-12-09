# hf_space/app.py
# Free-tier LLMOps RAG for Hugging Face Spaces (Gradio)
# Uses local open-source models only. No API keys required.
# Fully updated: PDF + TXT document ingestion, improved UI, stable HF Spaces version

import os
from pathlib import Path
import gradio as gr
import threading
import yaml
from scripts.download_models import main as download_models_main
from multi_doc_chat.rag_service import create_rag_service
from PyPDF2 import PdfReader
from io import BytesIO

# ----------------------------
# Ensure models are downloaded
# ----------------------------
download_models_main()

# ----------------------------
# Initialize RAGService
# ----------------------------
rag_service_instance = create_rag_service()


def load_config():
    cfg_path = Path(__file__).parent.parent / "configs" / "default.yaml"
    if cfg_path.exists():
        with open(cfg_path, "r") as f:
            return yaml.safe_load(f)
    return {
        "model_path": str(Path("models") / "ggml-model-q4_0.bin"),
        "embed_model": "sentence-transformers/all-MiniLM-L6-v2",
        "faiss_dir": "faiss_index"
    }


CFG = load_config()


# -------------------------------
# PDF Extractor
# -------------------------------
def pdf_to_text(file):
    try:
        data = BytesIO(file.read())
        reader = PdfReader(data)
        pages = []
        for p in reader.pages:
            txt = p.extract_text() or ""
            pages.append(txt)
        return "\n".join(pages)
    except Exception as e:
        return f"ERROR extracting PDF: {e}"


# -------------------------------
# CHAT
# -------------------------------
def submit_chat(history, user_message):
    if not user_message:
        return history
    try:
        answer = rag_service_instance.query(user_message)
        return history + [(user_message, answer)]
    except Exception as e:
        return history + [(user_message, f"Error: {e}")] 


# -------------------------------
# DOCUMENT INGESTION
# -------------------------------
def ingest_files(files):
    if not files:
        return "⚠️ No files provided."

    texts = []

    for f in files:
        name = f.name.lower()

        if name.endswith(".pdf"):
            txt = pdf_to_text(f)
        else:
            try:
                txt = f.read().decode("utf-8", errors="ignore")
            except:
                txt = ""

        if txt.strip():
            texts.append(txt)

    if not texts:
        return "⚠️ No readable text extracted."

    rag_service_instance.ingest_documents(texts)
    return f"✅ Successfully ingested {len(texts)} document(s)."


# -------------------------------
# GRADIO UI
# -------------------------------
with gr.Blocks(title="MultiDocChat - Free RAG") as demo:

    gr.Markdown("""
        # **MultiDocChat — 100% Free & Open RAG**
        Upload PDF or TXT documents → Ask questions → LLM answers using retrieval.

        **All offline models. No API keys. No OpenAI. Runs on CPU.**
    """)

    with gr.Tab("📄 Upload Documents"):
        file_input = gr.File(
            label="Upload PDF or TXT files",
            file_count="multiple",
            file_types=[".pdf", ".txt"],
            type="file"
        )
        ingest_output = gr.Textbox(label="Status", interactive=False)
        ingest_button = gr.Button("Ingest Documents")

        ingest_button.click(
            fn=ingest_files,
            inputs=file_input,
            outputs=ingest_output
        )

    with gr.Tab("💬 Chat"):
        chatbot = gr.Chatbot(height=400)
        user_msg = gr.Textbox(label="Ask a question about your documents")
        btn_send = gr.Button("Send")
        btn_clear = gr.Button("Clear Chat")
        state = gr.State([])

        btn_send.click(submit_chat, [state, user_msg], [chatbot])
        btn_clear.click(lambda: [], None, chatbot, queue=False)


# -------------------------------
# Launch (only for local run)
# -------------------------------
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
