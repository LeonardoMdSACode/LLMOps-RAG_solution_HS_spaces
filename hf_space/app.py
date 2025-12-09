# hf_space/app.py
import os
from pathlib import Path
import gradio as gr
from scripts.download_models import main as download_models_main
from multi_doc_chat.src.document_ingestion.data_ingestion import ingest_upload_files
from multi_doc_chat.rag_service import create_rag_service

# download models at container start (idempotent)
download_models_main()

rag_service = create_rag_service()

def ingest_files(files):
    if not files:
        return "No files provided"
    # ingest (files are file-like objects from Gradio)
    try:
        ingest_upload_files(files)
        return "Ingested files successfully."
    except Exception as e:
        return f"Ingest error: {e}"

def submit_chat(history, user_message):
    if not user_message:
        return history
    try:
        ans = rag_service.query(user_message)
        return history + [(user_message, ans)]
    except Exception as e:
        return history + [(user_message, f"Error: {e}")]

with gr.Blocks() as demo:
    gr.Markdown("# MultiDocChat — HF Space (Docker)\nUpload PDF/TXT, then ask questions.")
    with gr.Tab("Upload"):
        file_input = gr.File(file_count="multiple", file_types=[".pdf", ".txt"], type="file")
        ingest_btn = gr.Button("Ingest")
        status = gr.Textbox()
        ingest_btn.click(ingest_files, inputs=file_input, outputs=status)
    with gr.Tab("Chat"):
        chatbot = gr.Chatbot()
        state = gr.State([])
        txt = gr.Textbox()
        send = gr.Button("Send")
        send.click(submit_chat, [state, txt], [chatbot])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
