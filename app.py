# =====================================================================
# <span style="color:red;">POWER SYSTEMS RAG LEARNING ASSISTANT — Google Colab</span>
# =====================================================================
# HOW TO USE:
#   1. Open a new Google Colab notebook (GPU runtime recommended:
#      Runtime > Change runtime type > T4 GPU)
#   2. Copy each "# ===== CELL n =====" block below into its OWN
#      Colab cell, in order, and run them top to bottom.
#   3. The last cell launches a Gradio app with a shareable link
#      and a chat UI styled like your reference screenshot.
# =====================================================================


# ===== CELL 1: Install dependencies =====
# NOTE: we use `pypdf` (pure Python) instead of poppler/pdf2image to
# avoid the apt-get 404 issues you hit before — no system packages needed.
!pip install -q transformers torch scikit-learn pypdf gradio accelerate


# ===== CELL 2: Imports & config =====
import os
import json
import time
import uuid
import textwrap
from datetime import datetime

import torch
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import gradio as gr

# ---- Config (matches your existing pipeline parameters) ----
CHUNK_SIZE = 800            # characters per chunk
CHUNK_OVERLAP = 150         # overlap between chunks
TOP_K = 6                   # chunks retrieved per query
MAX_CONTEXT_CHARS = 1800    # context budget passed to the model
GEN_MODEL_NAME = "google/flan-t5-large"

# Where chat history + knowledge base metadata persist for this runtime.
# Point this at a Google Drive path if you want it to survive across
# Colab restarts, e.g. "/content/drive/MyDrive/power_systems_rag/history.json"
HISTORY_FILE = "/content/chat_history.json"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")


# ===== CELL 3: (Optional) Mount Google Drive for persistence =====
# Uncomment if you want chat history and uploaded PDFs to persist
# across Colab sessions.
#
# from google.colab import drive
# drive.mount('/content/drive')
# os.makedirs("/content/drive/MyDrive/power_systems_rag", exist_ok=True)
# HISTORY_FILE = "/content/drive/MyDrive/power_systems_rag/history.json"


# ===== CELL 4: Knowledge base — PDF ingestion, chunking, TF-IDF index =====
class KnowledgeBase:
    def __init__(self):
        self.chunks = []          # list[str]
        self.chunk_sources = []   # list[str] (filename each chunk came from)
        self.documents = []       # list[str] (uploaded filenames)
        self.vectorizer = None
        self.matrix = None

    def _chunk_text(self, text):
        chunks = []
        start = 0
        n = len(text)
        while start < n:
            end = min(start + CHUNK_SIZE, n)
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            start += CHUNK_SIZE - CHUNK_OVERLAP
        return chunks

    def add_pdf(self, filepath):
        filename = os.path.basename(filepath)
        try:
            reader = PdfReader(filepath)
            full_text = ""
            for page in reader.pages:
                full_text += (page.extract_text() or "") + "\n"
        except Exception as e:
            return f"Failed to read {filename}: {e}"

        if not full_text.strip():
            return f"No extractable text found in {filename} (it may be scanned/image-based)."

        new_chunks = self._chunk_text(full_text)
        self.chunks.extend(new_chunks)
        self.chunk_sources.extend([filename] * len(new_chunks))
        self.documents.append(filename)
        self._rebuild_index()
        return f"Added {filename}: {len(new_chunks)} chunks indexed."

    def _rebuild_index(self):
        if not self.chunks:
            return
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = self.vectorizer.fit_transform(self.chunks)

    def retrieve(self, query, top_k=TOP_K):
        if not self.chunks or self.vectorizer is None:
            return []
        query_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(query_vec, self.matrix).flatten()
        top_idx = sims.argsort()[::-1][:top_k]
        results = []
        for i in top_idx:
            if sims[i] > 0:
                results.append({
                    "text": self.chunks[i],
                    "source": self.chunk_sources[i],
                    "score": float(sims[i]),
                })
        return results

    def build_context(self, query):
        results = self.retrieve(query)
        context = ""
        used_sources = []
        for r in results:
            if len(context) + len(r["text"]) > MAX_CONTEXT_CHARS:
                remaining = MAX_CONTEXT_CHARS - len(context)
                if remaining > 100:
                    context += r["text"][:remaining]
                break
            context += r["text"] + "\n\n"
            used_sources.append(r["source"])
        # If no relevant results, set context to a message
        if not context:
            context = "No relevant information found in uploaded documents."
        return context.strip(), list(dict.fromkeys(used_sources))


kb = KnowledgeBase()


# ===== CELL 5: Load local generation model (flan-t5-large) =====
print("Loading flan-t5-large... this can take a minute on first run.")
tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL_NAME)
gen_model = AutoModelForSeq2SeqLM.from_pretrained(GEN_MODEL_NAME).to(DEVICE)
print("Model loaded.")


def generate_answer(query, context, sources):
    # Check if context indicates no relevant info
    if not context or "No relevant information" in context:
        # Generate answer based on general knowledge or indicate no info
        prompt = (
            "You are a power systems engineering tutor. The question is:\n"
            f"{query}\n"
            "There is no relevant information in the uploaded documents. "
            "Please answer based on your general knowledge, or state that "
            "the answer cannot be provided from the uploaded documents."
        )
    else:
        # Use the retrieved document chunks as context
        prompt = (
            "You are a power systems engineering tutor. Use only the information from "
            "the uploaded documents below to answer the question. If the information "
            "is insufficient, say so.\n\n"
            "Context:\n" + context + "\n\n"
            f"Question: {query}\n\n"
            "Answer:"
        )

    # Generate the answer
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(DEVICE)
    outputs = gen_model.generate(
        **inputs,
        max_new_tokens=300,
        num_beams=4,
        length_penalty=1.2,
        no_repeat_ngram_size=3,
        early_stopping=True,
    )
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer


# ===== CELL 6: Chat history persistence =====
def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def save_history(history):
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)


chat_log = load_history()  # list of {"role": "user"/"assistant", "content": str, "time": str}


# ===== CELL 7: Gradio UI (styled like your screenshot) =====
CUSTOM_CSS = """
#app-container {
    background: #0d1526;
    border-radius: 14px;
    padding: 0;
    max-width: 760px;
    margin: 0 auto;
    font-family: 'Segoe UI', sans-serif;
}
#header-bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: #101b33;
    padding: 14px 18px;
    border-radius: 14px 14px 0 0;
    border-bottom: 1px solid #1c2740;
}
#header-title {
    color: #e8ecf5;
    font-weight: 600;
    font-size: 16px;
    margin: 0;
}
#header-sub {
    color: #6b7899;
    font-size: 12px;
    margin: 0;
}
.gradio-container {
    background: #0a1120 !important;
}
#chatbox {
    background: #0d1526 !important;
    border: none !important;
}
.message-wrap {
    background: transparent !important;
}
/* Chat bubbles styling */
.message.user, .user-row .message, [data-testid="user"] {
    background: #2f5fd8 !important;
    color: #ffffff !important;
    border-radius: 16px 16px 4px 16px !important;
}
.message.bot, .bot-row .message, [data-testid="bot"] {
    background: #1b2740 !important;
    color: #ffffff !important;
    border-radius: 16px 16px 16px 4px !important;
}
.message.bot p, .bot-row .message p, [data-testid="bot"] p,
.message.bot span, .bot-row .message span, [data-testid="bot"] span,
.message.bot li, .bot-row .message li, [data-testid="bot"] li {
    color: #ffffff !important;
}
#input-row textarea {
    background: #131e35 !important;
    color: #e8ecf5 !important;
    border-radius: 20px !important;
    border: 1px solid #22304d !important;
}
#send-btn {
    background: #2f5fd8 !important;
    color: white !important;
    border-radius: 50% !important;
}
#kb-panel {
    background: #101b33;
    border-radius: 10px;
    padding: 12px;
    color: #cdd6e8;
}
"""

def format_history_for_chatbot(history):
    """Convert stored history into Gradio's message format"""
    messages = []
    for turn in history:
        messages.append({"role": turn["role"], "content": turn["content"]})
    return messages

def handle_upload(files):
    if not files:
        return "No files selected.", gr.update(choices=kb.documents)
    status_lines = []
    for f in files:
        status_lines.append(kb.add_pdf(f.name))
    return "\n".join(status_lines), gr.update(choices=kb.documents)

def handle_send(user_message, history):
    if not user_message or not user_message.strip():
        return history, format_history_for_chatbot(history), ""

    timestamp = datetime.now().strftime("%H:%M")
    history.append({"role": "user", "content": user_message, "time": timestamp})

    # Build context from uploaded PDFs
    context, sources = kb.build_context(user_message)

    # Generate answer based on context
    if not context or "No relevant information" in context:
        answer = generate_answer(user_message, "", sources)
        answer += "\n\n(Note: This answer is based only on general knowledge or may be incomplete, as no relevant uploaded documents were found.)"
        sources = []
    else:
        answer = generate_answer(user_message, context, sources)

    if sources:
        answer_with_src = answer + f"\n\n_Sources: {', '.join(sources)}_"
    else:
        answer_with_src = answer

    # Append assistant response
    history.append({
        "role": "assistant",
        "content": answer_with_src,
        "time": datetime.now().strftime("%H:%M"),
    })
    save_history(history)
    return history, format_history_for_chatbot(history), ""

def handle_clear():
    global chat_log
    chat_log = []
    save_history(chat_log)
    return [], []

with gr.Blocks(css=CUSTOM_CSS, title="Power Systems RAG Assistant") as demo:
    with gr.Column(elem_id="app-container"):
        with gr.Row(elem_id="header-bar"):
            gr.Markdown(
                "<p id='header-title'>Power Systems Learning Assistant</p>"
                "<p id='header-sub'>RAG-powered • local flan-t5-large</p>"
            )
        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    value=format_history_for_chatbot(chat_log),
                    elem_id="chatbox",
                    height=460,
                    show_label=False,
                )
                with gr.Row(elem_id="input-row"):
                    user_input = gr.Textbox(
                        placeholder="Ask about power systems...", show_label=False, scale=5
                    )
                    send_btn = gr.Button("Send", elem_id="send-btn", scale=1)
                clear_btn = gr.Button("Clear chat history")
            with gr.Column(scale=2, elem_id="kb-panel"):
                gr.Markdown("### Knowledge base")
                file_upload = gr.File(
                    label="Upload PDF(s)", file_types=[".pdf"], file_count="multiple"
                )
                upload_status = gr.Textbox(label="Status", interactive=False)
                doc_list = gr.CheckboxGroup(
                    choices=kb.documents, label="Indexed documents", interactive=False
                )
        history_state = gr.State(chat_log)

        # Bind events
        file_upload.upload(handle_upload, [file_upload], [upload_status, doc_list])
        send_btn.click(handle_send, [user_input, history_state], [history_state, chatbot, user_input])
        user_input.submit(handle_send, [user_input, history_state], [history_state, chatbot, user_input])
        clear_btn.click(handle_clear, outputs=[history_state, chatbot])

# ===== CELL 8: Launch =====
demo.launch(share=True, debug=True)
