import os
import json
import torch
import gradio as gr
from datetime import datetime
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Configuration
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOP_K = 6
MAX_CONTEXT_CHARS = 1800
MODEL_NAME = "google/flan-t5-large"
HISTORY_FILE = "chat_history.json"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# Load Model
print("Loading model, please wait...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(DEVICE)
print("Model loaded.")

# Knowledge base
class KnowledgeBase:
    def __init__(self):
        self.chunks = []
        self.sources = []
        self.documents = []
        self.vectorizer = None
        self.matrix = None

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
            return f"No extractable text in {filename}."
        new_chunks = self._chunk_text(full_text)
        self.chunks.extend(new_chunks)
        self.sources.extend([filename] * len(new_chunks))
        self.documents.append(filename)
        self._build_index()
        return f"Added {filename}: {len(new_chunks)} chunks."

    def _chunk_text(self, text):
        chunks = []
        start = 0
        n = len(text)
        while start < n:
            end = min(start + CHUNK_SIZE, n)
            chunks.append(text[start:end].strip())
            start += CHUNK_SIZE - CHUNK_OVERLAP
        return chunks

    def _build_index(self):
        if not self.chunks:
            return
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = self.vectorizer.fit_transform(self.chunks)

    def retrieve(self, query, top_k=TOP_K):
        if not self.chunks or self.vectorizer is None:
            return []
        query_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(query_vec, self.matrix).flatten()
        top_indices = sims.argsort()[::-1][:top_k]
        results = []
        for i in top_indices:
            if sims[i] > 0:
                results.append({
                    "text": self.chunks[i],
                    "source": self.sources[i],
                    "score": float(sims[i]),
                })
        return results

    def build_context(self, query):
        results = self.retrieve(query)
        context = ""
        sources = []
        total_chars = 0
        for r in results:
            if total_chars + len(r["text"]) > MAX_CONTEXT_CHARS:
                break
            context += r["text"] + "\n\n"
            sources.append(r["source"])
            total_chars += len(r["text"])
        if not context:
            context = "No relevant information found in uploaded documents."
        return context.strip(), list(dict.fromkeys(sources))

kb = KnowledgeBase()

# Chat history
def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            return json.load(f)
    return []

def save_history(history):
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f)

chat_history = load_history()

# Generate answer
def generate_answer(query, context, sources):
    if not context or "No relevant" in context:
        prompt = (
            "You are a power systems engineering tutor. The question is:\n"
            f"{query}\n"
            "There is no relevant information in the uploaded documents. "
            "Please answer based on your general knowledge, or say that "
            "the answer cannot be provided from the uploaded documents."
        )
    else:
        prompt = (
            "You are a power systems engineering tutor. Use only the information from "
            "the uploaded documents below to answer the question. If the information "
            "is insufficient, say so.\n\n"
            "Context:\n" + context + "\n\n"
            f"Question: {query}\n\n"
            "Answer:"
        )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(DEVICE)
    outputs = model.generate(
        **inputs,
        max_new_tokens=300,
        num_beams=4,
        length_penalty=1.2,
        no_repeat_ngram_size=3,
        early_stopping=True,
    )
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

# UI functions
def upload_pdfs(files):
    status_messages = []
    for f in files:
        msg = kb.add_pdf(f.name)
        status_messages.append(msg)
    return "\n".join(status_messages)

def chat(user_input, chat_history):
    timestamp = datetime.now().strftime("%H:%M")
    chat_history.append(("User", user_input))
    # Build context from PDFs
    context, sources = kb.build_context(user_input)
    # Generate answer
    answer = generate_answer(user_input, context, sources)
    if sources:
        answer += f"\n\n_Sources: {', '.join(sources)}_"
    else:
        answer += "\n\n(No sources found.)"
    chat_history.append(("Assistant", answer))
    # Save history
    save_history([{"role": role, "content": msg} for role, msg in chat_history])
    return chat_history

def clear_chat():
    global chat_history
    chat_history = []
    save_history([])
    return []

# Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown(
        """
        # Power Systems RAG Learning Assistant
        <br>
        Upload PDFs related to power systems, then ask questions!
        """
    )
    with gr.Row():
        with gr.Column():
            upload_files = gr.File(label="Upload PDF(s)", file_types=[".pdf"], file_count="multiple")
            upload_button = gr.Button("Upload PDFs")
        with gr.Column():
            clear_button = gr.Button("Clear Chat")
    chat_box = gr.Chatbot()
    user_input = gr.Textbox(placeholder="Ask me about power systems...", label="Your Question")
    send_button = gr.Button("Send")

    # Callbacks
    upload_button.click(upload_pdfs, [upload_files], None)
    clear_button.click(clear_chat, None, chat_box)
    send_button.click(chat, [user_input, chat_box], chat_box)
    # Also allow pressing Enter to send
    user_input.submit(chat, [user_input, chat_box], chat_box)

# Launch the app
if __name__ == "__main__":
    demo.launch()
