import streamlit as st
import os
from pathlib import Path
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import numpy as np
from numpy.linalg import norm
from dotenv import load_dotenv
from typing import List, Tuple
from groq import Groq

# --- Load secrets ---
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "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")
PDF_PATH = "ConvAI Documentation1.pdf"

# -------------------------
# Utilities: PDF -> chunks -> embeddings
# -------------------------
@st.cache_resource
def load_embedder(name="all-MiniLM-L6-v2"):
    return SentenceTransformer(name)

def pdf_to_chunks(pdf_path: str, chunk_size: int = 400, overlap: int = 80) -> List[Tuple[str,int]]:
    reader = PdfReader(pdf_path)
    chunks = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        start = 0
        while start < len(text):
            chunk = text[start:start+chunk_size].strip()
            if chunk:
                chunks.append((chunk, i+1))
            start += chunk_size - overlap
    return chunks

@st.cache_resource
def build_embeddings(chunks, _embedder):
    texts = [t for t,_ in chunks]
    embs = _embedder.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    return embs

def retrieve(query, _embedder, chunks, embs, k=4):
    q = _embedder.encode([query], convert_to_numpy=True)[0]
    sims = np.dot(embs, q) / (norm(embs, axis=1) * norm(q) + 1e-10)
    top_idx = sims.argsort()[::-1][:k]
    return [chunks[i] for i in top_idx]

# -------------------------
# Simple emotion detector
# -------------------------
def detect_emotion(text: str) -> str:
    low = text.lower()
    if any(w in low for w in ["sad","depressed","down","stressed","unhappy"]):
        return "stressed"
    if any(w in low for w in ["tired","exhausted","lazy","busy"]):
        return "tired"
    if any(w in low for w in ["happy","great","good","excited","energetic"]):
        return "happy"
    return "neutral"

# -------------------------
# Groq API wrapper
# -------------------------
def call_groq_chat(prompt: str, api_key: str, model="llama-3.3-70b-versatile", max_tokens: int = 512):
    if not api_key:
        raise ValueError("Missing GROQ_API_KEY")
    client = Groq(api_key=api_key)
    chat_completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are NutriBot, a helpful nutrition assistant."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=max_tokens,
        temperature=0.2
    )
    return chat_completion.choices[0].message.content

# -------------------------
# Streamlit UI & logic
# -------------------------
st.set_page_config(page_title="NutriBot (Groq + RAG)", layout="wide")
st.title("NutriBot — Nutrition Assistant (RAG)")

# Sidebar
with st.sidebar:
    st.header("Settings")
    groq_key = st.text_input("Groq API key", value=GROQ_API_KEY or "", type="password")
    embedding_model = st.selectbox("Embedding model", ["all-MiniLM-L6-v2","all-mpnet-base-v2"])
    top_k = st.slider("How many doc snippets to use", 1, 8, 4)
    st.write("Using `ConvAI Documentation1.pdf` as knowledge base.")

embedder = load_embedder(embedding_model)
if not Path(PDF_PATH).exists():
    st.error(f"PDF not found: {PDF_PATH}")
    st.stop()

with st.spinner("Indexing PDF..."):
    chunks = pdf_to_chunks(PDF_PATH)
    embs = build_embeddings(chunks, embedder)

# Initialize session state
if "profile" not in st.session_state:
    st.session_state.profile = {"age": 25, "gender": "Prefer not to say", "diet": "no preference", "allergies": []}
if "fridge" not in st.session_state:
    st.session_state.fridge = []
if "history" not in st.session_state:
    st.session_state.history = []

# Profile
with st.expander("Profile", expanded=False):
    p = st.session_state.profile
    p["age"] = st.number_input("Age", value=p["age"], min_value=1, max_value=120)
    p["gender"] = st.selectbox("Gender", ["Prefer not to say","Male","Female","Other"], index=["Prefer not to say","Male","Female","Other"].index(p["gender"]))
    p["diet"] = st.selectbox("Diet", ["no preference","vegetarian","vegan","pescatarian","low-carb","keto","diabetic-friendly"], index=["no preference","vegetarian","vegan","pescatarian","low-carb","keto","diabetic-friendly"].index(p["diet"]))
    allergies = st.text_input("Allergies (comma separated)", value=",".join(p["allergies"]))
    p["allergies"] = [a.strip() for a in allergies.split(",") if a.strip()]

# Fridge
with st.expander("Fridge", expanded=False):
    ing = st.text_input("Add ingredient")
    if st.button("Add ingredient"):
        if ing.strip():
            st.session_state.fridge.append(ing.strip().lower())
    if st.session_state.fridge:
        st.write("Items:", ", ".join(st.session_state.fridge))
        if st.button("Clear fridge"):
            st.session_state.fridge = []

# Chat
st.subheader("Chat with NutriBot")
user_q = st.text_input("Ask a nutrition question", key="user_input")

if st.button("Ask") and user_q.strip():
    emotion = detect_emotion(user_q)
    retrieved = retrieve(user_q, embedder, chunks, embs, k=top_k)
    context = "\n\n".join([f"[page {p}] {t}" for t,p in retrieved])

    system = (
        "You are NutriBot — a helpful nutrition assistant. "
        "Use the CONTEXT (document excerpts) for factual answers. "
        "If context is insufficient, say so and answer with general nutrition knowledge. "
        "Cite page numbers when relevant. Be concise and practical.\n\n"
    )
    fridge_text = f"Available ingredients: {', '.join(st.session_state.fridge)}\n\n" if st.session_state.fridge else ""
    profile_text = f"User profile: age={p['age']}, diet={p['diet']}, allergies={','.join(p['allergies'])}\n\n"

    prompt = system + "CONTEXT:\n" + (context or "No relevant excerpts.") + "\n\n" + profile_text + fridge_text + f"USER QUESTION:\n{user_q}\n\nAssistant answer:"

    try:
        reply = call_groq_chat(prompt, groq_key)
    except Exception as e:
        reply = f"[Error] {e}"

    if emotion == "stressed":
        reply += "\n\n— Extra: Detected stress, suggest easy, comforting, healthy foods."
    if st.session_state.fridge:
        reply += f"\n\n— Extra: Prioritized recipes using {', '.join(st.session_state.fridge)}."

    st.session_state.history.append(("You", user_q))
    st.session_state.history.append(("NutriBot", reply))

# Display chat history (most recent on top)
for s, m in st.session_state.history[::-1]:
    st.markdown(f"**{s}:** {m}")

st.markdown("---")
st.caption("NutriBot uses ConvAI Documentation1.pdf as domain knowledge. Replace Groq model name with one available in your account.")
