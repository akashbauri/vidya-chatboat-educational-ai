import streamlit as st
import os
import google.generativeai as genai
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import PyPDF2
from pptx import Presentation
import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
import json
from datetime import datetime

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Vidya Chatbot - AI Study Assistant",
    page_icon="🧩",
    layout="wide"
)

# ---------------- GEMINI INITIALIZATION ----------------
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

    # Detect installed SDK version and available models
    sdk_version = genai.__version__
    available_models = [m.name for m in genai.list_models()]

    st.sidebar.write("📦 Gemini SDK Version:", sdk_version)
    st.sidebar.write("✅ Available Models:", available_models)

    # ✅ Auto-detect and choose the best available Gemini model dynamically
    model_name = None
    preferred_order = [
        "models/gemini-2.5-flash",                      # latest stable flash
        "models/gemini-2.5-flash-preview-05-20",        # preview variant
        "models/gemini-2.5-flash-lite-preview-06-17",   # lightweight version
        "models/gemini-2.5-pro-preview-05-06",          # pro preview
        "models/gemini-2.5-pro-preview-03-25",          # older pro preview
    ]

    for name in preferred_order:
        if name in available_models:
            model_name = name
            break

    if model_name:
        model = genai.GenerativeModel(model_name)
        st.sidebar.success(f"✅ Using Model: {model_name}")
    else:
        # As a last fallback, try the latest detected model automatically
        for m in available_models:
            if "gemini" in m:
                model_name = m
                model = genai.GenerativeModel(model_name)
                st.sidebar.warning(f"⚙️ Using fallback model: {model_name}")
                break
        if not model_name:
            st.error("⚠️ No supported Gemini model found. Please check your API key.")
            model = None

except Exception as e:
    st.error(f"⚠️ Gemini initialization failed: {e}")
    model = None


# ---------------- EMBEDDING MODEL ----------------
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

embedding_model = load_embedding_model()

# ---------------- SESSION STATE ----------------
for key in ['messages', 'vector_store', 'documents', 'embeddings']:
    if key not in st.session_state:
        st.session_state[key] = [] if key == 'messages' else None

# ---------------- CONFIG CONSTANTS ----------------
MAX_FILE_SIZE = 500 * 1024 * 1024  # 500 MB
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
SIMILARITY_THRESHOLD = 0.40
TOP_K = 6

# ---------------- TEXT EXTRACTION ----------------
def extract_text_from_pdf(file):
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        return "\n".join(page.extract_text() or "" for page in pdf_reader.pages)
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return None

def extract_text_from_pptx(file):
    try:
        prs = Presentation(file)
        return "\n".join(
            shape.text for slide in prs.slides for shape in slide.shapes if hasattr(shape, "text")
        )
    except Exception as e:
        st.error(f"Error reading PPTX: {e}")
        return None

def extract_text_from_url(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        for s in soup(["script", "style"]): s.decompose()
        text = " ".join(t.strip() for t in soup.get_text().splitlines() if t.strip())
        return text
    except Exception as e:
        st.error(f"Error fetching URL: {e}")
        return None

# ---------------- CHUNKING ----------------
def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks, start, text_length = [], 0, len(text)
    while start < text_length:
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

# ---------------- VECTOR STORE ----------------
def create_vector_store(documents):
    if not documents:
        return None, None
    with st.spinner("Creating embeddings..."):
        embeddings = np.array(embedding_model.encode(documents, show_progress_bar=True)).astype('float32')
        faiss.normalize_L2(embeddings)
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
    return index, embeddings

def retrieve_relevant_chunks(query, top_k=TOP_K):
    if st.session_state.vector_store is None:
        return []
    query_embedding = np.array(embedding_model.encode([query])).astype('float32')
    faiss.normalize_L2(query_embedding)
    distances, indices = st.session_state.vector_store.search(query_embedding, top_k)
    return [
        {'text': st.session_state.documents[i], 'similarity': float(d)}
        for d, i in zip(distances[0], indices[0]) if d >= SIMILARITY_THRESHOLD
    ]

# ---------------- WEB SEARCH ----------------
def web_search(query, num_results=3):
    try:
        with DDGS() as ddgs:
            return list(ddgs.text(query, max_results=num_results))
    except Exception as e:
        st.error(f"Web search error: {e}")
        return []

# ---------------- GEMINI RESPONSE ----------------
def generate_response(query, context_chunks=None, use_web_search=False, web_results=None):
    if not model:
        return "⚠️ Gemini model not initialized properly. Please check your API key or SDK version."

    if context_chunks:
        context = "\n\n".join(
            f"[Chunk {i+1}] {chunk['text'][:500]}" for i, chunk in enumerate(context_chunks)
        )
        prompt = f"""
You are Vidya, an intelligent educational assistant.
Use the following context to answer the question clearly and accurately.

Context from uploaded materials:
{context}

Question: {query}

Guidelines:
- Use simple language (student-friendly)
- Cite chunks like [Chunk X]
- If context insufficient, say so politely
- Keep response under 400 tokens
"""
    elif use_web_search and web_results:
        web_context = "\n\n".join(
            f"[Source {i+1}] {r['title']}: {r['body'][:300]}" for i, r in enumerate(web_results)
        )
        prompt = f"""
You are Vidya, a helpful educational assistant.
These are search results from the web.

Web Results:
{web_context}

Question: {query}

Guidelines:
- Summarize clearly
- Cite sources using [Source X]
- Use simple educational tone
"""
    else:
        prompt = f"""
You are Vidya, an educational assistant.
Question: {query}
No materials are uploaded yet. Ask if the user wants to upload study materials or search online.
"""

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"⚠️ Error generating response: {e}")
        return "I encountered an issue generating a response. Please try again."

# ---------------- UI SECTION ----------------
st.title("🧩 Vidya Chatbot")
st.markdown("### AI-Powered Educational Assistant")
st.markdown("*Ask questions about your study materials or any topic!*")

# Sidebar for Uploads
with st.sidebar:
    st.header("📚 Upload Study Materials")

    uploaded_files = st.file_uploader(
        "Upload PDF or PPTX files", type=['pdf', 'pptx'], accept_multiple_files=True
    )
    url_input = st.text_input("Or enter a URL:")

    if st.button("Process Materials", type="primary"):
        all_text = []
        for file in uploaded_files or []:
            if file.size > MAX_FILE_SIZE:
                st.error(f"❌ {file.name} exceeds 500 MB limit")
                continue
            with st.spinner(f"Processing {file.name}..."):
                text = (
                    extract_text_from_pdf(file) if file.name.endswith(".pdf")
                    else extract_text_from_pptx(file)
                )
                if text:
                    all_text.append(text)
                    st.success(f"✅ {file.name} processed")

        if url_input:
            with st.spinner(f"Fetching {url_input}..."):
                text = extract_text_from_url(url_input)
                if text:
                    all_text.append(text)
                    st.success("✅ URL processed")

        if all_text:
            combined_text = "\n\n".join(all_text)
            chunks = chunk_text(combined_text)
            st.session_state.documents = chunks
            index, embeddings = create_vector_store(chunks)
            st.session_state.vector_store = index
            st.session_state.embeddings = embeddings
            st.success(f"✨ Created {len(chunks)} searchable chunks!")
        else:
            st.warning("No valid materials found.")

    st.markdown("---")
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

    if st.button("📥 Export Chat"):
        if st.session_state.messages:
            chat_export = {
                'timestamp': datetime.now().isoformat(),
                'messages': st.session_state.messages
            }
            st.download_button(
                label="Download JSON",
                data=json.dumps(chat_export, indent=2),
                file_name=f"vidya_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

    st.markdown("---")
    st.markdown("**📊 Status**")
    if st.session_state.vector_store:
        st.info(f"✅ {len(st.session_state.documents)} chunks loaded")
    else:
        st.warning("⚠️ No materials loaded")

# ---------------- CHAT INTERFACE ----------------
st.markdown("---")

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Ask me anything about your materials..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            chunks = retrieve_relevant_chunks(prompt)
            if chunks:
                response = generate_response(prompt, chunks)
            else:
                st.info("🔍 Searching the web for information...")
                web_results = web_search(prompt)
                response = generate_response(prompt, None, True, web_results)
            st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "Developed by <strong>Akash Bauri</strong> | "
    "Powered by <strong>Gemini (Auto-Detected)</strong> & RAG Architecture"
    "</div>",
    unsafe_allow_html=True
)
