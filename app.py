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
import re

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Vidya Chatbot - AI Study Assistant",
    page_icon="🧩",
    layout="wide"
)

# ============================================================
# GEMINI INITIALIZATION
# ============================================================
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    sdk_version = genai.__version__
    available_models = [m.name for m in genai.list_models()]

    st.sidebar.write("📦 Gemini SDK Version:", sdk_version)
    st.sidebar.write("✅ Available Models:", available_models)

    model_name = None
    preferred_order = [
        "models/gemini-2.5-flash",
        "models/gemini-2.5-flash-preview-05-20",
        "models/gemini-2.5-flash-lite-preview-06-17",
        "models/gemini-2.5-pro-preview-05-06",
        "models/gemini-2.5-pro-preview-03-25",
    ]

    for name in preferred_order:
        if name in available_models:
            model_name = name
            break

    if model_name:
        model = genai.GenerativeModel(model_name)
        st.sidebar.success(f"✅ Using Model: {model_name}")
    else:
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

# ============================================================
# EMBEDDING MODEL
# ============================================================
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

embedding_model = load_embedding_model()

# ============================================================
# SESSION STATE
# ============================================================
for key in ['messages', 'vector_store', 'documents', 'embeddings', 'sources']:
    if key not in st.session_state:
        st.session_state[key] = [] if key == 'messages' else None

# ============================================================
# CONFIG CONSTANTS
# ============================================================
MAX_FILE_SIZE = 500 * 1024 * 1024
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
SIMILARITY_THRESHOLD = 0.25
TOP_K = 6

# ============================================================
# TEXT EXTRACTION
# ============================================================
def extract_text_from_pdf(file):
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for i, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text()
            if page_text:
                text += f"\n[Source: {file.name}, Page {i+1}]\n{page_text}"
        return text or None
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return None

def extract_text_from_pptx(file):
    try:
        prs = Presentation(file)
        text = ""
        for i, slide in enumerate(prs.slides):
            slide_text = ""
            for shape in slide.shapes:
                if hasattr(shape, "text"):
                    slide_text += shape.text + "\n"
            if slide_text.strip():
                text += f"\n[Source: {file.name}, Slide {i+1}]\n{slide_text}"
        return text or None
    except Exception as e:
        st.error(f"Error reading PPTX: {e}")
        return None

def extract_text_from_url(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        for s in soup(["script", "style"]):
            s.decompose()
        text = " ".join(t.strip() for t in soup.get_text().splitlines() if t.strip())
        return f"[Source: {url}]\n{text}" if text else None
    except Exception as e:
        st.error(f"Error fetching URL: {e}")
        return None

# ============================================================
# CHUNKING
# ============================================================
def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    if not text:
        return []
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        start += max(1, chunk_size - overlap)
    
    return chunks

# ============================================================
# VECTOR STORE
# ============================================================
def create_vector_store(documents):
    if not documents:
        return None, None

    cleaned_docs = []
    
    for idx, d in enumerate(documents):
        if d is None:
            continue
            
        if isinstance(d, (list, tuple)):
            try:
                d = " ".join(str(x) for x in d if x is not None)
            except:
                continue
        
        try:
            d = str(d)
        except:
            continue
        
        d = d.strip()
        
        if len(d) >= 3:
            cleaned_docs.append(d)
    
    if not cleaned_docs:
        st.warning("No valid text chunks found after cleaning.")
        return None, None
    
    try:
        with st.spinner("Creating embeddings..."):
            embeddings = embedding_model.encode(
                cleaned_docs,
                batch_size=32,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=False
            )
            
            embeddings = np.array(embeddings, dtype='float32')
            faiss.normalize_L2(embeddings)
            
            dimension = embeddings.shape[1]
            index = faiss.IndexFlatIP(dimension)
            index.add(embeddings)
            
            return index, embeddings
            
    except Exception as e:
        st.error(f"❌ Error creating embeddings: {str(e)}")
        return None, None

def clean_text_for_embedding(text):
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def retrieve_relevant_chunks(query, top_k=TOP_K):
    if st.session_state.vector_store is None:
        return []
    
    query_clean = clean_text_for_embedding(query)
    if not query_clean:
        return []
    
    try:
        query_embedding = embedding_model.encode(
            [query_clean],
            convert_to_numpy=True,
            normalize_embeddings=False
        )
        query_embedding = np.array(query_embedding, dtype='float32')
        faiss.normalize_L2(query_embedding)
        
        distances, indices = st.session_state.vector_store.search(query_embedding, top_k)
        
        results = []
        for d, i in zip(distances[0], indices[0]):
            if d >= SIMILARITY_THRESHOLD and 0 <= i < len(st.session_state.documents):
                results.append({
                    'text': st.session_state.documents[i],
                    'similarity': float(d)
                })
        
        return results
        
    except Exception as e:
        st.error(f"Error retrieving chunks: {e}")
        return []

# ============================================================
# WEB SEARCH
# ============================================================
def web_search(query, num_results=3):
    try:
        with DDGS() as ddgs:
            return list(ddgs.text(query, max_results=num_results))
    except Exception as e:
        st.error(f"Web search error: {e}")
        return []

# ============================================================
# GEMINI RESPONSE (IMPROVED PROMPTS)
# ============================================================
def generate_response(query, context_chunks=None, web_results=None):
    if not model:
        return "⚠️ Gemini model not initialized properly."

    if context_chunks:
        # Build context from retrieved chunks
        context = "\n\n".join(
            f"[Context {i+1}] {chunk['text'][:600]}" for i, chunk in enumerate(context_chunks)
        )
        
        # IMPROVED PROMPT - More natural, conversational responses
        prompt = f"""You are Vidya, a friendly and helpful educational AI assistant. A student has asked you a question about their study materials.

Context from the student's materials:
{context}

Student's Question: {query}

Instructions:
- Give a clear, direct, and easy-to-understand answer
- Write naturally as if explaining to a friend
- Synthesize information from the context instead of listing it
- If the question asks for code, provide complete, working code examples
- Add sources at the END of your answer in a "Sources" section, not inline
- Be concise but thorough
- Use bullet points or numbered lists when appropriate
- If multiple files mention the same concept, combine them smoothly

Format your response like this:
[Your clear, helpful answer here]

**Sources:**
- [File name, Page X]
- [File name, Page Y]
"""
    
    elif web_results:
        # Build context from web search
        web_context = "\n\n".join(
            f"[Result {i+1}] {r['title']}\n{r['body'][:400]}\nURL: {r['href']}"
            for i, r in enumerate(web_results)
        )
        
        # IMPROVED WEB SEARCH PROMPT
        prompt = f"""You are Vidya, a friendly educational AI assistant. A student asked a question, and I searched the web for information.

Web Search Results:
{web_context}

Student's Question: {query}

Instructions:
- Give a clear, helpful answer based on the search results
- Write naturally and conversationally
- If the question asks for code or examples, provide them
- Cite sources at the END in a "Sources" section
- Be accurate but easy to understand
- Don't say "based on the search results" - just answer directly

Format:
[Your helpful answer]

**Sources:**
- [Website name - URL]
"""
    
    else:
        # No context - use general knowledge
        prompt = f"""You are Vidya, a friendly educational AI assistant.

Student's Question: {query}

Instructions:
- Answer clearly and helpfully using your knowledge
- If it's a coding question, provide complete working code
- Use simple language suitable for students
- Be concise but thorough
- Use examples when helpful
"""

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"⚠️ Error generating response: {e}")
        return "I encountered an issue generating a response."

# ============================================================
# UI SECTION
# ============================================================
st.title("🧩 Vidya Chatbot")
st.markdown("### AI-Powered Educational Assistant")
st.markdown("*Ask questions about your study materials or any topic!*")

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.header("📚 Upload Study Materials")
    uploaded_files = st.file_uploader("Upload PDF or PPTX files", type=['pdf', 'pptx'], accept_multiple_files=True)
    url_input = st.text_input("Or enter a URL:")

    if st.button("Process Materials", type="primary"):
        all_text = []
        
        for file in uploaded_files or []:
            if file.size > MAX_FILE_SIZE:
                st.error(f"❌ {file.name} exceeds 500 MB limit")
                continue
                
            with st.spinner(f"Processing {file.name}..."):
                if file.name.lower().endswith(".pdf"):
                    text = extract_text_from_pdf(file)
                else:
                    text = extract_text_from_pptx(file)
                    
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
            
            if chunks:
                st.session_state.documents = chunks
                index, embeddings = create_vector_store(chunks)
                st.session_state.vector_store = index
                st.session_state.embeddings = embeddings
                
                if index is not None:
                    st.success(f"✨ Created {len(chunks)} searchable chunks!")
            else:
                st.warning("No chunks created from materials.")
        else:
            st.warning("No valid materials found.")

    st.markdown("---")
    
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

    st.markdown("**📊 Status**")
    if st.session_state.vector_store:
        st.info(f"✅ {len(st.session_state.documents)} chunks loaded")
    else:
        st.warning("⚠️ No materials loaded")

# ============================================================
# CHAT INTERFACE
# ============================================================
st.markdown("---")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask me anything about your materials..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            chunks = retrieve_relevant_chunks(prompt)
            
            if chunks and len(chunks) > 0:
                response = generate_response(prompt, chunks)
            else:
                st.info("🔍 Searching the web for information...")
                web_results = web_search(prompt)
                response = generate_response(prompt, None, web_results)
                
            st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "Developed by <strong>Akash Bauri</strong> | "
    "Powered by <strong>Gemini (Hybrid RAG + Web)</strong>"
    "</div>",
    unsafe_allow_html=True
)
