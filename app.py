import streamlit as st
import os
from openai import OpenAI
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
import io

# Page configuration
st.set_page_config(
    page_title="Vidya Chatbot - AI Study Assistant",
    page_icon="🧩",
    layout="wide"
)

# Initialize OpenAI client
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Initialize embedding model
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

embedding_model = load_embedding_model()

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'documents' not in st.session_state:
    st.session_state.documents = []
if 'embeddings' not in st.session_state:
    st.session_state.embeddings = None

# Configuration
MAX_FILE_SIZE = 500 * 1024 * 1024  # 500 MB
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
SIMILARITY_THRESHOLD = 0.40
TOP_K = 6
MAX_RESPONSE_TOKENS = 400

def extract_text_from_pdf(file):
    """Extract text from PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return None

def extract_text_from_pptx(file):
    """Extract text from PowerPoint file"""
    try:
        prs = Presentation(file)
        text = ""
        for slide in prs.slides:
            for shape in slide.shapes:
                if hasattr(shape, "text"):
                    text += shape.text + "\n"
        return text
    except Exception as e:
        st.error(f"Error reading PPTX: {str(e)}")
        return None

def extract_text_from_url(url):
    """Extract text from URL"""
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)
        
        return text
    except Exception as e:
        st.error(f"Error fetching URL: {str(e)}")
        return None

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into overlapping chunks"""
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    
    return chunks

def create_vector_store(documents):
    """Create FAISS vector store from documents"""
    if not documents:
        return None, None
    
    with st.spinner("Creating embeddings..."):
        embeddings = embedding_model.encode(documents, show_progress_bar=True)
        embeddings = np.array(embeddings).astype('float32')
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)  # Inner Product for cosine similarity
        index.add(embeddings)
        
    return index, embeddings

def retrieve_relevant_chunks(query, top_k=TOP_K):
    """Retrieve top-k relevant chunks for a query"""
    if st.session_state.vector_store is None:
        return []
    
    query_embedding = embedding_model.encode([query])
    query_embedding = np.array(query_embedding).astype('float32')
    faiss.normalize_L2(query_embedding)
    
    distances, indices = st.session_state.vector_store.search(query_embedding, top_k)
    
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if dist >= SIMILARITY_THRESHOLD:
            results.append({
                'text': st.session_state.documents[idx],
                'similarity': float(dist)
            })
    
    return results

def web_search(query, num_results=3):
    """Perform web search using DuckDuckGo"""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=num_results))
            return results
    except Exception as e:
        st.error(f"Web search error: {str(e)}")
        return []

def generate_response(query, context_chunks, use_web_search=False, web_results=None):
    """Generate response using GPT-4o"""
    
    if context_chunks:
        context = "\n\n".join([f"[Chunk {i+1}] {chunk['text'][:500]}" 
                                 for i, chunk in enumerate(context_chunks)])
        
        prompt = f"""You are Vidya, an intelligent educational assistant. Answer the question based on the provided context.

Context from uploaded materials:
{context}

Question: {query}

Instructions:
- Provide a clear, accurate answer based on the context
- Cite specific chunks using [Chunk X] notation
- If the context doesn't fully answer the question, say so
- Keep the response under 400 tokens
- Use simple language suitable for students"""

    elif use_web_search and web_results:
        web_context = "\n\n".join([f"[Source {i+1}] {r['title']}: {r['body'][:300]}" 
                                      for i, r in enumerate(web_results)])
        
        prompt = f"""You are Vidya, an intelligent educational assistant. The uploaded materials don't contain information about this question, so I've searched the web.

Web search results:
{web_context}

Question: {query}

Instructions:
- Provide a clear, accurate answer based on the web results
- Cite sources using [Source X] notation
- Keep the response under 400 tokens
- Use simple language suitable for students"""

    else:
        prompt = f"""You are Vidya, an intelligent educational assistant. 

Question: {query}

I don't have specific materials uploaded about this topic. Would you like me to search the web for information, or would you prefer to upload relevant study materials first?"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are Vidya, a helpful educational AI assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=MAX_RESPONSE_TOKENS,
            temperature=0.7
        )
        
        return response.choices[0].message.content
    
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        return "I apologize, but I encountered an error generating the response. Please try again."

# UI Components
st.title("🧩 Vidya Chatbot")
st.markdown("### AI-Powered Educational Assistant")
st.markdown("*Ask questions about your study materials or any topic!*")

# Sidebar for file upload
with st.sidebar:
    st.header("📚 Upload Study Materials")
    
    uploaded_files = st.file_uploader(
        "Upload PDF or PPTX files",
        type=['pdf', 'pptx'],
        accept_multiple_files=True
    )
    
    url_input = st.text_input("Or enter a URL:")
    
    if st.button("Process Materials", type="primary"):
        all_text = []
        
        # Process uploaded files
        if uploaded_files:
            for file in uploaded_files:
                if file.size > MAX_FILE_SIZE:
                    st.error(f"❌ {file.name} exceeds 500 MB limit")
                    continue
                
                with st.spinner(f"Processing {file.name}..."):
                    if file.name.endswith('.pdf'):
                        text = extract_text_from_pdf(file)
                    elif file.name.endswith('.pptx'):
                        text = extract_text_from_pptx(file)
                    
                    if text:
                        all_text.append(text)
                        st.success(f"✅ {file.name} processed")
        
        # Process URL
        if url_input:
            with st.spinner(f"Fetching {url_input}..."):
                text = extract_text_from_url(url_input)
                if text:
                    all_text.append(text)
                    st.success(f"✅ URL processed")
        
        # Create vector store
        if all_text:
            combined_text = "\n\n".join(all_text)
            chunks = chunk_text(combined_text)
            st.session_state.documents = chunks
            
            index, embeddings = create_vector_store(chunks)
            st.session_state.vector_store = index
            st.session_state.embeddings = embeddings
            
            st.success(f"✨ Created {len(chunks)} searchable chunks!")
        else:
            st.warning("No materials to process")
    
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

# Chat interface
st.markdown("---")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask me anything about your materials..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Try retrieval first
            relevant_chunks = retrieve_relevant_chunks(prompt)
            
            if relevant_chunks:
                response = generate_response(prompt, relevant_chunks)
            else:
                # Fallback to web search
                st.info("🔍 Searching the web for information...")
                web_results = web_search(prompt)
                response = generate_response(prompt, None, use_web_search=True, web_results=web_results)
            
            st.markdown(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "Developed by <strong>Akash Bauri</strong> | "
    "Powered by GPT-4o & RAG Architecture"
    "</div>",
    unsafe_allow_html=True
)
