# 🧩 Vidya Chatbot - AI-Powered Educational Assistant

An intelligent study assistant that lets you chat with your study materials (PDFs, PowerPoint, URLs) using GPT-4o and RAG architecture.

**Developed by Akash Bauri**

---

## 🚀 Quick Start Guide

### 1️⃣ Clone or Create Repository

```bash
# Create new repository on GitHub named "vidya-chatbot"
# Then clone it:
git clone https://github.com/YOUR_USERNAME/vidya-chatbot.git
cd vidya-chatbot
```

### 2️⃣ Add Project Files

Copy these files to your repository:
- `app.py` (main application)
- `requirements.txt` (dependencies)
- `README.md` (this file)
- `.gitignore` (security)

### 3️⃣ Push to GitHub

```bash
git add .
git commit -m "Initial commit - Vidya Chatbot"
git push origin main
```

---

## 🌐 Deploy on Streamlit Cloud

### Step 1: Go to Streamlit Cloud
Visit: [share.streamlit.io](https://share.streamlit.io)

### Step 2: Connect GitHub
- Click "New app"
- Select your GitHub repository: `vidya-chatbot`
- Main file path: `app.py`

### Step 3: Add Secrets (CRITICAL!)
Click "Advanced settings" → "Secrets"

Add this:
```toml
OPENAI_API_KEY = "sk-...1PQA"
```

### Step 4: Deploy!
Click "Deploy" and wait 2-3 minutes.

---

## 📋 Features

✅ Upload PDFs and PowerPoint files (up to 500 MB)  
✅ Extract text from web URLs  
✅ RAG-powered responses with citations  
✅ Automatic web search fallback  
✅ Session chat memory  
✅ Export chat history as JSON  
✅ Grade-5 level clarity  

---

## 🛠️ Local Development (Optional)

```bash
# Install dependencies
pip install -r requirements.txt

# Create .env file with your API key
echo 'OPENAI_API_KEY=sk-...1PQA' > .env

# Run app
streamlit run app.py
```

---

## 🏗️ Architecture

```
User Upload (PDF/PPTX/URL)
        ↓
Text Extraction & Chunking
        ↓
Sentence-Transformer Embeddings
        ↓
FAISS Vector Store
        ↓
Query → Retrieve Top-K Chunks
        ↓
GPT-4o Response with Citations
        ↓
(Fallback: DuckDuckGo Web Search)
```

---

## 📊 Performance Specs

| Metric | Target |
|--------|--------|
| Upload Processing | < 15s |
| Retrieval Latency | < 3s |
| Total Response Time | < 10s |
| Accuracy | ≥ 90% |

---

## 🔒 Security

- API keys stored in Streamlit Secrets
- `.env` excluded via `.gitignore`
- No plaintext keys in code
- Session data cleared on reload

---

## 📞 Support

**Developer**: Akash Bauri  
**Issues**: Open a GitHub issue in this repository

---

## 📄 License

MIT License - Free for educational use

---

**🎓 Empowering learning through intelligent, explainable AI.**
