WhatsApp Chat Analyzer + RAG (Groq)
==================================

Prereqs:
- Python 3.9+
- Recommended: create a virtualenv

Install:
pip install -r requirements.txt

Set Groq API key (Linux/macOS):
export GROQ_API_KEY="sk-xxxx"

(Windows PowerShell)
$Env:GROQ_API_KEY="sk-xxxx"

Run:
streamlit run app.py

Notes:
- The RAG module builds embeddings from `cleaned_messages`.
- If faiss fails to install on your environment, code will fall back to sklearn's NearestNeighbors.
- Groq Cloud generation uses the endpoint `https://api.groq.com/openai/v1/chat/completions`.
- Keep retrieved context small to avoid context-window limits; code truncates context at ~4000 chars.
