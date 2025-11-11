# rag_module.py
import os
import numpy as np
from sentence_transformers import SentenceTransformer

try:
    import faiss
    _FAISS_AVAILABLE = True
except Exception:
    _FAISS_AVAILABLE = False
    from sklearn.neighbors import NearestNeighbors

from groq import Groq  # ✅ Groq SDK

# ✅ Use a valid embedding model from SentenceTransformers
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

class Retriever:
    def __init__(self, model_name=EMBED_MODEL_NAME):
        self.embedder = SentenceTransformer(model_name)
        self.index = None
        self.messages = []
        self.embeddings = None
        self._use_faiss = _FAISS_AVAILABLE

    def build(self, messages):
        """Build embedding index from list of strings."""
        self.messages = [str(m) for m in messages]
        self.embeddings = np.asarray(self.embedder.encode(self.messages, convert_to_numpy=True))
        dim = self.embeddings.shape[1]
        if self._use_faiss:
            self.index = faiss.IndexFlatL2(dim)
            self.index.add(self.embeddings)
        else:
            self.index = NearestNeighbors(n_neighbors=5, metric='cosine').fit(self.embeddings)

    def query(self, text, top_k=5):
        """Return top_k most similar messages for the text."""
        q_emb = np.asarray(self.embedder.encode([text], convert_to_numpy=True))
        if self._use_faiss:
            D, I = self.index.search(q_emb, top_k)
            results = [self.messages[i] for i in I[0] if i < len(self.messages)]
        else:
            distances, indices = self.index.kneighbors(q_emb, n_neighbors=min(top_k, len(self.messages)))
            results = [self.messages[int(idx)] for idx in indices[0]]
        return results


def generate_with_groq(query, retrieved_docs, model="llama-3.3-70b-versatile", temperature=0.2, max_tokens=512):
    """
    Generate answer using Groq Cloud LLaMA model.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("❌ GROQ_API_KEY environment variable not set. Please set it before running.")

    client = Groq(api_key=api_key)

    context = "\n\n".join(retrieved_docs)
    if len(context) > 4000:
        context = context[:4000] + "\n\n[TRUNCATED CONTEXT]"

    messages = [
        {"role": "system", "content": "You are an intelligent assistant that answers based on provided WhatsApp chat context."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"}
    ]

    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_completion_tokens=max_tokens,
        top_p=1,
        stream=False
    )

    return completion.choices[0].message.content.strip()
