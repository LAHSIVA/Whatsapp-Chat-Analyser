# rag_module.py
import os
import numpy as np
import hashlib
from datetime import datetime
import dateparser
from typing import List, Union

# --- Embedding and similarity imports ---
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors

try:
    import faiss
    _FAISS_AVAILABLE = True
except Exception:
    _FAISS_AVAILABLE = False

from groq import Groq

# -----------------------------------------------------------
# ✅ 1️⃣ Constants and Model Settings
# -----------------------------------------------------------
# Prefer strong embedding model, fallback automatically
PRIMARY_EMBED_MODEL = "intfloat/e5-large-v2"
FALLBACK_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Chunking parameters for larger semantic context
CHUNK_SIZE = 1000
OVERLAP = 200

# -----------------------------------------------------------
# ✅ 2️⃣ Helper Functions
# -----------------------------------------------------------
def normalize_dates(text: str) -> str:
    """Convert relative/natural dates (like 'yesterday') into absolute ISO strings."""
    words = text.split()
    new_words = []
    today = datetime.now()
    for word in words:
        parsed = dateparser.parse(word, settings={"RELATIVE_BASE": today})
        new_words.append(parsed.strftime("%Y-%m-%d") if parsed else word)
    return " ".join(new_words)


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP) -> List[str]:
    """Split text into overlapping semantic chunks for embeddings."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks


def compute_md5(content: str) -> str:
    """Compute MD5 hash for caching."""
    return hashlib.md5(content.encode("utf-8")).hexdigest()

# -----------------------------------------------------------
# ✅ 3️⃣ Retriever Class
# -----------------------------------------------------------
class Retriever:
    def __init__(self, model_name: str = PRIMARY_EMBED_MODEL, cache_dir: str = "embedding_cache"):
        """Initializes retriever with fallback embedding and caching."""
        try:
            self.embedder = SentenceTransformer(model_name)
        except Exception as e:
            print(f"⚠️ Could not load {model_name}. Falling back to {FALLBACK_EMBED_MODEL}. Error: {e}")
            self.embedder = SentenceTransformer(FALLBACK_EMBED_MODEL)

        self.index = None
        self.chunks = []
        self.embeddings = None
        self.cache_dir = cache_dir
        self._use_faiss = _FAISS_AVAILABLE

        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

    def _get_cache_paths(self, file_hash: str):
        emb_path = os.path.join(self.cache_dir, f"{file_hash}_embeddings.npy")
        txt_path = os.path.join(self.cache_dir, f"{file_hash}_chunks.txt")
        return emb_path, txt_path

    def build(self, input_data: Union[str, List[str], List[os.PathLike]]):
        """
        Build or load cached embeddings from text or multiple files.
        Accepts: single string, list of strings, or list of file paths.
        """
        all_text = []

        # 🧩 Read from multiple files if given
        if isinstance(input_data, list):
            for item in input_data:
                if os.path.isfile(item):
                    with open(item, "r", encoding="utf-8") as f:
                        all_text.append(f.read())
                else:
                    all_text.append(str(item))
            combined_text = "\n".join(all_text)
        else:
            combined_text = str(input_data)

        combined_text = normalize_dates(combined_text)
        chunks = chunk_text(combined_text)
        file_hash = compute_md5(combined_text)

        emb_path, txt_path = self._get_cache_paths(file_hash)

        if os.path.exists(emb_path) and os.path.exists(txt_path):
            print("🔄 Using cached embeddings...")
            self.embeddings = np.load(emb_path)
            with open(txt_path, "r", encoding="utf-8") as f:
                self.chunks = f.read().splitlines()
        else:
            print("⚙️ Building new embeddings...")
            self.chunks = chunks
            self.embeddings = np.asarray(
                self.embedder.encode(self.chunks, normalize_embeddings=True, show_progress_bar=True)
            )
            np.save(emb_path, self.embeddings)
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write("\n".join(self.chunks))

        dim = self.embeddings.shape[1]
        if self._use_faiss:
            self.index = faiss.IndexFlatIP(dim)
            self.index.add(self.embeddings)
        else:
            self.index = NearestNeighbors(n_neighbors=10, metric="cosine").fit(self.embeddings)

    def query(self, query_text: str, top_k: int = 10) -> List[str]:
        """Return top_k most semantically similar chunks with date normalization."""
        if not self.index:
            raise ValueError("Index not built. Please call build() first.")

        query_text = normalize_dates(query_text)
        q_emb = np.asarray(self.embedder.encode([query_text], normalize_embeddings=True))

        if self._use_faiss:
            D, I = self.index.search(q_emb, top_k)
            results = [self.chunks[i] for i in I[0] if i < len(self.chunks)]
        else:
            distances, indices = self.index.kneighbors(q_emb, n_neighbors=min(top_k, len(self.chunks)))
            results = [self.chunks[int(idx)] for idx in indices[0]]

        return results

# -----------------------------------------------------------
# ✅ 4️⃣ Groq LLM Integration
# -----------------------------------------------------------
def generate_with_groq(query: str, retrieved_docs: List[str],
                       model: str = "llama-3.3-70b-versatile",
                       temperature: float = 0.2, max_tokens: int = 1024) -> str:
    """Generate response using Groq LLM with extended semantic context."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("❌ GROQ_API_KEY environment variable not set. Please set it before running.")

    client = Groq(api_key=api_key)

    context = "\n\n".join(retrieved_docs)
    # Allow longer context for better semantic recall
    if len(context) > 16000:
        context = context[:16000] + "\n\n[TRUNCATED CONTEXT]"

    messages = [
        {"role": "system", "content": (
            "You are an intelligent assistant analyzing WhatsApp or multi-file chat context. "
            "Always consider semantic meaning, temporal context, and date references accurately."
        )},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer clearly:"},
    ]

    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_completion_tokens=max_tokens,
        top_p=1,
        stream=False,
    )

    return completion.choices[0].message.content.strip()
