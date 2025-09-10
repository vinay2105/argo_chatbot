import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from google import genai
from google.genai import types
import os

# Load embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Load vectorstore
with open("vectorstore.pkl", "rb") as f:
    data = pickle.load(f)

embeddings = data["embeddings"]
metadata = data["metadata"]
index = data["faiss_index"]

# Setup Gemini API (replace with your key in env var)
client = genai.Client(api_key=os.getenv("API"))

def embed_query(query: str) -> np.ndarray:
    """Generate embedding for query."""
    return embedder.encode([query], convert_to_numpy=True)

def search_index(query: str, top_k: int = 10):
    """Search FAISS index for top-k relevant chunks."""
    query_vec = embed_query(query)
    distances, indices = index.search(query_vec, top_k)
    results = [metadata[i]["text"] for i in indices[0]]
    return results

def generate_answer(query: str) -> str:
    """Retrieve top chunks and send to Gemini for final answer."""
    try:
        # Step 1: Get top chunks
        context_chunks = search_index(query, top_k=10)
        context_text = "\n".join(context_chunks)

        # Step 2: Construct prompt
        prompt = f"""
        You are an assistant answering questions based on oceanographic ARGO float data.
        User query: {query}
        Context from documents:
        {context_text}

        Answer in a clear and concise way, only using context if possible.
        """

        # Step 3: Call Gemini
        response = client.models.generate_content(
                model="gemini-2.5-flash", contents=prompt,
        )

        return response.text

    except Exception as e:
        return f"❌ Error: {str(e)}"





