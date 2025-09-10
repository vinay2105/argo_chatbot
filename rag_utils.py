import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from google import genai
from google.genai import types
import os
from sklearn.metrics.pairwise import cosine_similarity
from decouple import config

# ------------------------------
# Load embedding model
# ------------------------------
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ------------------------------
# Load embeddings + metadata
# (pickle file should contain only embeddings + metadata now, no faiss_index)
# ------------------------------
with open("vectorstore.pkl", "rb") as f:
    data = pickle.load(f)

embeddings = np.array(data["embeddings"])
metadata = data["metadata"]

# ------------------------------
# Setup Gemini API
# ------------------------------
api = config('API')
client = genai.Client(api_key = api)

# ------------------------------
# Functions
# ------------------------------
def embed_query(query: str) -> np.ndarray:
    """Generate embedding for query."""
    return embedder.encode([query], convert_to_numpy=True)

def search_index(query: str, top_k: int = 10):
    """Search embeddings using cosine similarity instead of FAISS."""
    query_vec = embed_query(query)
    sims = cosine_similarity(query_vec, embeddings)[0]
    top_idx = sims.argsort()[-top_k:][::-1]
    results = [metadata[i]["text"] for i in top_idx]
    distances = [1 - sims[i] for i in top_idx]  # treat distance as (1 - similarity)
    return results, top_idx, distances

def generate_answer(query: str, top_k: int = 10) -> str:
    """Retrieve top chunks and send to Gemini for final answer."""
    try:
        context_chunks, _, _ = search_index(query, top_k=top_k)
        context_text = "\n".join(context_chunks)

        prompt = f"""You are an assistant answering questions based on oceanographic ARGO float data.

User query: {query}

Context from documents:
{context_text}

Instructions:
- Answer clearly and concisely based on the provided context
- If the context doesn't contain relevant information, state that clearly
- Focus on factual information from the ARGO float data
- Provide specific details when available
- Provide the text in normal form as a single paragraph (do not give '/n' in the response even at the end of the response) 
-don't mention the Float ID in the response

Answer:"""

        response = client.models.generate_content(
            model="gemini-2.0-flash-exp",
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=1024,
                top_p=0.8,
                thinking_config=types.ThinkingConfig(thinking_budget=0)
            )
        )
        return response.text

    except Exception as e:
        return f"❌ Error generating answer: {str(e)}"

def generate_answer_with_sources(query: str, top_k: int = 10) -> dict:
    """Enhanced version that returns both answer and source chunks."""
    try:
        context_chunks, top_idx, distances = search_index(query, top_k=top_k)
        sources = []
        for i, idx in enumerate(top_idx):
            sources.append({
                "text": metadata[idx]["text"],
                "distance": float(distances[i]),
                "metadata": {k: v for k, v in metadata[idx].items() if k != "text"}
            })

        context_text = "\n".join(context_chunks)

        prompt = f"""You are an assistant answering questions based on oceanographic ARGO float data.

User query: {query}

Context from documents:
{context_text}

Instructions:
- Answer clearly and concisely based on the provided context
- If the context doesn't contain relevant information, state that clearly
- Focus on factual information from the ARGO float data
- Provide specific details when available

Answer:"""

        response = client.models.generate_content(
            model="gemini-2.0-flash-exp",
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=1024,
                top_p=0.8,
                thinking_config=types.ThinkingConfig(thinking_budget=0)
            )
        )

        return {
            "answer": response.text,
            "sources": sources,
            "query": query
        }

    except Exception as e:
        return {
            "answer": f"❌ Error generating answer: {str(e)}",
            "sources": [],
            "query": query
        }

def batch_generate_answers(queries: list, top_k: int = 10) -> list:
    """Generate answers for multiple queries efficiently."""
    results = []
    for query in queries:
        try:
            result = generate_answer_with_sources(query, top_k)
            results.append(result)
        except Exception as e:
            results.append({
                "answer": f"❌ Error processing query '{query}': {str(e)}",
                "sources": [],
                "query": query
            })
    return results

# ------------------------------
# Usage Example
# ------------------------------
if __name__ == "__main__":
    test_query = "What is the temperature distribution in the Atlantic Ocean?"

    print("=== Basic Answer ===")
    print(generate_answer(test_query))

    print("\n=== Answer with Sources ===")
    detailed = generate_answer_with_sources(test_query, top_k=5)
    print(f"Query: {detailed['query']}")
    print(f"Answer: {detailed['answer']}")
    print(f"\nTop {len(detailed['sources'])} relevant sources:")
    for i, src in enumerate(detailed['sources'], 1):
        print(f"{i}. Distance: {src['distance']:.4f}")
        print(f"   Text: {src['text'][:200]}...")
        if src['metadata']:
            print(f"   Metadata: {src['metadata']}")
        print()









