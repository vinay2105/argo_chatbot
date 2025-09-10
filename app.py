from fastapi import FastAPI
from pydantic import BaseModel
from rag_utils import generate_answer

app = FastAPI(title="RAG Chatbot API", version="1.0")

class QueryRequest(BaseModel):
    query: str

@app.post("/chat")
async def chat(request: QueryRequest):
    """Chat endpoint for RAG chatbot."""
    try:
        answer = generate_answer(request.query)
        return {"query": request.query, "answer": answer}
    except Exception as e:
        return {"error": str(e)}



