"""FastAPI server for RAG pipeline"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rag_pipeline import RAGPipeline
import asyncio

app = FastAPI(title="RAG Enterprise KB API")
rag = RAGPipeline(model="gpt-4")

class DocumentInput(BaseModel):
    title: str
    source: str
    content: str

class QueryInput(BaseModel):
    question: str
    top_k: int = 5

@app.post("/ingest")
async def ingest_documents(docs: list[DocumentInput]):
    """Ingest documents into knowledge base"""
    documents = [doc.dict() for doc in docs]
    result = await rag.ingest_documents(documents)
    return result

@app.post("/query")
async def query_kb(query: QueryInput):
    """Query the knowledge base"""
    result = await rag.query(query.question, top_k=query.top_k)
    return result

@app.get("/stats")
async def get_stats():
    """Get pipeline statistics"""
    return rag.get_stats()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
