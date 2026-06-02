"""
RAG Enterprise Knowledge Base Pipeline
Retrieval-Augmented Generation for enterprise knowledge querying
"""

import json
import logging
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from pathlib import Path

import numpy as np
from openai import AsyncAzureOpenAI

logger = logging.getLogger(__name__)


class DocumentChunker:
    """Intelligent document chunking with context preservation"""
    
    def __init__(self, chunk_size: int = 1024, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk_text(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Split text into semantic chunks"""
        
        chunks = []
        words = text.split()
        
        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk_words = words[i:i + self.chunk_size]
            chunk_text = " ".join(chunk_words)
            
            chunks.append({
                "text": chunk_text,
                "metadata": {
                    **metadata,
                    "chunk_index": len(chunks),
                    "word_count": len(chunk_words)
                }
            })
        
        return chunks


class EmbeddingGenerator:
    """Generate embeddings for documents and queries"""
    
    def __init__(self, client: AsyncAzureOpenAI, deployment_name: str = "text-embedding-3-large"):
        self.client = client
        self.deployment_name = deployment_name
        self.cache = {}
    
    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text"""
        
        # Check cache
        text_hash = hash(text)
        if text_hash in self.cache:
            return self.cache[text_hash]
        
        try:
            response = await self.client.embeddings.create(
                model=self.deployment_name,
                input=text
            )
            
            embedding = response.data[0].embedding
            self.cache[text_hash] = embedding
            return embedding
            
        except Exception as e:
            logger.error(f"Embedding generation failed: {str(e)}")
            raise
    
    async def generate_batch_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts"""
        
        embeddings = []
        for text in texts:
            embedding = await self.generate_embedding(text)
            embeddings.append(embedding)
        
        return embeddings


class VectorStore:
    """Vector database abstraction for similarity search"""
    
    def __init__(self, db_type: str = "pinecone", index_name: str = "enterprise-kb"):
        self.db_type = db_type
        self.index_name = index_name
        self.documents: Dict[str, Dict[str, Any]] = {}
        self.embeddings: Dict[str, List[float]] = {}
    
    def add_documents(self, documents: List[Dict[str, Any]], embeddings: List[List[float]]) -> None:
        """Add documents to vector store"""
        
        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            doc_id = f"doc_{len(self.documents)}_{i}"
            self.documents[doc_id] = doc
            self.embeddings[doc_id] = embedding
            
            logger.info(f"Added document: {doc_id}")
    
    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Tuple[str, float]]:
        """Search for similar documents"""
        
        query_array = np.array(query_embedding)
        results = []
        
        for doc_id, embedding in self.embeddings.items():
            # Cosine similarity
            similarity = np.dot(query_array, embedding) / (
                np.linalg.norm(query_array) * np.linalg.norm(embedding) + 1e-10
            )
            results.append((doc_id, similarity))
        
        # Sort by similarity and return top_k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]


class RAGPipeline:
    """Complete RAG pipeline for enterprise knowledge base"""
    
    def __init__(
        self,
        model: str = "gpt-4",
        vector_db_type: str = "pinecone",
        chunk_size: int = 1024,
        chunk_overlap: int = 200,
        temperature: float = 0.1
    ):
        self.model = model
        self.temperature = temperature
        
        # Initialize components
        self.client = AsyncAzureOpenAI(api_version="2024-02-15-preview")
        self.chunker = DocumentChunker(chunk_size, chunk_overlap)
        self.embedder = EmbeddingGenerator(self.client)
        self.vector_store = VectorStore(db_type=vector_db_type)
        
        self.query_history: List[Dict[str, Any]] = []
    
    async def ingest_documents(
        self,
        documents: List[Dict[str, str]],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Ingest documents into knowledge base"""
        
        start_time = datetime.now()
        chunks = []
        
        # Chunk documents
        for doc in documents:
            doc_chunks = self.chunker.chunk_text(
                doc["content"],
                metadata={
                    "source": doc.get("source", "unknown"),
                    "title": doc.get("title", ""),
                    **(metadata or {})
                }
            )
            chunks.extend(doc_chunks)
        
        # Generate embeddings
        chunk_texts = [chunk["text"] for chunk in chunks]
        embeddings = await self.embedder.generate_batch_embeddings(chunk_texts)
        
        # Add to vector store
        self.vector_store.add_documents(chunks, embeddings)
        
        return {
            "status": "success",
            "documents_ingested": len(documents),
            "chunks_created": len(chunks),
            "duration_ms": (datetime.now() - start_time).total_seconds() * 1000
        }
    
    async def query(
        self,
        question: str,
        top_k: int = 5,
        include_citations: bool = True
    ) -> Dict[str, Any]:
        """Query the knowledge base"""
        
        start_time = datetime.now()
        retrieval_start = datetime.now()
        
        try:
            # Generate query embedding
            query_embedding = await self.embedder.generate_embedding(question)
            
            # Retrieve relevant documents
            search_results = self.vector_store.search(query_embedding, top_k=top_k)
            retrieval_time = (datetime.now() - retrieval_start).total_seconds() * 1000
            
            # Build context from retrieved documents
            context_docs = []
            for doc_id, similarity in search_results:
                doc = self.vector_store.documents[doc_id]
                context_docs.append({
                    "text": doc["text"],
                    "source": doc["metadata"].get("source", "unknown"),
                    "similarity": float(similarity)
                })
            
            # Generate response using LLM
            generation_start = datetime.now()
            
            system_prompt = """You are a helpful AI assistant that answers questions based on provided context.
Always cite your sources and be specific about where information comes from.
If the context doesn't contain relevant information, say so clearly."""
            
            context_text = "\n\n".join([
                f"Source: {doc['source']}\n{doc['text']}"
                for doc in context_docs
            ])
            
            messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": f"Context:\n{context_text}\n\nQuestion: {question}"
                }
            ]
            
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=1000
            )
            
            answer = response.choices[0].message.content
            generation_time = (datetime.now() - generation_start).total_seconds() * 1000
            
            # Calculate confidence
            avg_similarity = np.mean([doc["similarity"] for doc in context_docs])
            confidence = float(avg_similarity)
            
            result = {
                "status": "success",
                "answer": answer,
                "confidence": confidence,
                "sources": context_docs,
                "retrieval_time_ms": retrieval_time,
                "generation_time_ms": generation_time,
                "total_time_ms": (datetime.now() - start_time).total_seconds() * 1000,
                "tokens_used": response.usage.total_tokens if response.usage else 0
            }
            
            # Record query
            self.query_history.append({
                "timestamp": datetime.now().isoformat(),
                "question": question,
                "result": result
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Query failed: {str(e)}")
            return {
                "status": "failed",
                "error": str(e),
                "question": question
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics"""
        
        return {
            "total_documents": len(self.vector_store.documents),
            "total_queries": len(self.query_history),
            "embedding_cache_size": len(self.embedder.cache),
            "avg_query_time_ms": (
                np.mean([q["result"].get("total_time_ms", 0) for q in self.query_history])
                if self.query_history else 0
            )
        }


# Example usage
if __name__ == "__main__":
    import asyncio
    
    async def main():
        # Initialize pipeline
        rag = RAGPipeline(model="gpt-4")
        
        # Sample documents
        documents = [
            {
                "title": "Data Retention Policy",
                "source": "company-policies",
                "content": """Our data retention policy ensures compliance with regulations.
                Customer data is retained for 7 years after account closure.
                Personal identifiable information (PII) is encrypted at rest.
                Backups are maintained for disaster recovery purposes."""
            },
            {
                "title": "Security Guidelines",
                "source": "security-docs",
                "content": """All employees must follow security guidelines.
                Use strong passwords with 12+ characters.
                Enable two-factor authentication on all accounts.
                Report security incidents immediately to the security team."""
            }
        ]
        
        # Ingest documents
        ingest_result = await rag.ingest_documents(documents)
        print(f"Ingestion: {ingest_result}")
        
        # Query knowledge base
        query_result = await rag.query("What is our data retention policy?")
        print(f"\nQuery Result:")
        print(f"Answer: {query_result['answer']}")
        print(f"Confidence: {query_result['confidence']:.2f}")
        print(f"Sources: {[doc['source'] for doc in query_result['sources']]}")
    
    asyncio.run(main())
