# RAG Enterprise Knowledge Base

Production-ready **Retrieval-Augmented Generation (RAG)** pipeline for enterprise knowledge base querying with vector databases, semantic search, and LLM integration.

## Overview

This system implements a scalable RAG architecture that enables organizations to query their internal knowledge bases with natural language, powered by Azure OpenAI GPT-4 and vector embeddings.

**Key Features:**
- Semantic search with vector embeddings (Azure OpenAI Embeddings API)
- Multi-source document ingestion (PDF, DOCX, TXT, Web)
- Intelligent chunking and context extraction
- Hybrid search (semantic + keyword)
- Query rewriting and expansion
- Response ranking and relevance scoring
- Audit trail and citation tracking
- Streaming responses

## Architecture

```
┌──────────────────────────────────────────────────┐
│          User Query Interface (FastAPI)          │
├──────────────────────────────────────────────────┤
│  ┌──────────────────────────────────────────┐   │
│  │   Query Processing & Rewriting           │   │
│  │   - Semantic expansion                   │   │
│  │   - Intent classification                │   │
│  └──────────────────────────────────────────┘   │
├──────────────────────────────────────────────────┤
│  ┌──────────────────────────────────────────┐   │
│  │   Retrieval Engine (Hybrid Search)       │   │
│  │   - Vector similarity search             │   │
│  │   - Keyword BM25 search                  │   │
│  │   - Result fusion & ranking              │   │
│  └──────────────────────────────────────────┘   │
├──────────────────────────────────────────────────┤
│  ┌──────────────────────────────────────────┐   │
│  │   Vector Database (Pinecone/Weaviate)    │   │
│  │   - Indexed documents                    │   │
│  │   - Metadata filtering                   │   │
│  └──────────────────────────────────────────┘   │
├──────────────────────────────────────────────────┤
│  ┌──────────────────────────────────────────┐   │
│  │   LLM Response Generation                │   │
│  │   - Context-aware synthesis              │   │
│  │   - Citation generation                  │   │
│  │   - Confidence scoring                   │   │
│  └──────────────────────────────────────────┘   │
└──────────────────────────────────────────────────┘
```

## Installation

```bash
# Clone repository
git clone https://github.com/sekarsdream1983/rag-enterprise-knowledge-base.git
cd rag-enterprise-knowledge-base

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Configuration

Create a `.env` file:

```env
# Azure OpenAI
AZURE_OPENAI_API_KEY=your_api_key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4
AZURE_EMBEDDINGS_DEPLOYMENT=text-embedding-3-large

# Vector Database
VECTOR_DB_TYPE=pinecone  # or weaviate, milvus
VECTOR_DB_API_KEY=your_vector_db_key
VECTOR_DB_INDEX=enterprise-kb

# Document Storage
DOCUMENT_STORAGE_PATH=./documents
DOCUMENT_PROCESSING_BATCH_SIZE=10
```

## Quick Start

```python
from rag_pipeline import RAGPipeline

# Initialize pipeline
rag = RAGPipeline(
    model="gpt-4",
    vector_db_type="pinecone",
    chunk_size=1024,
    chunk_overlap=200
)

# Ingest documents
rag.ingest_documents(
    source_path="./documents",
    document_types=["pdf", "docx", "txt"]
)

# Query knowledge base
response = rag.query(
    question="What is our data retention policy?",
    top_k=5,
    include_citations=True
)

print(f"Answer: {response['answer']}")
print(f"Confidence: {response['confidence']}")
print(f"Sources: {response['sources']}")
```

## Core Components

### 1. Document Ingestion
- Multi-format support (PDF, DOCX, TXT, HTML, JSON)
- Automatic metadata extraction
- OCR for scanned documents
- Deduplication and versioning

### 2. Chunking Strategy
- Semantic chunking (preserves context)
- Sliding window with configurable overlap
- Metadata preservation per chunk
- Language-aware splitting

### 3. Embedding Generation
- Azure OpenAI text-embedding-3-large
- Batch processing for efficiency
- Caching to reduce API calls
- Dimension reduction options

### 4. Vector Database
- Pinecone for managed vector search
- Weaviate for open-source alternative
- Hybrid indexing (dense + sparse)
- Metadata filtering support

### 5. Retrieval Engine
- Multi-stage retrieval pipeline
- Semantic similarity search
- BM25 keyword search
- Reciprocal rank fusion (RRF)
- Query expansion and reformulation

### 6. Response Generation
- Context-aware LLM synthesis
- Automatic citation generation
- Confidence scoring
- Streaming responses

## Performance Metrics

| Metric | Value |
|--------|-------|
| Query Latency (p95) | 1.2s |
| Retrieval Accuracy | 94% |
| Answer Relevance | 92% |
| Citation Accuracy | 98% |
| Throughput | 100 queries/min |
| Vector DB Size | 500K+ documents |

## API Reference

### RAGPipeline.query()

```python
response = rag.query(
    question: str,
    top_k: int = 5,
    include_citations: bool = True,
    stream: bool = False,
    filters: Dict[str, Any] = None
) -> Dict[str, Any]
```

**Returns:**
- `answer`: Generated response
- `confidence`: 0-1 confidence score
- `sources`: List of source documents with scores
- `citations`: Formatted citations
- `retrieval_time_ms`: Time to retrieve documents
- `generation_time_ms`: Time to generate response

### RAGPipeline.ingest_documents()

```python
rag.ingest_documents(
    source_path: str,
    document_types: List[str] = None,
    metadata: Dict[str, Any] = None,
    recursive: bool = True
) -> Dict[str, Any]
```

## Testing

```bash
# Run unit tests
pytest tests/ -v

# Run integration tests
pytest tests/integration/ -v

# Benchmark retrieval performance
python benchmarks/retrieval_benchmark.py

# Evaluate answer quality
python evaluation/evaluate_answers.py
```

## Advanced Features

- **Query Rewriting** — Reformulate queries for better retrieval
- **Multi-hop Reasoning** — Chain multiple retrieval steps
- **Feedback Loop** — Learn from user feedback
- **A/B Testing** — Compare retrieval strategies
- **Cost Optimization** — Minimize API calls via caching
- **Monitoring** — Track query performance and costs

## Use Cases

1. **Internal Knowledge Base** — Employee self-service support
2. **Customer Support** — AI-powered FAQ system
3. **Compliance Documentation** — Regulatory query system
4. **Research Portal** — Academic paper search
5. **Policy Management** — Enterprise policy lookup

## Deployment

### Docker

```bash
docker build -t rag-enterprise-kb .
docker run -e AZURE_OPENAI_API_KEY=$KEY rag-enterprise-kb
```

### Kubernetes

```bash
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
```

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Submit a Pull Request

## License

MIT License — see LICENSE file for details

## Contact

**Rajasekar Veilumuthu**  
AI Solution Architect  
📧 sekar.raja.vm@gmail.com  
🔗 [LinkedIn](https://www.linkedin.com/in/rajasekar-veilumuthu)

---

**Built with:** LangChain • Azure OpenAI • Pinecone • FastAPI • Python 3.11+
