# HybridRAG

A two-stage hybrid retrieval-augmented generation (RAG) pipeline combining BM25 sparse retrieval, dense vector search (Nomic embeddings + ChromaDB), cross-encoder reranking, and Groq-hosted LLM generation.

## Architecture

```
query
  │
  ▼
┌──────────────────────────────────────────────────┐
│  Retriever (retriever.py)                        │
│  BM25Retriever + ChromaDB vector search          │
│  → EnsembleRetriever (weighted fusion)           │
│  → CrossEncoder reranking                        │
└─────────────────────┬────────────────────────────┘
                      │ top-k docs
                      ▼
┌──────────────────────────────────────────────────┐
│  LLM (llm.py)                                    │
│  GroqSummarizer — llama-3.3-70b-versatile        │
│  JSON-structured answer generation               │
└──────────────────────────────────────────────────┘
```

## Project Structure

| File | Responsibility |
|---|---|
| `embedder.py` | `NomicEmbeddingFunction` — wraps `nomic-ai/nomic-embed-text-v1` for ChromaDB and LangChain |
| `retriever.py` | `chunk`, `pre_process`, `build_ensemble_retriever`, `rerank` — all retrieval logic |
| `llm.py` | `GroqSummarizer` — Groq API client for answer generation |
| `pipeline.py` | `HybridRAGPipeline` — orchestrates ingest and query stages; CLI entry point |
| `rag_hybrid.py` | Thin backward-compatible shim; imports from `pipeline.py` |

## Setup

### 1. Install dependencies

```bash
pip install langchain langchain-chroma langchain-community chromadb \
            sentence-transformers groq python-docx docx2txt nltk python-dotenv
```

### 2. Configure environment

Create a `.env` file in this directory:

```
GROQ_API_KEY=your_groq_api_key_here
```

## Usage

### Interactive CLI

```bash
python pipeline.py
```

Select `1` to ingest a document, `2` to query, `3` to exit.

### Programmatic

```python
from pipeline import HybridRAGPipeline

pipeline = HybridRAGPipeline()

# One-time ingest
pipeline.ingest("path/to/document.docx")

# Query (repeatable, no re-ingestion needed)
result = pipeline.query("What are the different tray types?")
print(result["response"])
```

### Custom configuration

```python
pipeline = HybridRAGPipeline(
    persist_directory="store/chroma",  # ChromaDB storage path
    collection_name="my_collection",
    top_k_retrieve=5,                  # docs fetched by BM25 + vector each
    top_k_rerank=3,                    # docs kept after cross-encoder reranking
    bm25_weight=0.4,                   # weight for BM25 in ensemble
    vector_weight=0.6,                 # weight for vector search in ensemble
)
```

## Pipeline Stages

### Ingest

1. **Chunk** — splits `.docx` on numbered section headings, then applies `RecursiveCharacterTextSplitter` (500 chars, 60 overlap).
2. **Pre-process** — removes stop-words, punctuation, HTML tags; applies Porter stemming.
3. **Embed** — encodes chunks with `nomic-ai/nomic-embed-text-v1` via `NomicEmbeddingFunction`.
4. **Persist** — stores vectors in ChromaDB; pickles raw chunks for BM25 reconstruction.

### Query

1. **Ensemble retrieval** — BM25 (weight 0.4) + cosine vector search (weight 0.6), top-5 each.
2. **Deduplication** — removes exact-duplicate page content.
3. **Reranking** — `cross-encoder/ms-marco-MiniLM-L-6-v2` scores each (query, doc) pair; top-3 kept.
4. **Generation** — Groq `llama-3.3-70b-versatile` generates a JSON-structured answer from the context.

## Notes

- ChromaDB data is persisted in `store/chroma/` by default; delete this folder to re-ingest from scratch.
- BM25 chunks are saved alongside the vector store as `store/chroma/bm25_chunks.pkl`.
- The LLM is instructed to return `{"response": "..."}` and will not fabricate facts outside the retrieved context.
