"""
Hybrid RAG pipeline: BM25 + dense vector retrieval with cross-encoder reranking.

Two separate lifecycle stages:
  ingest  -- chunk a .docx, embed with Nomic, persist to ChromaDB (one-time)
  query   -- load persisted store, retrieve via BM25+vector ensemble, rerank, generate

CLI usage:
    python pipeline.py ingest --filepath doc.docx
    python pipeline.py query "what are the different tray types?"

Programmatic usage:
    from pipeline import HybridRAGPipeline
    p = HybridRAGPipeline()
    p.ingest("doc.docx")
    print(p.query("what are the tray types?"))

Environment variables (or .env file):
    GROQ_API_KEY  -- required for the query stage
"""

from __future__ import annotations

import os
import pickle

from dotenv import load_dotenv
from langchain_chroma import Chroma

from embedder import NomicEmbeddingFunction
from llm import GroqSummarizer
from retriever import build_ensemble_retriever, chunk, pre_process, rerank

load_dotenv()

_CHUNKS_FILENAME = "bm25_chunks.pkl"


class HybridRAGPipeline:
    """
    Two-stage hybrid RAG pipeline.

    Stage 1 -- ingest (one-time per document):
        pipeline = HybridRAGPipeline()
        pipeline.ingest("doc.docx")

    Stage 2 -- query (repeatable, no re-ingestion needed):
        pipeline = HybridRAGPipeline()
        result = pipeline.query("your question")
        print(result["response"])
    """

    def __init__(
        self,
        persist_directory: str = "store/chroma",
        collection_name: str = "hybrid_rag_kb",
        top_k_retrieve: int = 5,
        top_k_rerank: int = 3,
        bm25_weight: float = 0.4,
        vector_weight: float = 0.6,
    ) -> None:
        self._persist_dir = persist_directory
        self._collection_name = collection_name
        self._top_k_retrieve = top_k_retrieve
        self._top_k_rerank = top_k_rerank
        self._bm25_weight = bm25_weight
        self._vector_weight = vector_weight
        self._chunks_path = os.path.join(persist_directory, _CHUNKS_FILENAME)

        self._embedding_fn: NomicEmbeddingFunction | None = None
        self._vector_store: Chroma | None = None
        self._chunks: list | None = None

    # ------------------------------------------------------------------
    # Stage 1: ingest
    # ------------------------------------------------------------------

    def ingest(self, filepath: str, document_type: str = "generic") -> None:
        """
        Chunk, embed, and persist a .docx document.

        Chunks are also pickled alongside the vector store so BM25 can be
        rebuilt at query time without re-reading the original file.
        """
        print(f"[ingest 1/3] Chunking: {filepath}")
        chunks = chunk(filepath)
        cleaned_chunks = pre_process(chunks)
        source_name = os.path.basename(filepath)

        for i, doc in enumerate(cleaned_chunks):
            doc.metadata["source"] = source_name
            doc.metadata["chunk_id"] = i
            doc.metadata["document_type"] = document_type

        print(f"[ingest 2/3] Embedding {len(cleaned_chunks)} chunks -> ChromaDB")
        os.makedirs(self._persist_dir, exist_ok=True)
        vector_store = Chroma(
            collection_name=self._collection_name,
            embedding_function=self._get_embedding_fn(),
            collection_metadata={"hnsw:space": "cosine"},
            persist_directory=self._persist_dir,
        )
        vector_store.add_documents(cleaned_chunks)
        self._vector_store = vector_store

        print(f"[ingest 3/3] Saving chunks for BM25 -> {self._chunks_path}")
        with open(self._chunks_path, "wb") as f:
            pickle.dump(cleaned_chunks, f)
        self._chunks = cleaned_chunks

        print(f"Done. {len(cleaned_chunks)} chunks stored in '{self._persist_dir}'.")

    # ------------------------------------------------------------------
    # Stage 2: query
    # ------------------------------------------------------------------

    def query(self, question: str) -> dict:
        """
        Answer a question against the persisted vector store.

        Loads the store and BM25 chunks from disk on first call, then reuses
        them for subsequent queries in the same session.
        """
        self._ensure_loaded()

        print("[query 1/3] Retrieving with BM25 + vector ensemble")
        retriever = build_ensemble_retriever(
            self._chunks,
            self._vector_store,
            self._bm25_weight,
            self._vector_weight,
            self._top_k_retrieve,
        )
        raw_docs = retriever.invoke(question)
        #deduplication
        seen: set[str] = set()
        unique_docs = []
        for doc in raw_docs:
            if doc.page_content not in seen:
                unique_docs.append(doc)
                seen.add(doc.page_content)

        print(f"[query 2/3] Reranking {len(unique_docs)} docs -> top {self._top_k_rerank}")
        top_docs = rerank(question, unique_docs, self._top_k_rerank)

        context = "\n\n".join(doc.page_content for doc in top_docs)

        print("[query 3/3] Generating answer with Groq")
        return GroqSummarizer().summarize(context, question)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_embedding_fn(self) -> NomicEmbeddingFunction:
        if self._embedding_fn is None:
            self._embedding_fn = NomicEmbeddingFunction()
        return self._embedding_fn

    def _ensure_loaded(self) -> None:
        if self._vector_store is None:
            self._vector_store = Chroma(
                collection_name=self._collection_name,
                embedding_function=self._get_embedding_fn(),
                collection_metadata={"hnsw:space": "cosine"},
                persist_directory=self._persist_dir,
            )

        if self._chunks is None:
            if not os.path.exists(self._chunks_path):
                raise FileNotFoundError(
                    f"No ingested chunks found at '{self._chunks_path}'. "
                    "Run ingest first:\n"
                    "  python pipeline.py ingest --filepath <doc.docx>"
                )
            with open(self._chunks_path, "rb") as f:
                self._chunks = pickle.load(f)


def main() -> None:
    pipeline = HybridRAGPipeline()

    while True:
        print("\nAvailable Operations")
        print("--------------------")
        print("1. ingest")
        print("2. query")
        print("3. exit")

        choice = input("\nSelect operation (1/2/3): ").strip()

        if choice in ("3", "exit"):
            break
        elif choice in ("1", "ingest"):
            filepath = input("Enter path to .docx file: ").strip()
            pipeline.ingest(filepath)
        elif choice in ("2", "query"):
            question = input("Enter your question: ").strip()
            result = pipeline.query(question)
            print("\nAnswer:", result.get("response", result))
        else:
            print("Invalid option. Enter 1, 2, or 3.")


if __name__ == "__main__":
    main()
