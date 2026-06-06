"""Embedding wrapper: Nomic sentence-transformer model for ChromaDB and LangChain."""

from __future__ import annotations

from typing import List

from chromadb.api.types import EmbeddingFunction
from sentence_transformers import SentenceTransformer


class NomicEmbeddingFunction(EmbeddingFunction):
    """Wraps nomic-ai/nomic-embed-text-v1 for use with ChromaDB and LangChain."""

    def __init__(self, model_name: str = "nomic-ai/nomic-embed-text-v1") -> None:
        self._model = SentenceTransformer(model_name, trust_remote_code=True)

    def __call__(self, input: List[str]) -> List[List[float]]:
        return self._model.encode(input).tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._model.encode(texts, show_progress_bar=False).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self._model.encode([text], convert_to_numpy=True)[0].tolist()
