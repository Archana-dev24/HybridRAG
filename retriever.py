"""Document chunking, pre-processing, ensemble retrieval, and cross-encoder reranking."""

from __future__ import annotations

import re
import string
from typing import List

import nltk
from langchain.retrievers import EnsembleRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.retrievers import BM25Retriever
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sentence_transformers import CrossEncoder

nltk.download("stopwords", quiet=True)


def chunk(filepath: str) -> list:
    """
    Split a .docx file into chunks.

    Splits first on numbered section headings (e.g. '1.2 Scope'), then further
    splits large sections with RecursiveCharacterTextSplitter.
    """
    loader = Docx2txtLoader(filepath)
    documents = loader.load()
    text = documents[0].page_content

    section_pattern = re.compile(r"(?m)^(\d+(?:\.\d+)*\s+[^\n]+)$")
    matches = list(section_pattern.finditer(text))

    sections: list[str] = []
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections.append(text[start:end].strip())

    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", ", ", " "],
        chunk_size=500,
        chunk_overlap=60,
        length_function=len,
    )
    return splitter.create_documents(sections)


def pre_process(chunks: list) -> list:
    """Lowercase, remove stop-words, punctuation, HTML tags, and stem each chunk."""
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words("english"))
    cleaned: list = []

    for doc in chunks:
        text = doc.page_content
        text = " ".join(w for w in text.split() if w.lower() not in stop_words)
        text = "".join(c for c in text if c not in string.punctuation)
        text = re.sub(r"<.*?>", "", text)
        text = " ".join(stemmer.stem(w) for w in text.lower().split())
        doc.page_content = text
        cleaned.append(doc)

    return cleaned


def build_ensemble_retriever(
    chunks: list,
    vector_store,
    bm25_weight: float,
    vector_weight: float,
    top_k: int,
) -> EnsembleRetriever:
    """Build a BM25 + vector EnsembleRetriever from pre-processed chunks."""
    bm25 = BM25Retriever.from_documents(chunks)
    bm25.k = top_k

    vector = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k},
    )

    return EnsembleRetriever(
        retrievers=[bm25, vector],
        weights=[bm25_weight, vector_weight],
    )


def rerank(question: str, docs: list, top_k: int) -> list:
    """Re-rank retrieved docs with a cross-encoder and return the top_k."""
    model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    pairs = [(question, doc.page_content) for doc in docs]
    scores = model.predict(pairs, batch_size=20)
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in ranked[:top_k]]
