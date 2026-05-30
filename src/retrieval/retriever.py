"""
src/retrieval/retriever.py
Semantic search over ChromaDB with optional postcode radius filtering.
"""

from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

from typing import List, Optional, Tuple

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from src.ingestion.ingest import load_vector_store


def retrieve_pets(
    query: str,
    vector_store: Chroma | None = None,
    top_k: int = 3,
    filter_postcodes: Optional[List[str]] = None,
) -> List[Tuple[Document, float]]:
    """
    Return top_k (Document, similarity_score) tuples for a query.

    Args:
        query:            User lifestyle description.
        vector_store:     Pre-loaded Chroma instance (lazy-loaded if None).
        top_k:            Number of results to return.
        filter_postcodes: If provided, only consider animals whose postcode
                          is in this list. Used for location filtering.
    """
    if vector_store is None:
        vector_store = load_vector_store()

    embeddings   = OpenAIEmbeddings(model="text-embedding-3-small")
    query_vector = embeddings.embed_query(query)

    # Build ChromaDB where filter if postcodes specified
    where = None
    if filter_postcodes:
        if len(filter_postcodes) == 1:
            where = {"postcode": {"$eq": filter_postcodes[0]}}
        else:
            where = {"postcode": {"$in": filter_postcodes}}

    # Fetch more candidates when filtering so top_k results survive
    n_results = top_k * 4 if filter_postcodes else top_k

    kwargs = {
        "query_embeddings": [query_vector],
        "n_results":        min(n_results, vector_store._collection.count()),
        "include":          ["documents", "metadatas", "distances"],
    }
    if where:
        kwargs["where"] = where

    raw = vector_store._collection.query(**kwargs)

    results: List[Tuple[Document, float]] = []
    for doc_text, metadata, distance in zip(
        raw["documents"][0],
        raw["metadatas"][0],
        raw["distances"][0],
    ):
        similarity = round(1 - (distance / 2), 4)
        results.append((Document(page_content=doc_text, metadata=metadata), similarity))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]


def format_context(results: List[Tuple[Document, float]]) -> str:
    """Serialise retrieved docs into a context block for the LLM."""
    sections: List[str] = []

    for rank, (doc, score) in enumerate(results, start=1):
        m   = doc.metadata
        age = (
            m.get("age_group")
            or (f"{m['age_years']} yr" if m.get("age_years") else "Unknown")
        )
        org      = m.get("org_name", "")
        location = ", ".join(filter(None, [m.get("city"), m.get("state")]))

        section = (
            f"--- Match #{rank} (similarity: {score:.2f}) ---\n"
            f"Name: {m.get('name')}  |  {m.get('species')} — {m.get('breed')}\n"
            f"Age: {age}  |  Size: {m.get('size')}  |  Energy: {m.get('energy_level')}\n"
            f"Good with kids: {m.get('good_with_kids')}  |  "
            f"Good with dogs: {m.get('good_with_dogs')}  |  "
            f"Good with cats: {m.get('good_with_cats')}\n"
            f"Requires yard: {m.get('requires_yard')}  |  "
            f"Special needs: {m.get('special_needs')}\n"
        )
        if org or location:
            section += f"Shelter: {org}  |  Location: {location}\n"
        section += f"\n{doc.page_content}"
        sections.append(section)

    return "\n\n".join(sections)