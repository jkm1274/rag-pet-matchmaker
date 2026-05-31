"""
src/retrieval/retriever.py
Semantic search over ChromaDB or Supabase pgvector with optional
postcode radius filtering.
"""

from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

from typing import List, Optional, Tuple

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from src.ingestion.vector_store import use_supabase


def retrieve_pets(
    query: str,
    vector_store=None,
    top_k: int = 3,
    filter_postcodes: Optional[List[str]] = None,
) -> List[Tuple[Document, float]]:
    """
    Return top_k (Document, similarity_score) tuples for a query.
    Works with both ChromaDB and Supabase pgvector.
    """
    if vector_store is None:
        from src.ingestion.vector_store import get_vector_store
        vector_store = get_vector_store()

    embeddings   = OpenAIEmbeddings(model="text-embedding-3-small")
    query_vector = embeddings.embed_query(query)

    if use_supabase():
        return _retrieve_supabase(query_vector, vector_store, top_k, filter_postcodes)
    else:
        return _retrieve_chroma(query_vector, vector_store, top_k, filter_postcodes)


def _retrieve_supabase(
    query_vector: list,
    vector_store,
    top_k: int,
    filter_postcodes: Optional[List[str]],
) -> List[Tuple[Document, float]]:
    """Retrieve from Supabase pgvector."""
    # Fetch more candidates when filtering
    fetch_k = top_k * 4 if filter_postcodes else top_k

    results = vector_store.similarity_search_by_vector_with_relevance_scores(
        query_vector, k=fetch_k
    )

    if filter_postcodes:
        results = [
            (doc, score) for doc, score in results
            if doc.metadata.get("postcode") in filter_postcodes
        ]

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]


def _retrieve_chroma(
    query_vector: list,
    vector_store,
    top_k: int,
    filter_postcodes: Optional[List[str]],
) -> List[Tuple[Document, float]]:
    """Retrieve from ChromaDB."""
    where = None
    if filter_postcodes:
        if len(filter_postcodes) == 1:
            where = {"postcode": {"$eq": filter_postcodes[0]}}
        else:
            where = {"postcode": {"$in": filter_postcodes}}

    n_results = min(
        top_k * 4 if filter_postcodes else top_k,
        vector_store._collection.count()
    )

    kwargs = {
        "query_embeddings": [query_vector],
        "n_results":        n_results,
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