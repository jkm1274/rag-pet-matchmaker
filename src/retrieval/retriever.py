"""
src/retrieval/retriever.py
Converts a user's natural-language query into an embedding and retrieves
the top-k most semantically similar pet profiles from ChromaDB.
"""

from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

from typing import List, Tuple

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from src.ingestion.ingest import load_vector_store


def retrieve_pets(
    query: str,
    vector_store: Chroma | None = None,
    top_k: int = 3,
) -> List[Tuple[Document, float]]:
    """Return top_k (Document, similarity_score) tuples for a query.

    Args:
        query:        The user's lifestyle description.
        vector_store: Pre-loaded Chroma instance (lazy-loaded if None).
        top_k:        Number of candidates to return.

    Returns:
        List of (Document, score) tuples sorted best-first.
        Score is cosine similarity 0.0-1.0 — higher is better.
    """
    if vector_store is None:
        vector_store = load_vector_store()

    # Embed the query with OpenAI explicitly — do NOT pass query_texts
    # to the collection directly, as that uses ChromaDB's default local
    # model (384 dims) instead of OpenAI (1536 dims).
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    query_vector = embeddings.embed_query(query)

    # Query ChromaDB with the pre-computed vector
    raw = vector_store._collection.query(
        query_embeddings=[query_vector],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    # ChromaDB cosine distance: 0 = identical, 2 = opposite
    # Convert to similarity: 1 - (distance / 2)
    results: List[Tuple[Document, float]] = []
    for doc_text, metadata, distance in zip(
        raw["documents"][0],
        raw["metadatas"][0],
        raw["distances"][0],
    ):
        similarity = round(1 - (distance / 2), 4)
        results.append((Document(page_content=doc_text, metadata=metadata), similarity))

    results.sort(key=lambda x: x[1], reverse=True)
    return results


def format_context(results: List[Tuple[Document, float]]) -> str:
    """Serialise retrieved docs into a clean context block for the LLM.

    Works with both mock CSV schema (age_years) and RescueGroups schema (age_group).
    Uses .get() throughout so missing fields never cause KeyErrors.
    """
    sections: List[str] = []

    for rank, (doc, score) in enumerate(results, start=1):
        m = doc.metadata

        # Handle both mock (age_years) and RescueGroups (age_group) schemas
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