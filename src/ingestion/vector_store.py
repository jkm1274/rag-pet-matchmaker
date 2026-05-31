"""
src/ingestion/vector_store.py

Vector store abstraction — switches between ChromaDB (local dev)
and Supabase pgvector (cloud/production) based on environment variables.

If SUPABASE_URL and SUPABASE_KEY are set → use Supabase (persistent, cloud)
Otherwise → use ChromaDB (local file, dev only)

This means:
  - Local development: zero config, works as before
  - Cloud deployment: persistent index that survives redeploys
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

from langchain_openai import OpenAIEmbeddings

logger = logging.getLogger(__name__)

COLLECTION_NAME = "documents"   # Supabase table name


def _embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model="text-embedding-3-small")


def use_supabase() -> bool:
    """Return True if Supabase credentials are configured."""
    return bool(os.getenv("SUPABASE_URL")) and bool(os.getenv("SUPABASE_KEY"))


def get_vector_store_supabase():
    """Return a LangChain SupabaseVectorStore instance."""
    from langchain_community.vectorstores import SupabaseVectorStore
    from supabase import create_client

    url = os.environ["SUPABASE_URL"]
    key = os.environ["SUPABASE_KEY"]

    client = create_client(url, key)
    return SupabaseVectorStore(
        client=client,
        embedding=_embeddings(),
        table_name="documents",
        query_name="match_documents",
    )


def get_vector_store_chroma(persist_dir: str = ".chroma_db"):
    """Return a LangChain Chroma instance."""
    from langchain_chroma import Chroma
    return Chroma(
        collection_name="shelter_pets",
        embedding_function=_embeddings(),
        persist_directory=persist_dir,
    )


def get_vector_store():
    """
    Return the appropriate vector store based on environment.
    Supabase if credentials present, ChromaDB otherwise.
    """
    if use_supabase():
        logger.info("Using Supabase pgvector store")
        return get_vector_store_supabase()
    else:
        logger.info("Using ChromaDB local store")
        return get_vector_store_chroma()


def upsert_documents(docs: list, vector_store=None) -> None:
    """
    Upsert LangChain Documents into the vector store.
    Uses rescuegroups_id as the stable dedup key.
    """
    if vector_store is None:
        vector_store = get_vector_store()

    if use_supabase():
        _upsert_supabase(docs, vector_store)
    else:
        _upsert_chroma(docs, vector_store)


def _supabase_request(method: str, path: str, data=None) -> dict:
    """Make a direct HTTP request to Supabase REST API, bypassing client URL issues."""
    import requests as req
    url = os.environ["SUPABASE_URL"].rstrip("/")
    key = os.environ["SUPABASE_KEY"]
    # Ensure we use the correct REST path
    if "/rest/v1" not in url:
        full_url = f"{url}/rest/v1/{path}"
    else:
        full_url = f"{url}/{path}"

    headers = {
        "apikey":        key,
        "Authorization": f"Bearer {key}",
        "Content-Type":  "application/json",
        "Prefer":        "resolution=merge-duplicates,return=minimal",
    }
    resp = req.request(method, full_url, json=data, headers=headers, timeout=30)
    if not resp.ok:
        logger.error("Supabase %s %s → %d: %s", method, full_url, resp.status_code, resp.text[:200])
        resp.raise_for_status()
    return resp


def _upsert_supabase(docs: list, vector_store) -> None:
    """Upsert into Supabase using direct HTTP calls to avoid client URL bugs."""
    batch_size       = 100
    texts            = [d.page_content for d in docs]
    metas            = [d.metadata for d in docs]
    embeddings_model = _embeddings()

    for i in range(0, len(docs), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_metas = metas[i:i + batch_size]
        vectors     = embeddings_model.embed_documents(batch_texts)

        rows = []
        for text, meta, vec in zip(batch_texts, batch_metas, vectors):
            rows.append({
                "content":         text,
                "metadata":        meta,
                "embedding":       vec,
                "rescuegroups_id": meta.get("rescuegroups_id", ""),
            })

        _supabase_request(
            "POST",
            "documents?on_conflict=rescuegroups_id",
            data=rows,
        )
        logger.info("Upserted batch of %d into Supabase", len(rows))


def _upsert_chroma(docs: list, vector_store) -> None:
    """Upsert into ChromaDB using rescuegroups_id as stable key."""
    ids     = [d.metadata.get("rescuegroups_id", "") for d in docs]
    texts   = [d.page_content for d in docs]
    metas   = [d.metadata for d in docs]
    vectors = _embeddings().embed_documents(texts)

    vector_store._collection.upsert(
        ids=ids,
        documents=texts,
        metadatas=metas,
        embeddings=vectors,
    )
    logger.info("Upserted %d documents into ChromaDB", len(docs))


def delete_documents(rescuegroups_ids: list, vector_store=None) -> None:
    """Remove documents by rescuegroups_id from the vector store."""
    if not rescuegroups_ids:
        return

    if vector_store is None:
        vector_store = get_vector_store()

    if use_supabase():
        for i in range(0, len(rescuegroups_ids), 100):
            batch = rescuegroups_ids[i:i+100]
            ids_str = ",".join(batch)
            _supabase_request(
                "DELETE",
                f"documents?rescuegroups_id=in.({ids_str})",
            )
        logger.info("Deleted %d docs from Supabase", len(rescuegroups_ids))
    else:
        vector_store._collection.delete(ids=rescuegroups_ids)
        logger.info("Deleted %d docs from ChromaDB", len(rescuegroups_ids))