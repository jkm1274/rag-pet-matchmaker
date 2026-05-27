"""
src/ingestion/sync.py

Orchestrates a full RescueGroups sync cycle:
  1. Fetch adoptable pets near a postal code
  2. Upsert raw records into SQLite (metadata_store)
  3. Embed new/changed records and upsert into ChromaDB
  4. Mark pets that disappeared from the API as inactive in both stores
  5. Write a sync log entry

Run manually:
    python -m src.ingestion.sync

Environment variables required (see .env.example):
    RESCUEGROUPS_API_KEY
    OPENAI_API_KEY
    SYNC_LOCATION        (default: "08817")  Edison NJ ZIP
    SYNC_DISTANCE_MILES  (default: "25")
    SYNC_MAX_PAGES       (default: "10")     10 pages x 100 = 1,000 pets
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Dict, List, Set

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("sync")

from src.ingestion.rescuegroups_client import RescueGroupsClient
from src.ingestion.rescuegroups_transformer import animal_to_document
from src.ingestion.metadata_store import MetadataStore
from src.ingestion.ingest import CHROMA_DIR, COLLECTION_NAME

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings


def _get_or_create_vector_store() -> Chroma:
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )


def _chroma_upsert(vector_store: Chroma, docs_by_id: Dict) -> None:
    if not docs_by_id:
        return
    ids      = list(docs_by_id.keys())
    docs     = list(docs_by_id.values())
    texts    = [d.page_content for d in docs]
    metas    = [d.metadata for d in docs]
    vectors  = vector_store._embedding_function.embed_documents(texts)

    vector_store._collection.upsert(
        ids=ids,
        documents=texts,
        metadatas=metas,
        embeddings=vectors,
    )
    logger.info("Upserted %d documents into ChromaDB.", len(ids))


def _chroma_delete(vector_store: Chroma, ids: List[str]) -> None:
    if not ids:
        return
    vector_store._collection.delete(ids=ids)
    logger.info("Deleted %d inactive pets from ChromaDB.", len(ids))


def run_sync(
    location: str | None = None,
    distance: int | None = None,
    max_pages: int | None = None,
) -> None:
    location  = location  or os.getenv("SYNC_LOCATION", "08817")
    distance  = distance  or int(os.getenv("SYNC_DISTANCE_MILES", "25"))
    max_pages = max_pages or int(os.getenv("SYNC_MAX_PAGES", "10"))

    logger.info("=== PawMatch sync started ===")
    logger.info("Location: %s | Distance: %s mi | Max pages: %s", location, distance, max_pages)

    store  = MetadataStore()
    log_id = store.start_sync(location)

    fetched = added = updated = removed_count = 0
    error_msg = None
    seen_ids: Set[str] = set()

    try:
        client       = RescueGroupsClient()
        vector_store = _get_or_create_vector_store()
        docs_to_upsert: Dict = {}

        for animal in client.fetch_animals_by_location(
            postal_code=location,
            distance_miles=distance,
            max_pages=max_pages,
        ):
            fetched += 1
            doc = animal_to_document(animal)
            if doc is None:
                continue

            pid = doc.metadata["rescuegroups_id"]
            seen_ids.add(pid)

            result = store.upsert_pet(doc.metadata, raw_json=animal)
            added   += result == "added"
            updated += result == "updated"

            docs_to_upsert[pid] = doc

        _chroma_upsert(vector_store, docs_to_upsert)

        previously_active: Set[str] = set(store.get_active_ids())
        removed_ids = list(previously_active - seen_ids)
        if removed_ids:
            removed_count = store.mark_inactive(removed_ids)
            _chroma_delete(vector_store, removed_ids)
            logger.info("%d pets marked inactive.", removed_count)

        logger.info(
            "Sync complete — fetched: %d | added: %d | updated: %d | removed: %d",
            fetched, added, updated, removed_count,
        )

    except Exception as exc:
        error_msg = str(exc)
        logger.error("Sync failed: %s", exc, exc_info=True)

    finally:
        store.finish_sync(
            log_id=log_id,
            fetched=fetched,
            added=added,
            updated=updated,
            removed=removed_count,
            error=error_msg,
        )
        store.close()

    logger.info("=== PawMatch sync finished ===")


if __name__ == "__main__":
    run_sync()