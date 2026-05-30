"""
src/ingestion/sync.py

Multi-anchor sync — fetches adoptable animals from multiple ZIP code
anchors covering NJ, NYC, and Philadelphia, deduplicates by
rescuegroups_id, then embeds and indexes into ChromaDB.

Run manually:
    python -m src.ingestion.sync

Run on a schedule (GitHub Actions, cron):
    0 3 * * * cd /app && python -m src.ingestion.sync >> logs/sync.log 2>&1

Environment variables (see .env.example):
    RESCUEGROUPS_API_KEY
    OPENAI_API_KEY
    SYNC_DISTANCE_MILES   (default: 30)
    SYNC_MAX_PAGES        (default: 10, 100 animals/page)
    SYNC_ANCHORS          (optional override, comma-separated ZIPs)
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
from src.utils.location import get_zip_coords, haversine_miles

# States we want in the index — tri-state + DC
ALLOWED_STATES = {"NJ", "NY", "PA", "CT", "DE", "MD", "DC"}

# Hard distance cap from the nearest anchor — animals further than this
# are almost certainly misconfigured in RescueGroups
MAX_DISTANCE_FROM_ANCHOR_MILES = 60

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings


# ── Default anchor ZIP codes — tri-state coverage ─────────────────────────────
DEFAULT_ANCHORS = [
    ("08817", "Central NJ — Edison / New Brunswick / Princeton"),
    ("07102", "North NJ  — Newark / Jersey City / Hoboken"),
    ("08096", "South NJ  — Camden / Cherry Hill / Atlantic City"),
    ("11201", "NYC       — Brooklyn / Manhattan / Staten Island"),
    ("10601", "Westchester — White Plains / Yonkers / Long Island"),
    ("19103", "Philly    — Philadelphia / South NJ / Delaware County PA"),
]


def _get_anchors() -> List[tuple]:
    """Return anchor list — respects SYNC_ANCHORS env override."""
    override = os.getenv("SYNC_ANCHORS", "").strip()
    if override:
        return [(z.strip(), f"custom anchor {z.strip()}") for z in override.split(",") if z.strip()]
    return DEFAULT_ANCHORS


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
    ids     = list(docs_by_id.keys())
    docs    = list(docs_by_id.values())
    texts   = [d.page_content for d in docs]
    metas   = [d.metadata for d in docs]
    vectors = vector_store._embedding_function.embed_documents(texts)
    vector_store._collection.upsert(
        ids=ids, documents=texts, metadatas=metas, embeddings=vectors,
    )
    logger.info("Upserted %d documents into ChromaDB.", len(ids))


def _chroma_delete(vector_store: Chroma, ids: List[str]) -> None:
    if not ids:
        return
    vector_store._collection.delete(ids=ids)
    logger.info("Deleted %d inactive pets from ChromaDB.", len(ids))


def run_sync(
    anchors: List[tuple] | None = None,
    distance: int | None = None,
    max_pages: int | None = None,
) -> None:
    anchors   = anchors   or _get_anchors()
    distance  = distance  or int(os.getenv("SYNC_DISTANCE_MILES", "30"))
    max_pages = max_pages or int(os.getenv("SYNC_MAX_PAGES", "10"))

    logger.info("=== PawMatch multi-anchor sync started ===")
    logger.info(
        "Anchors: %d | Distance: %d mi | Max pages/anchor: %d",
        len(anchors), distance, max_pages,
    )

    store        = MetadataStore()
    vector_store = _get_or_create_vector_store()
    client       = RescueGroupsClient()

    total_fetched = 0
    total_added   = 0
    total_updated = 0
    seen_ids: Set[str] = set()           # global dedup across all anchors
    docs_to_upsert: Dict[str, object] = {}

    for zip_code, description in anchors:
        logger.info("── Anchor: %s (%s)", zip_code, description)
        anchor_count = 0

        try:
            for animal in client.fetch_animals_by_location(
                postal_code=zip_code,
                distance_miles=distance,
                max_pages=max_pages,
            ):
                total_fetched += 1
                anchor_count  += 1

                doc = animal_to_document(animal)
                if doc is None:
                    continue

                pid = doc.metadata["rescuegroups_id"]

                # ── Global deduplication ───────────────────────────────────
                if pid in seen_ids:
                    continue   # already queued from a previous anchor
                seen_ids.add(pid)

                # ── State filter — skip out-of-region animals ──────────
                animal_state = doc.metadata.get("state", "")
                if animal_state and animal_state not in ALLOWED_STATES:
                    logger.debug(
                        "Skipping %s — state %s not in allowed list",
                        doc.metadata.get("name"), animal_state,
                    )
                    continue

                # ── Distance sanity check ──────────────────────────────────
                animal_zip = doc.metadata.get("postcode", "")
                if animal_zip:
                    anchor_coords = get_zip_coords(zip_code)
                    animal_coords = get_zip_coords(animal_zip)
                    if anchor_coords and animal_coords:
                        dist = haversine_miles(
                            anchor_coords[0], anchor_coords[1],
                            animal_coords[0], animal_coords[1],
                        )
                        if dist > MAX_DISTANCE_FROM_ANCHOR_MILES:
                            logger.debug(
                                "Skipping %s (ZIP %s) — %.0f mi from anchor %s",
                                doc.metadata.get("name"), animal_zip, dist, zip_code,
                            )
                            continue

                result = store.upsert_pet(doc.metadata, raw_json=animal)
                if result == "added":   total_added   += 1
                else:                   total_updated += 1

                docs_to_upsert[pid] = doc

        except Exception as e:
            logger.error("Anchor %s failed: %s", zip_code, e, exc_info=True)
            continue

        logger.info("Anchor %s: %d fetched, %d unique so far", zip_code, anchor_count, len(seen_ids))

    # ── Batch embed and upsert all unique docs ─────────────────────────────
    logger.info("Embedding %d unique animals across all anchors…", len(docs_to_upsert))
    _chroma_upsert(vector_store, docs_to_upsert)

    # ── Detect removals (animals no longer in any anchor's results) ────────
    previously_active: Set[str] = set(store.get_active_ids())
    removed_ids = list(previously_active - seen_ids)
    removed_count = 0

    if removed_ids:
        removed_count = store.mark_inactive(removed_ids)
        _chroma_delete(vector_store, removed_ids)
        logger.info("%d pets marked inactive (adopted or removed).", removed_count)

    logger.info(
        "=== Sync complete — fetched: %d | unique: %d | added: %d | "
        "updated: %d | removed: %d ===",
        total_fetched, len(seen_ids), total_added, total_updated, removed_count,
    )


if __name__ == "__main__":
    run_sync()