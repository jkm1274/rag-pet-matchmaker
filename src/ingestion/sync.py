"""
src/ingestion/sync.py

Multi-anchor sync using the vector store abstraction.
Works with both ChromaDB (local) and Supabase pgvector (cloud).
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Dict, List, Optional, Set

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
from src.ingestion.vector_store import get_vector_store, upsert_documents, delete_documents, use_supabase
from src.utils.location import get_zip_coords, haversine_miles

ALLOWED_STATES = {"NJ", "NY", "PA", "CT", "DE", "MD", "DC"}
MAX_DISTANCE_FROM_ANCHOR_MILES = 60

DEFAULT_ANCHORS = [
    ("08817", "Central NJ — Edison / New Brunswick / Princeton"),
    ("07102", "North NJ  — Newark / Jersey City / Hoboken"),
    ("08096", "South NJ  — Camden / Cherry Hill / Atlantic City"),
    ("11201", "NYC       — Brooklyn / Manhattan / Staten Island"),
    ("10601", "Westchester — White Plains / Yonkers / Long Island"),
    ("19103", "Philly    — Philadelphia / South NJ / Delaware County PA"),
]


def _get_anchors() -> List[tuple]:
    override = os.getenv("SYNC_ANCHORS", "").strip()
    if override:
        return [(z.strip(), f"custom anchor {z.strip()}") for z in override.split(",") if z.strip()]
    return DEFAULT_ANCHORS


def run_sync(
    anchors: Optional[List[tuple]] = None,
    distance: Optional[int] = None,
    max_pages: Optional[int] = None,
) -> None:
    anchors   = anchors   or _get_anchors()
    distance  = distance  or int(os.getenv("SYNC_DISTANCE_MILES", "30"))
    max_pages = max_pages or int(os.getenv("SYNC_MAX_PAGES", "10"))

    backend = "Supabase pgvector" if use_supabase() else "ChromaDB local"
    logger.info("=== PawMatch multi-anchor sync started ===")
    logger.info("Backend: %s | Anchors: %d | Distance: %d mi | Max pages: %d",
                backend, len(anchors), distance, max_pages)

    store        = MetadataStore()
    vector_store = get_vector_store()
    client       = RescueGroupsClient()

    total_fetched = 0
    total_added   = 0
    total_updated = 0
    seen_ids: Set[str] = set()
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

                # State filter
                animal_state = doc.metadata.get("state", "")
                if animal_state and animal_state not in ALLOWED_STATES:
                    continue

                # Distance sanity check
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
                            continue

                pid = doc.metadata["rescuegroups_id"]
                if pid in seen_ids:
                    continue
                seen_ids.add(pid)

                result = store.upsert_pet(doc.metadata, raw_json=animal)
                total_added   += result == "added"
                total_updated += result == "updated"
                docs_to_upsert[pid] = doc

        except Exception as e:
            logger.error("Anchor %s failed: %s", zip_code, e, exc_info=True)
            continue

        logger.info("Anchor %s: %d fetched, %d unique so far",
                    zip_code, anchor_count, len(seen_ids))

    # Batch upsert
    logger.info("Upserting %d unique animals into %s…", len(docs_to_upsert), backend)
    upsert_documents(list(docs_to_upsert.values()), vector_store)

    # Removals
    previously_active: Set[str] = set(store.get_active_ids())
    removed_ids = list(previously_active - seen_ids)
    removed_count = 0
    if removed_ids:
        removed_count = store.mark_inactive(removed_ids)
        delete_documents(removed_ids, vector_store)
        logger.info("%d pets marked inactive.", removed_count)

    logger.info(
        "=== Sync complete — fetched: %d | unique: %d | added: %d | "
        "updated: %d | removed: %d ===",
        total_fetched, len(seen_ids), total_added, total_updated, removed_count,
    )
    store.close()


if __name__ == "__main__":
    run_sync()