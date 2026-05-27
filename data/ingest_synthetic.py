"""
data/ingest_synthetic.py

Loads the synthetic RescueGroups JSON dataset through the real
transformer and ingestion pipeline — exactly as real API data would flow.

Run:
    python data/ingest_synthetic.py

What this does:
    1. Loads data/rg_synthetic.json
    2. Passes each animal through rescuegroups_transformer.animal_to_document()
    3. Embeds and indexes into ChromaDB (same as sync.py does for real data)
    4. Upserts metadata into SQLite (same schema as real data)

When your RescueGroups API key arrives, just run:
    python -m src.ingestion.sync
...and the real data replaces this seamlessly. No other changes needed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

# Add project root to path
sys.path.insert(0, str(Path(__file__).parents[1]))

from src.ingestion.rescuegroups_transformer import animals_to_documents
from src.ingestion.metadata_store import MetadataStore
from src.ingestion.ingest import CHROMA_DIR, COLLECTION_NAME

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings


SYNTHETIC_PATH = Path(__file__).parent / "rg_synthetic.json"


def ingest_synthetic() -> None:
    if not SYNTHETIC_PATH.exists():
        print(f"❌  {SYNTHETIC_PATH} not found.")
        print("    Run first: python data/generate_rg_synthetic.py")
        sys.exit(1)

    print(f"Loading synthetic data from {SYNTHETIC_PATH}…")
    with open(SYNTHETIC_PATH) as f:
        animals = json.load(f)

    print(f"Transforming {len(animals)} animals through RescueGroups transformer…")
    docs = animals_to_documents(animals)
    print(f"  → {len(docs)} valid documents")

    # ── SQLite ────────────────────────────────────────────────────────────
    print("Upserting into SQLite metadata store…")
    store = MetadataStore()
    added = updated = 0
    for animal, doc in zip(animals, docs):
        result = store.upsert_pet(doc.metadata, raw_json=animal)
        if result == "added":   added += 1
        else:                   updated += 1
    store.close()
    print(f"  → added: {added}, updated: {updated}")

    # ── ChromaDB ──────────────────────────────────────────────────────────
    print("Embedding and indexing into ChromaDB…")
    embeddings   = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )

    ids      = [d.metadata["rescuegroups_id"] for d in docs]
    texts    = [d.page_content for d in docs]
    metas    = [d.metadata for d in docs]
    vectors  = embeddings.embed_documents(texts)

    vector_store._collection.upsert(
        ids=ids,
        documents=texts,
        metadatas=metas,
        embeddings=vectors,
    )
    print(f"  → indexed {len(ids)} documents into ChromaDB at {CHROMA_DIR}")

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n✅  Synthetic RescueGroups data ready.")
    print("    Run: streamlit run app.py")
    print("    Run: python debug.py  (to verify all steps pass)")


if __name__ == "__main__":
    ingest_synthetic()