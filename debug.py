"""
debug.py
Run this from your project root to diagnose each pipeline component.

Usage:
    python debug.py

Each step prints PASS or FAIL with a clear explanation.
"""

import os
import sys

# ── Load .env first, before anything else ─────────────────────────────────────
from dotenv import load_dotenv
load_dotenv()

PASS = "✅ PASS"
FAIL = "❌ FAIL"
INFO = "ℹ️  INFO"
SEP  = "─" * 60


def section(title):
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


# ── Step 1: Environment variables ─────────────────────────────────────────────
section("STEP 1 — Environment variables")

openai_key = os.getenv("OPENAI_API_KEY")
if openai_key and openai_key.startswith("sk-"):
    print(f"{PASS}  OPENAI_API_KEY found: {openai_key[:8]}...{openai_key[-4:]}")
else:
    print(f"{FAIL}  OPENAI_API_KEY missing or invalid.")
    print("       Add it to your .env file or run: export OPENAI_API_KEY=sk-...")
    print("       Cannot continue — exiting.")
    sys.exit(1)

rg_check = os.getenv("RESCUEGROUPS_API_KEY")
if rg_check:
    print(f"{PASS}  RESCUEGROUPS_API_KEY found (Stage 4 ready)")
else:
    print(f"{INFO}  RESCUEGROUPS_API_KEY not set — fine for Stages 1-3")


# ── Step 2: CSV dataset ────────────────────────────────────────────────────────
section("STEP 2 — Mock CSV dataset")

from pathlib import Path
csv_path = Path("data/shelter_pets.csv")

if not csv_path.exists():
    print(f"{FAIL}  data/shelter_pets.csv not found.")
    print("       Run: python data/generate_pets.py")
    sys.exit(1)

import csv
with open(csv_path) as f:
    rows = list(csv.DictReader(f))

print(f"{PASS}  CSV found with {len(rows)} rows")

required_cols = ["id", "name", "species", "breed", "age_years", "energy_level", "bio"]
missing = [c for c in required_cols if c not in rows[0]]
if missing:
    print(f"{FAIL}  Missing columns: {missing}")
    sys.exit(1)
else:
    print(f"{PASS}  All required columns present")
    print(f"{INFO}  Sample pet: {rows[0]['name']} — {rows[0]['breed']}")


# ── Step 3: OpenAI embeddings ──────────────────────────────────────────────────
section("STEP 3 — OpenAI embeddings")

try:
    from langchain_openai import OpenAIEmbeddings
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    test_vector = embeddings.embed_query("test query")
    dim = len(test_vector)
    if dim == 1536:
        print(f"{PASS}  OpenAI embeddings working — dimension: {dim}")
    else:
        print(f"{FAIL}  Unexpected embedding dimension: {dim} (expected 1536)")
        print("       This means a different embedding model is being used.")
        sys.exit(1)
except Exception as e:
    print(f"{FAIL}  OpenAI embeddings failed: {e}")
    print("       Check your API key is valid and has billing enabled.")
    sys.exit(1)


# ── Step 4: ChromaDB index ─────────────────────────────────────────────────────
section("STEP 4 — ChromaDB index")

chroma_path = Path(".chroma_db")
if not chroma_path.exists():
    print(f"{FAIL}  .chroma_db folder not found.")
    print("       Run: python -m src.ingestion.ingest")
    sys.exit(1)

try:
    import chromadb
    client = chromadb.PersistentClient(path=str(chroma_path))
    col = client.get_collection("shelter_pets")
    count = col.count()
    print(f"{PASS}  ChromaDB collection found with {count} documents")

    # Check embedding dimensions match
    sample = col.get(limit=1, include=["embeddings"])
    stored_dim = len(sample["embeddings"][0])
    if stored_dim == 1536:
        print(f"{PASS}  Stored embeddings have correct dimension: {stored_dim}")
    else:
        print(f"{FAIL}  Stored embeddings have wrong dimension: {stored_dim} (expected 1536)")
        print("       Delete .chroma_db and re-run: python -m src.ingestion.ingest")
        sys.exit(1)

except Exception as e:
    print(f"{FAIL}  ChromaDB error: {e}")
    print("       Delete .chroma_db and re-run: python -m src.ingestion.ingest")
    sys.exit(1)


# ── Step 5: Retrieval ──────────────────────────────────────────────────────────
section("STEP 5 — Retrieval (semantic search)")

try:
    from src.ingestion.ingest import load_vector_store
    from src.retrieval.retriever import retrieve_pets

    vs = load_vector_store()
    query = "I live in a small apartment and want a calm low-energy cat"
    results = retrieve_pets(query, vector_store=vs, top_k=3)

    print(f"{PASS}  Retrieval returned {len(results)} results")
    print(f"\n       Query: \"{query}\"\n")

    all_scores_ok = True
    for i, (doc, score) in enumerate(results, 1):
        name     = doc.metadata.get("name", "?")
        energy   = doc.metadata.get("energy_level", "?")
        species  = doc.metadata.get("species", "?")
        score_pct = f"{score*100:.1f}%"
        flag = "✅" if score > 0.50 else "⚠️ "
        print(f"       {flag} #{i}: {name:10} | {species:4} | {energy:9} energy | score={score_pct}")
        if score < 0.50:
            all_scores_ok = False

    if all_scores_ok:
        print(f"\n{PASS}  All similarity scores above 50% — retrieval quality looks good")
    else:
        print(f"\n⚠️   WARNING: Some scores below 50% — retrieval quality may be poor")
        print("       This is normal with only 15 mock pets. Will improve with real data.")

    # Check top result is a low-energy pet
    top_name   = results[0][0].metadata.get("name")
    top_energy = results[0][0].metadata.get("energy_level", "")
    if top_energy in ["Low", "Medium"]:
        print(f"{PASS}  Top result ({top_name}) has appropriate energy level: {top_energy}")
    else:
        print(f"⚠️   Top result ({top_name}) has energy level: {top_energy} — may indicate retrieval issue")

except Exception as e:
    print(f"{FAIL}  Retrieval failed: {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)


# ── Step 6: LLM chain ──────────────────────────────────────────────────────────
section("STEP 6 — LLM recommendation chain")

try:
    from src.retrieval.retriever import format_context
    from src.llm.chain import generate_recommendation

    context = format_context(results)
    recommendation = generate_recommendation(query, context)

    if len(recommendation) > 50:
        print(f"{PASS}  LLM responded ({len(recommendation)} chars)")
        print(f"\n       Preview: {recommendation[:200]}...")
    else:
        print(f"{FAIL}  LLM response too short: '{recommendation}'")

except Exception as e:
    print(f"{FAIL}  LLM chain failed: {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)


# ── All steps passed ───────────────────────────────────────────────────────────
section("RESULT")
print(f"{PASS}  All 6 steps passed — your pipeline is working correctly!")
print(f"       Run: streamlit run app.py\n")


# ── Step 7: RescueGroups API (Stage 4 — only runs if key is set) ──────────────
section("STEP 7 — RescueGroups API connection (Stage 4)")

rg_key = os.getenv("RESCUEGROUPS_API_KEY")
if not rg_key:
    print(f"{INFO}  RESCUEGROUPS_API_KEY not set — skipping (add key to .env when received)")
else:
    try:
        from src.ingestion.rescuegroups_client import RescueGroupsClient
        client = RescueGroupsClient()
        ok = client.test_connection()
        if ok:
            print(f"{PASS}  RescueGroups API key is valid")
            print(f"{INFO}  Fetching first 5 animals near Edison NJ (08817)...")
            count = 0
            for animal in client.fetch_animals_by_location(
                postal_code="08817", distance_miles=25, limit=5, max_pages=1
            ):
                name     = animal.get("animalName", "?")
                species  = animal.get("animalSpecies", "?")
                org      = animal.get("_org", {})
                org_name = org.get("orgName", "—")
                org_city = org.get("orgCity", "—")
                org_phone= org.get("orgPhone", "—")
                print(f"       • {name:15} | {species:6} | {org_name} ({org_city}) {org_phone}")
                count += 1
                if count >= 5:
                    break

            if count > 0:
                print(f"\n{PASS}  Fetched {count} real animals with org data")
                print(f"       Ready to run: python -m src.ingestion.sync")
            else:
                print(f"⚠️   No animals returned — check ZIP code and distance settings")
        else:
            print(f"{FAIL}  RescueGroups API key invalid or connection failed")

    except Exception as e:
        print(f"{FAIL}  RescueGroups test failed: {e}")
        import traceback; traceback.print_exc()