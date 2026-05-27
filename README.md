# 🐾 PawMatch — RAG-Powered Pet Matchmaker

A Generative AI portfolio project demonstrating **Retrieval-Augmented Generation (RAG)**
for conversational pet adoption matching.

Instead of filter-based search, users describe their lifestyle in plain English —
*"I live in a small apartment, work 9-5, and want a calm companion"* — and the system
retrieves the most semantically compatible shelter pets and generates a personalised
explanation of why each one is a good match.

> **Note:** Petfinder deprecated their public API in December 2025. This project
> uses the [RescueGroups.org v5 API](https://rescuegroups.org/services/adoptable-pet-data-api/)
> as its live data source, which provides richer pet attributes and covers
> thousands of shelters and rescues across the US.

---

## Architecture

```
User lifestyle query
        │
        ▼
[OpenAI text-embedding-3-small]   query → 1536-dim vector
        │
        ▼
[ChromaDB cosine similarity]      retrieves top-k matching pet profiles
        │
        ▼
[LangChain + GPT-4o-mini]         generates conversational recommendation
        │
        ▼
[Streamlit UI]                    displays matches with photos & adoption links
```

---

## Project Structure

```
rag-pet-matchmaker/
├── app.py                              # Streamlit application
├── debug.py                            # Pipeline diagnostic tool
├── requirements.txt
├── .env.example                        # Environment variable template
├── SETUP_PLAN.md                       # Staged setup & testing guide
│
├── data/
│   ├── generate_pets.py                # Generates 15 mock pets (Stage 1)
│   ├── shelter_pets.csv                # Mock dataset (gitignored after Stage 4)
│   └── pets.db                         # SQLite metadata store (gitignored)
│
├── src/
│   ├── ingestion/
│   │   ├── ingest.py                   # CSV → LangChain Documents → ChromaDB
│   │   ├── rescuegroups_client.py      # RescueGroups v5 API client
│   │   ├── rescuegroups_transformer.py # API response → LangChain Documents
│   │   ├── metadata_store.py           # SQLite layer for sync tracking
│   │   └── sync.py                     # Nightly sync orchestrator
│   ├── retrieval/
│   │   └── retriever.py                # Semantic search + context formatting
│   └── llm/
│       └── chain.py                    # LangChain prompt chain + LLM call
│
└── .github/
    └── workflows/
        └── nightly_sync.yml            # GitHub Actions cron (3 AM nightly)
```

---

## Quick Start

### Prerequisites

- Python 3.11+
- conda (recommended) or venv
- OpenAI API key ([platform.openai.com](https://platform.openai.com))
- RescueGroups API key ([rescuegroups.org](https://rescuegroups.org/services/adoptable-pet-data-api/)) — for live data

### 1. Clone & create environment

```bash
git clone https://github.com/YOUR_USERNAME/rag-pet-matchmaker.git
cd rag-pet-matchmaker

conda create -n pawmatch python=3.11 -y
conda activate pawmatch
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Add your OPENAI_API_KEY (required)
# Add your RESCUEGROUPS_API_KEY (required for live data, optional for mock)
```

### 3. Stage 1 — Mock data (no API keys needed)

```bash
python data/generate_pets.py
```

### 4. Stage 2 — Build vector index (OpenAI key required)

```bash
python -m src.ingestion.ingest
```

### 5. Run the diagnostic tool

```bash
python debug.py    # verifies all pipeline components pass
```

### 6. Launch the app

```bash
streamlit run app.py
```

### 7. Stage 4 — Sync real shelter pets (RescueGroups key required)

```bash
python -m src.ingestion.sync
```

---

## Tech Stack

| Layer | Technology | Notes |
|---|---|---|
| Language | Python 3.11 | |
| RAG Framework | LangChain | LCEL chains |
| Vector Database | ChromaDB | Local persistence |
| Embeddings | OpenAI `text-embedding-3-small` | 1536 dimensions |
| LLM | OpenAI `gpt-4o-mini` | ~$0.002 per query |
| Pet Data | RescueGroups.org v5 API | Free, non-profit |
| Metadata Store | SQLite | Tracks sync state |
| Frontend | Streamlit | |
| Sync Schedule | GitHub Actions | Nightly 3 AM ET |

---

## Cost Estimate

Running this project costs almost nothing:

| Operation | Cost |
|---|---|
| Embedding 500 real pets (once) | ~$0.01 |
| 100 user queries | ~$0.20 |
| Monthly nightly syncs | ~$0.30 |
| **Total/month (light usage)** | **< $1.00** |

---

## Key Design Decisions

**Why RAG instead of plain LLM?** A language model has no knowledge of which
specific pets are currently available at local shelters. RAG grounds the
recommendation in real, current data.

**Why ChromaDB over Pinecone?** For a local-first portfolio project, ChromaDB
requires zero infrastructure — it's a Python package that persists to a local
folder. Migrating to Pinecone or Supabase pgvector later is a one-file swap.

**Why SQLite alongside ChromaDB?** ChromaDB handles vectors well but is awkward
for relational queries (e.g. "which pets were adopted since yesterday?").
SQLite gives a proper audit trail and sync log for free.

**Why RescueGroups over Petfinder?** Petfinder deprecated their public API in
December 2025. RescueGroups is a non-profit with a stable v5 API, richer
pet attributes (`energyLevel`, `activityLevel`, `fenceNeeds`, `ownerExperience`),
and no OAuth complexity — just a simple API key header.

---

## Portfolio Highlights

This project demonstrates end-to-end ML engineering skills relevant to the 2026 job market:

- **RAG pipeline design** — ingestion → embedding → retrieval → generation
- **Real API integration** — RescueGroups v5, pagination, rate limiting, JSON:API parsing
- **Vector database operations** — ChromaDB upsert, cosine similarity, dimension management
- **LangChain LCEL** — composable prompt chains with system/user message structure
- **Data engineering** — dual-store architecture (SQLite + ChromaDB), sync state tracking, removal detection
- **Production patterns** — environment config, modular src layout, cached resources, diagnostic tooling
- **Ecosystem awareness** — tracked Petfinder API deprecation (Dec 2025) and migrated to RescueGroups

---

## Roadmap

### Phase 1 — Deploy & validate (current)
- [ ] Streamlit Community Cloud deployment
- [ ] RescueGroups live data sync (awaiting API key)
- [ ] Expand search radius to 50 miles for broader NJ coverage

### Phase 2 — Shelter partnerships
- [ ] **Shelter partner view** — scoped public URL per shelter (e.g. `/shelter/edison`) showing only that shelter's pets, shareable directly with adopters
- [ ] **ShelterLuv direct integration** — bypass aggregator for shelters we partner with directly; richer data, faster sync, stronger relationship story
- [ ] Shelter outreach kit — demo link + one-pager to send to Middlesex County shelters

### Phase 3 — Priority & urgent pets
- [ ] **"Needs a Home" featured listings** — shelters flag hard-to-place pets (long stay, senior, special needs); these surface with a priority badge and receive a relevance boost in RAG retrieval for compatible queries
- [ ] Long-stay detection — automatically flag pets in SQLite that have been active 60+ days
- [ ] Priority boost in retrieval — weight `priority` and `isNeedingFoster` fields from RescueGroups schema in embedding context

### Phase 4 — Product polish
- [ ] Species and location filter UI controls
- [ ] User feedback button ("Was this match helpful?")
- [ ] FastAPI backend + React frontend
- [ ] Supabase pgvector for persistent cloud vector store
