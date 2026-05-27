"""
app.py  —  PawMatch: RAG-Powered Pet Matchmaker
Run:  streamlit run app.py
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
load_dotenv()  # must be before any langchain/openai imports

import streamlit as st
from langchain_chroma import Chroma

from src.ingestion.ingest import build_vector_store, load_pet_documents, load_vector_store
from src.retrieval.retriever import format_context, retrieve_pets
from src.llm.chain import generate_recommendation

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PawMatch · Find Your Perfect Pet",
    page_icon="🐾",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.stTextArea textarea { font-size: 15px; }

.match-card {
    background: #fdf6ee;
    border-left: 4px solid #e07b39;
    border-radius: 10px;
    padding: 1rem 1.25rem;
    margin-bottom: 1rem;
}
.match-card h4 { margin: 0 0 .2rem; color: #c25a1a; font-size: 16px; }
.match-card .meta { font-size: 12px; color: #888; margin-bottom: 8px; }

.badge {
    display: inline-block;
    background: #fde8d4;
    color: #9b4515;
    border-radius: 12px;
    font-size: 11px;
    padding: 2px 9px;
    margin: 2px;
}
.badge-green {
    background: #e6f4ea;
    color: #2d6a4f;
}
.badge-gray {
    background: #f0f0f0;
    color: #666;
}

.rec-box {
    background: #f0f7ff;
    border-left: 4px solid #3b82f6;
    border-radius: 10px;
    padding: 1rem 1.25rem;
    margin-bottom: 1.5rem;
    font-size: 15px;
    line-height: 1.7;
}

.adopt-btn {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    background: #e07b39;
    color: white !important;
    border-radius: 6px;
    padding: 7px 14px;
    font-size: 12px;
    font-weight: 500;
    text-decoration: none !important;
    margin-top: 8px;
}
.adopt-btn:hover { background: #c25a1a; }

.no-link-block {
    background: #f8f6f3;
    border-radius: 8px;
    padding: 10px 14px;
    margin-top: 10px;
    border: 0.5px solid #e8e0d8;
}
.no-link-title {
    font-size: 12px;
    font-weight: 500;
    color: #888;
    margin-bottom: 8px;
}
.next-step {
    display: flex;
    align-items: flex-start;
    gap: 8px;
    font-size: 12px;
    color: #666;
    line-height: 1.5;
    margin-bottom: 5px;
}
.next-step a {
    color: #185FA5;
    text-decoration: none;
}
.next-step a:hover { text-decoration: underline; }
.next-step-icon {
    flex-shrink: 0;
    margin-top: 1px;
    color: #aaa;
}
</style>
""", unsafe_allow_html=True)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _badge(text: str, style: str = "") -> str:
    cls = f"badge {style}".strip()
    return f'<span class="{cls}">{text}</span>'


def _build_badges(m: dict) -> str:
    """Build badge HTML from metadata — works for both mock and RescueGroups data."""
    badges = []

    # Compatibility
    if m.get("good_with_kids") is True:   badges.append(_badge("👶 Kids OK", "badge-green"))
    if m.get("good_with_dogs") is True:   badges.append(_badge("🐕 Dogs OK", "badge-green"))
    if m.get("good_with_cats") is True:   badges.append(_badge("🐈 Cats OK", "badge-green"))
    if m.get("is_seniors_ok") is True:    badges.append(_badge("👴 Seniors OK", "badge-green"))

    # Living situation
    requires_yard = m.get("requires_yard") or m.get("requires_yard")
    if requires_yard is False:            badges.append(_badge("🏢 Apartment OK", "badge-green"))
    if requires_yard is True:             badges.append(_badge("🏡 Needs yard", "badge-gray"))

    # Health
    if m.get("hypoallergenic"):           badges.append(_badge("🌿 Hypoallergenic", "badge-green"))
    if m.get("house_trained") or m.get("is_altered") is False:
        pass  # don't show negative health badges
    if m.get("is_altered") is True:       badges.append(_badge("✂️ Altered", "badge-gray"))
    if m.get("shots_current") is True:    badges.append(_badge("💉 Vaccinated", "badge-gray"))
    if m.get("special_needs") is True:    badges.append(_badge("💊 Special needs", "badge-gray"))

    return " ".join(badges)


def _meta_line(m: dict) -> str:
    """One-line summary — species, age, size, energy, location.
    Breed is omitted here since it already appears in the card title.
    """
    parts = []

    species = m.get("species", "")
    if species and species not in ("Unknown", ""): parts.append(species)

    # Age — RescueGroups uses age_group string, mock uses age_years int
    age = m.get("age_group") or (
        f"{m['age_years']} yr" if m.get("age_years") else ""
    )
    if age: parts.append(age)

    # Size — guard against numeric 0.0 fallback
    size = m.get("size", "")
    try:
        size_valid = size and size not in ("Unknown", "") and float(size) != 0.0
    except (ValueError, TypeError):
        size_valid = bool(size and size not in ("Unknown", ""))
    if size_valid: parts.append(str(size))

    # Energy
    energy = m.get("energy_level", "")
    if energy and energy not in ("Unknown", ""): parts.append(f"{energy} energy")

    # Location
    city  = m.get("city", "")
    state = m.get("state", "")
    if city and state: parts.append(f"{city}, {state}")
    elif city:         parts.append(city)

    return " · ".join(parts)


# ── Vector store (cached across reruns) ───────────────────────────────────────
@st.cache_resource(show_spinner="Loading pet profiles…")
def get_vector_store() -> Chroma:
    if os.path.exists(".chroma_db"):
        return load_vector_store()
    # Fall back to mock data if no index exists
    docs = load_pet_documents()
    return build_vector_store(docs)


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🐾 PawMatch")
    st.caption("RAG-powered pet adoption matching")
    st.divider()

    # Data source indicator
    using_real_data = os.path.exists("data/pets.db")
    if using_real_data:
        try:
            from src.ingestion.metadata_store import MetadataStore
            db = MetadataStore()
            active = len(db.get_active_ids())
            last = db.last_sync_summary()
            db.close()
            st.success(f"✅ Live data: {active} pets")
            if last:
                st.caption(f"Last sync: {last['finished_at'][:10] if last.get('finished_at') else 'unknown'}")
                st.caption(f"Location: {last.get('location', '—')}")
        except Exception:
            st.info("📁 Using live data")
    else:
        st.info("📋 Using mock data (15 pets)\nAdd RescueGroups key to sync real animals.")

    st.divider()
    st.markdown("**How it works**")
    st.caption(
        "1. Your query is converted to an embedding\n"
        "2. ChromaDB finds the closest pet profiles\n"
        "3. GPT-4o-mini generates a personalised recommendation"
    )
    st.divider()
    st.caption(
        "Data from [RescueGroups.org](https://rescuegroups.org) · "
        "Built with LangChain, ChromaDB & OpenAI"
    )


# ── Main UI ────────────────────────────────────────────────────────────────────
st.title("🐾 PawMatch")
st.caption("Describe your lifestyle — find your perfect shelter pet.")

st.divider()

example_prompts = [
    "I live in a small apartment, work 9-5, and want a low-energy companion.",
    "I'm an active runner with a big yard who wants a dog that can keep up.",
    "I have young kids and two cats. I want a calm, friendly dog.",
    "I have mild allergies and work from home most days.",
    "I'm a senior looking for a quiet, gentle companion pet.",
]

with st.expander("✨ Try an example prompt"):
    for ex in example_prompts:
        if st.button(ex, key=ex):
            st.session_state["query"] = ex

query = st.text_area(
    "Describe your lifestyle, living situation, and what you're looking for:",
    value=st.session_state.get("query", ""),
    height=120,
    placeholder="e.g. I live alone in a city apartment, work 9-5, and want a calm low-maintenance cat…",
)

col1, col2 = st.columns([3, 1])
with col1:
    top_k = st.slider("Number of matches", min_value=1, max_value=5, value=3)
with col2:
    st.write("")
    st.write("")
    run = st.button("🔍 Find My Match", type="primary", disabled=not query.strip())

# ── Results ────────────────────────────────────────────────────────────────────
if run and query.strip():
    vector_store = get_vector_store()

    with st.spinner("Searching pet profiles…"):
        results = retrieve_pets(query, vector_store=vector_store, top_k=top_k)
        context = format_context(results)

    with st.spinner("Generating your personalised recommendation…"):
        recommendation = generate_recommendation(query, context)

    st.subheader("🏆 Your Personalised Recommendation")
    st.markdown(f'<div class="rec-box">{recommendation}</div>', unsafe_allow_html=True)

    st.subheader("📋 Matched Pet Profiles")

    for rank, (doc, score) in enumerate(results, start=1):
        import urllib.parse

        m            = doc.metadata
        badges_html  = _build_badges(m)
        meta_line    = _meta_line(m)
        score_pct    = f"{score:.0%}"
        photo_url    = m.get("photo_url", "")
        org_name     = m.get("org_name", "")
        city         = m.get("city", "")
        state        = m.get("state", "")
        pet_name     = m.get("name", "?")
        breed        = m.get("breed", "?")
        location     = ", ".join(filter(None, [city, state]))
        adoption_url = m.get("adoption_url", "") or m.get("org_url", "")
        org_phone    = m.get("org_phone", "")
        org_email    = m.get("org_email", "")

        # ── Build action block based on what data we have ─────────────────
        if adoption_url:
            # State 1 — direct link available
            action_html = f'<div style="margin-top:10px"><a class="adopt-btn" href="{adoption_url}" target="_blank">&#128062; View on RescueGroups</a></div>'

        else:
            # States 2 & 3 — no direct link, build contextual next steps
            steps = []

            if org_name:
                steps.append(
                    f'<div class="next-step"><span class="next-step-icon">&#128222;</span>'
                    f'<span>Call or visit <strong>{org_name}</strong> and ask about {pet_name}'
                    + (f' &mdash; <a href="tel:{org_phone}">{org_phone}</a>' if m.get("org_phone") else "")
                    + '</span></div>'
                )

            google_q   = " ".join(filter(None, [pet_name, breed, org_name or location, "adopt"]))
            google_url = f"https://www.google.com/search?q={urllib.parse.quote(google_q)}"
            steps.append(
                f'<div class="next-step"><span class="next-step-icon">&#128269;</span>'
                f'<span><a href="{google_url}" target="_blank">Search for {pet_name} on Google</a></span></div>'
            )

            rg_location = urllib.parse.quote(m.get("postcode") or "08817")
            rg_species  = (m.get("species") or "cat").lower()
            rg_url      = f"https://rescuegroups.org/adopt/?postalcode={rg_location}&miles=25"
            steps.append(
                f'<div class="next-step"><span class="next-step-icon">&#128196;</span>'
                f'<span><a href="{rg_url}" target="_blank">Browse adoptable pets near {location or "Edison, NJ"} on RescueGroups</a></span></div>'
            )

            if not org_name:
                steps.append(
                    f'<div class="next-step"><span class="next-step-icon">&#128172;</span>'
                    f'<span>Ask a local shelter if they have a {breed} or similar available</span></div>'
                )

            action_html = (
                f'<div class="no-link-block">'
                f'<div class="no-link-title">No direct listing &mdash; how to adopt {pet_name}</div>'
                + "".join(steps)
                + '</div>'
            )

        card_html = f"""
        <div class="match-card">
          <h4>#{rank} &middot; {pet_name} &mdash; {breed}</h4>
          <div class="meta">{meta_line} &middot; Match: {score_pct}</div>
          {badges_html}
          {action_html}
        </div>
        """

        # Layout: photo left, card right (only if photo exists)
        if photo_url:
            img_col, card_col = st.columns([1, 2])
            with img_col:
                st.image(photo_url, use_container_width=True)
            with card_col:
                st.markdown(card_html, unsafe_allow_html=True)
        else:
            st.markdown(card_html, unsafe_allow_html=True)

        # Full bio expander
        with st.expander(f"Read {pet_name}'s full bio"):
            bio_start = doc.page_content.find("Bio: ")
            bio = doc.page_content[bio_start + 5:] if bio_start != -1 else doc.page_content
            st.write(bio)

        st.write("")

st.divider()
st.caption(
    "PawMatch is an AI-powered tool — recommendations are suggestions only. "
    "Always visit the shelter to meet your potential pet. "
    "Pet availability changes daily."
)