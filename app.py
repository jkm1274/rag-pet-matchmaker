"""
app.py  —  PawMatch: RAG-Powered Pet Matchmaker
Run:  streamlit run app.py
"""

from __future__ import annotations

import io
import json
import os
import urllib.parse
from urllib.parse import urlparse

from dotenv import load_dotenv
load_dotenv()

import streamlit as st

# ── Bridge Streamlit Cloud secrets → environment variables ────────────────────
# On Streamlit Community Cloud, secrets are in st.secrets, not os.environ.
# We copy them into os.environ so all downstream code (OpenAI, RescueGroups)
# picks them up regardless of whether we're running locally or on the cloud.
def _load_streamlit_secrets() -> None:
    try:
        for key in ["OPENAI_API_KEY", "RESCUEGROUPS_API_KEY",
                    "SYNC_DISTANCE_MILES", "SYNC_MAX_PAGES"]:
            if key in st.secrets and not os.environ.get(key):
                os.environ[key] = str(st.secrets[key])
    except Exception:
        pass  # st.secrets not available locally — that's fine, .env handles it

_load_streamlit_secrets()
from langchain_chroma import Chroma

from src.ingestion.ingest import build_vector_store, load_pet_documents, load_vector_store
from src.utils.location import (
    filter_postcodes_by_radius, get_zip_coords, NJ_ZIP_SUGGESTIONS
)
from src.retrieval.retriever import format_context, retrieve_pets
from src.llm.chain import generate_recommendation

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PawMatch · Find Your Perfect Pet",
    page_icon="🐾",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ── Session state init ─────────────────────────────────────────────────────────
if "favorites" not in st.session_state:
    st.session_state["favorites"] = {}
if "results" not in st.session_state:
    st.session_state["results"] = None
if "recommendation" not in st.session_state:
    st.session_state["recommendation"] = None
if "last_search_zip" not in st.session_state:
    st.session_state["last_search_zip"] = None
if "last_radius" not in st.session_state:
    st.session_state["last_radius"] = None
if "show_more" not in st.session_state:
    st.session_state["show_more"] = False
if "all_results" not in st.session_state:
    st.session_state["all_results"] = None
if "filters" not in st.session_state:
    st.session_state["filters"] = {}


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
.badge-green { background: #e6f4ea; color: #2d6a4f; }
.badge-gray  { background: #f0f0f0; color: #666; }

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
.next-step a { color: #185FA5; text-decoration: none; }
.next-step a:hover { text-decoration: underline; }
.next-step-icon { flex-shrink: 0; margin-top: 1px; color: #aaa; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _badge(text: str, style: str = "") -> str:
    return f'<span class="badge {style}".strip()>{text}</span>'


def _build_badges(m: dict) -> str:
    badges = []
    if m.get("good_with_kids") is True:  badges.append(_badge("👶 Kids OK", "badge-green"))
    if m.get("good_with_dogs") is True:  badges.append(_badge("🐕 Dogs OK", "badge-green"))
    if m.get("good_with_cats") is True:  badges.append(_badge("🐈 Cats OK", "badge-green"))
    if m.get("is_seniors_ok") is True:   badges.append(_badge("👴 Seniors OK", "badge-green"))
    requires_yard = m.get("requires_yard")
    if requires_yard is False:           badges.append(_badge("🏢 Apartment OK", "badge-green"))
    if requires_yard is True:            badges.append(_badge("🏡 Needs yard", "badge-gray"))
    if m.get("hypoallergenic"):          badges.append(_badge("🌿 Hypoallergenic", "badge-green"))
    if m.get("is_altered") is True:      badges.append(_badge("✂️ Altered", "badge-gray"))
    if m.get("shots_current") is True:   badges.append(_badge("💉 Vaccinated", "badge-gray"))
    if m.get("special_needs") is True:   badges.append(_badge("💊 Special needs", "badge-gray"))
    return " ".join(badges)


def _bio_quality(page_content: str) -> str:
    """Return a warning badge HTML if the bio is suspiciously short."""
    bio_start = page_content.find("Bio: ")
    if bio_start == -1:
        return '<span class="badge badge-gray">⚠️ Limited info</span>'
    bio = page_content[bio_start + 5:].strip()
    # A synthetic bio built purely from structured fields is typically < 200 chars
    if len(bio) < 200:
        return '<span class="badge badge-gray">⚠️ Limited info — call shelter</span>'
    return ""


def _meta_line(m: dict) -> str:
    parts = []
    species = m.get("species", "")
    if species and species not in ("Unknown", ""): parts.append(species)
    age = m.get("age_group") or (f"{m['age_years']} yr" if m.get("age_years") else "")
    if age: parts.append(age)
    size = m.get("size", "")
    try:
        size_valid = size and size not in ("Unknown", "") and float(size) != 0.0
    except (ValueError, TypeError):
        size_valid = bool(size and size not in ("Unknown", ""))
    if size_valid: parts.append(str(size))
    energy = m.get("energy_level", "")
    if energy and energy not in ("Unknown", ""): parts.append(f"{energy} energy")
    city, state = m.get("city", ""), m.get("state", "")
    if city and state: parts.append(f"{city}, {state}")
    elif city:         parts.append(city)
    return " · ".join(parts)


def _validate_adoption_url(m: dict) -> str:
    """Return a valid RescueGroups adoption URL, reconstructing if misconfigured."""
    url = m.get("adoption_url", "")
    if not url:
        return ""
    if "rescuegroups.org" not in url:
        return ""
    if "AnimalID=" in url:
        url_animal_id = url.split("AnimalID=")[-1].strip()
        org_id = m.get("org_id", "")
        rg_id  = m.get("rescuegroups_id", "")
        if url_animal_id == org_id and url_animal_id != rg_id and rg_id:
            parsed = urlparse(url)
            return f"{parsed.scheme}://{parsed.netloc}/animals/detail?AnimalID={rg_id}"
    return url


def _build_action_html(m: dict, adoption_url: str) -> str:
    """Build the action block HTML for a pet card."""
    if adoption_url:
        return (
            f'<div style="margin-top:10px">'
            f'<a class="adopt-btn" href="{adoption_url}" target="_blank">'
            f'&#128062; View on RescueGroups</a></div>'
        )

    pet_name = m.get("name", "?")
    breed    = m.get("breed", "?")
    org_name = m.get("org_name", "")
    org_phone= m.get("org_phone", "")
    org_fb   = m.get("org_url", "")
    city     = m.get("city", "")
    state    = m.get("state", "")
    location = ", ".join(filter(None, [city, state]))
    postcode = m.get("postcode") or "08817"

    steps = []

    if org_name:
        phone_html = (
            f' &mdash; <a href="tel:{org_phone}">{org_phone}</a>'
            if org_phone else ""
        )
        steps.append(
            f'<div class="next-step"><span class="next-step-icon">&#128222;</span>'
            f'<span>Call or visit <strong>{org_name}</strong> and ask about {pet_name}'
            f'{phone_html}</span></div>'
        )

    if org_fb and "facebook.com" in org_fb:
        steps.append(
            f'<div class="next-step"><span class="next-step-icon">&#128279;</span>'
            f'<span><a href="{org_fb}" target="_blank">'
            f'Find {org_name or "this shelter"} on Facebook</a></span></div>'
        )
    elif org_fb:
        steps.append(
            f'<div class="next-step"><span class="next-step-icon">&#127968;</span>'
            f'<span><a href="{org_fb}" target="_blank">'
            f'Visit {org_name or "shelter"} website</a></span></div>'
        )

    google_q   = " ".join(filter(None, [pet_name, breed, org_name or location, "adopt"]))
    google_url = f"https://www.google.com/search?q={urllib.parse.quote(google_q)}"
    steps.append(
        f'<div class="next-step"><span class="next-step-icon">&#128269;</span>'
        f'<span><a href="{google_url}" target="_blank">Search for {pet_name} on Google</a></span></div>'
    )

    # Search Google for adoptable pets near the shelter location
    area_q   = " ".join(filter(None, [location or "NJ", "animal shelter adopt pet"]))
    area_url = f"https://www.google.com/search?q={urllib.parse.quote(area_q)}"
    steps.append(
        f'<div class="next-step"><span class="next-step-icon">&#128196;</span>'
        f'<span><a href="{area_url}" target="_blank">'
        f'Find animal shelters near {location or "your area"} on Google</a></span></div>'
    )

    if not org_name:
        steps.append(
            f'<div class="next-step"><span class="next-step-icon">&#128172;</span>'
            f'<span>Ask a local shelter if they have a {breed} or similar available</span></div>'
        )

    return (
        f'<div class="no-link-block">'
        f'<div class="no-link-title">No direct listing &mdash; how to adopt {pet_name}</div>'
        + "".join(steps)
        + '</div>'
    )


# ── Favorites helpers ──────────────────────────────────────────────────────────

def _favorite_key(m: dict) -> str:
    return m.get("rescuegroups_id") or m.get("name", "unknown")


def _toggle_favorite(key: str, m: dict) -> None:
    if key in st.session_state["favorites"]:
        del st.session_state["favorites"][key]
    else:
        st.session_state["favorites"][key] = m


def _build_favorites_csv() -> str:
    """Generate a CSV string from saved favorites."""
    favs = list(st.session_state["favorites"].values())
    if not favs:
        return ""
    cols = ["name", "species", "breed", "age_group", "size", "energy_level",
            "city", "state", "org_name", "org_phone", "org_email",
            "adoption_url", "photo_url", "good_with_kids", "good_with_dogs",
            "good_with_cats", "requires_yard", "special_needs"]
    lines = [",".join(cols)]
    for m in favs:
        row = [str(m.get(c, "")).replace(",", ";").replace("\n", " ") for c in cols]
        lines.append(",".join(row))
    return "\n".join(lines)


def _build_favorites_text() -> str:
    """Generate a readable text summary of saved favorites."""
    favs = list(st.session_state["favorites"].values())
    if not favs:
        return ""
    lines = ["🐾 PawMatch — My Saved Pets\n" + "=" * 40]
    for i, m in enumerate(favs, 1):
        name     = m.get("name", "?")
        breed    = m.get("breed", "?")
        species  = m.get("species", "?")
        age      = m.get("age_group", "?")
        location = ", ".join(filter(None, [m.get("city"), m.get("state")]))
        org      = m.get("org_name", "")
        phone    = m.get("org_phone", "")
        url      = _validate_adoption_url(m) or m.get("org_url", "")

        lines.append(f"\n{i}. {name} — {breed}")
        lines.append(f"   {species} · {age} · {location}")
        if org:   lines.append(f"   Shelter: {org}")
        if phone: lines.append(f"   Phone:   {phone}")
        if url:   lines.append(f"   Link:    {url}")
    return "\n".join(lines)


# ── Vector store ───────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading pet profiles…")
def get_vector_store() -> Chroma:
    """
    Load or build the vector store.

    On Streamlit Community Cloud the filesystem is ephemeral — .chroma_db
    resets on every deploy. If both API keys are present we run a live sync
    on cold start (takes ~2 min). Otherwise we fall back to the mock CSV
    so the app is always usable even without API keys.
    """
    if os.path.exists(".chroma_db"):
        return load_vector_store()

    has_rg_key = bool(os.getenv("RESCUEGROUPS_API_KEY"))
    has_oai_key = bool(os.getenv("OPENAI_API_KEY"))

    # Debug — show what keys are available (remove after confirming working)
    st.info(
        f"🔑 Key check — RG: {'✅' if has_rg_key else '❌'} | "
        f"OAI: {'✅' if has_oai_key else '❌'} | "
        f"Secrets available: {list(st.secrets.keys()) if hasattr(st, 'secrets') else 'n/a'}"
    )

    if has_rg_key and has_oai_key:
        # Cold start on cloud — run a fast single-anchor sync
        st.info(
            "🔄 First run detected — syncing shelter data. "
            "This takes about 2 minutes and only happens once per deploy."
        )
        try:
            from src.ingestion.sync import run_sync
            run_sync(
                anchors=[("08817", "Central NJ — cold start")],
                distance=30,
                max_pages=5,
            )
            return load_vector_store()
        except Exception as e:
            st.warning(f"Live sync failed: {e}")

    # Fallback: mock CSV
    docs = load_pet_documents()
    return build_vector_store(docs)


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🐾 PawMatch")
    st.caption("RAG-powered pet adoption matching")
    st.divider()

    # Data source indicator
    if os.path.exists("data/pets.db"):
        try:
            from src.ingestion.metadata_store import MetadataStore
            db = MetadataStore()
            active = len(db.get_active_ids())
            last   = db.last_sync_summary()
            db.close()
            st.success(f"✅ Live data: {active} pets")
            if last:
                st.caption(f"Last sync: {last['finished_at'][:10] if last.get('finished_at') else 'unknown'}")
                st.caption(f"Location: {last.get('location', '—')}")
        except Exception:
            st.info("📁 Using live data")
    elif bool(os.getenv("RESCUEGROUPS_API_KEY")) and bool(os.getenv("OPENAI_API_KEY")):
        st.info("🔄 Live data ready — will sync on first search.")
    else:
        st.info("📋 Using mock data (15 pets)\nAdd RescueGroups key to sync real animals.")

    st.divider()

    # ── Favorites panel ────────────────────────────────────────────────────────
    favs = st.session_state["favorites"]
    n    = len(favs)

    st.markdown(f"### ⭐ Saved Pets ({n})")

    if n == 0:
        st.caption("Star a pet to save it here during your session.")
    else:
        for fav_key, fav_m in list(favs.items()):
            fav_name = fav_m.get("name", "?")
            fav_loc  = ", ".join(filter(None, [fav_m.get("city"), fav_m.get("state")]))
            col_a, col_b = st.columns([4, 1])
            with col_a:
                st.caption(f"**{fav_name}** · {fav_loc}")
            with col_b:
                if st.button("✕", key=f"remove_{fav_key}", help="Remove"):
                    del st.session_state["favorites"][fav_key]
                    st.rerun()

        st.divider()

        # Individual downloads
        st.caption("**Download individual profiles:**")
        for fav_key, fav_m in favs.items():
            fav_name    = fav_m.get("name", "pet")
            fav_url     = _validate_adoption_url(fav_m) or fav_m.get("org_url", "")
            fav_org     = fav_m.get("org_name", "")
            fav_phone   = fav_m.get("org_phone", "")
            fav_species = fav_m.get("species", "")
            fav_breed   = fav_m.get("breed", "")
            fav_age     = fav_m.get("age_group", "")
            fav_loc     = ", ".join(filter(None, [fav_m.get("city"), fav_m.get("state")]))

            profile = (
                f"🐾 {fav_name} — {fav_breed}\n"
                f"{'─' * 30}\n"
                f"Species:  {fav_species}\n"
                f"Age:      {fav_age}\n"
                f"Size:     {fav_m.get('size', '—')}\n"
                f"Energy:   {fav_m.get('energy_level', '—')}\n"
                f"Location: {fav_loc}\n"
            )
            if fav_org:   profile += f"Shelter:  {fav_org}\n"
            if fav_phone: profile += f"Phone:    {fav_phone}\n"
            if fav_url:   profile += f"Link:     {fav_url}\n"

            st.download_button(
                label=f"⬇ {fav_name}",
                data=profile,
                file_name=f"pawmatch_{fav_name.lower().replace(' ', '_')}.txt",
                mime="text/plain",
                key=f"dl_{fav_key}",
            )

        st.divider()

        # Batch downloads
        st.caption("**Download all saved pets:**")
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="⬇ CSV",
                data=_build_favorites_csv(),
                file_name="pawmatch_favorites.csv",
                mime="text/csv",
                key="dl_all_csv",
            )
        with col2:
            st.download_button(
                label="⬇ Text",
                data=_build_favorites_text(),
                file_name="pawmatch_favorites.txt",
                mime="text/plain",
                key="dl_all_txt",
            )

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

# ── Location controls ─────────────────────────────────────────────────────────
# Defaults — overridden inside the expander if user changes them
search_zip   = st.session_state.get("search_zip_ui", "08817")
radius_miles = st.session_state.get("radius_miles_ui", 25)

with st.expander("📍 Search location & radius", expanded=False):
    loc_col1, loc_col2 = st.columns([2, 1])
    with loc_col1:
        # Selectbox with NJ suggestions + custom option
        location_options = ["📍 Use my custom ZIP"] + list(NJ_ZIP_SUGGESTIONS.keys())
        location_choice  = st.selectbox("Select a location", location_options, index=1)

        if location_choice == "📍 Use my custom ZIP":
            custom_zip = st.text_input(
                "Enter ZIP code", value="08817",
                max_chars=5, placeholder="e.g. 07030"
            ).strip()
            search_zip = custom_zip if len(custom_zip) == 5 and custom_zip.isdigit() else "08817"
        else:
            search_zip = NJ_ZIP_SUGGESTIONS[location_choice]

        # Validate ZIP
        zip_coords = get_zip_coords(search_zip)
        if zip_coords:
            st.caption(f"✅ Searching from ZIP {search_zip}")
        else:
            st.caption(f"⚠️ ZIP {search_zip} not recognised — using Edison (08817)")
            search_zip = "08817"

    with loc_col2:
        radius_miles = st.select_slider(
            "Radius",
            options=[10, 25, 50],
            value=25,
            format_func=lambda x: f"{x} mi",
        )

    st.caption(
        f"Showing pets within **{radius_miles} miles** of **{search_zip}**. "
        "Note: results depend on animals currently in the index. "
        "Expand the radius if you see too few matches."
    )
    # Persist UI choices in session state
    st.session_state["search_zip_ui"]   = search_zip
    st.session_state["radius_miles_ui"] = radius_miles

# ── Optional filters ──────────────────────────────────────────────────────────
with st.expander("🔎 Filter by species, age & size", expanded=False):
    f_col1, f_col2, f_col3 = st.columns(3)

    with f_col1:
        st.caption("**Species**")
        filter_dogs = st.checkbox("🐕 Dogs", value=True, key="f_dogs")
        filter_cats = st.checkbox("🐈 Cats", value=True, key="f_cats")
        filter_other = st.checkbox("Other", value=True, key="f_other")

    with f_col2:
        st.caption("**Age group**")
        filter_baby   = st.checkbox("Baby",   value=True, key="f_baby")
        filter_young  = st.checkbox("Young",  value=True, key="f_young")
        filter_adult  = st.checkbox("Adult",  value=True, key="f_adult")
        filter_senior = st.checkbox("Senior", value=True, key="f_senior")

    with f_col3:
        st.caption("**Size**")
        filter_small  = st.checkbox("Small",       value=True, key="f_small")
        filter_medium = st.checkbox("Medium",       value=True, key="f_medium")
        filter_large  = st.checkbox("Large",        value=True, key="f_large")
        filter_xlarge = st.checkbox("Extra Large",  value=True, key="f_xlarge")

    # Build filter dict
    active_species = []
    if filter_dogs:  active_species.append("Dog")
    if filter_cats:  active_species.append("Cat")
    if filter_other: active_species.extend(["Rabbit", "Bird", "Reptile", "Small & Furry", "Horse", "Barnyard"])

    active_ages  = [a for a, v in [("Baby", filter_baby), ("Young", filter_young),
                                    ("Adult", filter_adult), ("Senior", filter_senior)] if v]
    active_sizes = [s for s, v in [("Small", filter_small), ("Medium", filter_medium),
                                    ("Large", filter_large), ("Extra Large", filter_xlarge)] if v]

    st.session_state["filters"] = {
        "species": active_species,
        "ages":    active_ages,
        "sizes":   active_sizes,
    }

    all_species_checked = filter_dogs and filter_cats and filter_other
    all_ages_checked    = filter_baby and filter_young and filter_adult and filter_senior
    all_sizes_checked   = filter_small and filter_medium and filter_large and filter_xlarge
    if not (all_species_checked and all_ages_checked and all_sizes_checked):
        active_labels = []
        if not all_species_checked: active_labels.append(", ".join(active_species) or "none")
        if not all_ages_checked:    active_labels.append(", ".join(active_ages) or "none")
        if not all_sizes_checked:   active_labels.append(", ".join(active_sizes) or "none")
        st.caption(f"Filtering by: {' · '.join(active_labels)}")
    else:
        st.caption("No filters active — showing all species, ages, and sizes.")

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
        # Get postcodes within the user's requested radius
        vs_meta      = vector_store._collection.get(include=["metadatas"])
        all_postcodes = list({m.get("postcode","") for m in vs_meta["metadatas"] if m.get("postcode")})
        nearby_pcs   = filter_postcodes_by_radius(all_postcodes, search_zip, radius_miles)

        # Store location context regardless of results
        st.session_state["search_zip"]      = search_zip
        st.session_state["radius_miles"]    = radius_miles
        st.session_state["nearby_count"]    = len(nearby_pcs)
        st.session_state["last_search_zip"] = search_zip
        st.session_state["last_radius"]     = radius_miles

        # ── No nearby postcodes at all — skip retrieval entirely ──────────
        if not nearby_pcs:
            st.session_state["results"]        = []
            st.session_state["recommendation"] = None
        else:
            # Fetch generously then apply filters in Python
            # (ChromaDB metadata filtering on multiple optional fields is complex;
            # Python post-filtering is simpler and fast at this scale)
            raw_results = retrieve_pets(
                query,
                vector_store=vector_store,
                top_k=min(top_k + 20, 30),
                filter_postcodes=nearby_pcs,
            )

            # Apply species / age / size filters
            filters = st.session_state.get("filters", {})
            active_species = filters.get("species", [])
            active_ages    = filters.get("ages",    [])
            active_sizes   = filters.get("sizes",   [])

            def _passes_filters(doc_meta):
                if active_species:
                    species = doc_meta.get("species", "")
                    if species not in active_species:
                        # Map common variants
                        if species == "Cat" and "Cat" not in active_species: return False
                        if species == "Dog" and "Dog" not in active_species: return False
                        if species not in ("Cat", "Dog") and not any(
                            s not in ("Cat", "Dog") for s in active_species
                        ): return False
                if active_ages:
                    age = doc_meta.get("age_group", "")
                    if age and age not in active_ages: return False
                if active_sizes:
                    size = doc_meta.get("size", "")
                    if size and size not in active_sizes: return False
                return True

            filtered = [(doc, score) for doc, score in raw_results if _passes_filters(doc.metadata)]
            all_results = filtered
            results     = filtered[:top_k]
            context     = format_context(results)

            if not results:
                st.session_state["results"]        = []
                st.session_state["all_results"]    = []
                st.session_state["recommendation"] = None
            else:
                with st.spinner("Generating your personalised recommendation…"):
                    from src.utils.location import get_zip_coords
                    zip_coords = get_zip_coords(search_zip)
                    loc_label  = f"ZIP {search_zip}"
                    recommendation = generate_recommendation(
                        user_query      = query,
                        context         = context,
                        location        = loc_label,
                        radius          = radius_miles,
                        active_filters  = st.session_state.get("filters"),
                    )
                st.session_state["results"]        = results
                st.session_state["all_results"]    = all_results
                st.session_state["recommendation"] = recommendation
                st.session_state["show_more"]      = False

# ── Invalidate stale results when location/radius changes ─────────────────────
if (search_zip  != st.session_state.get("last_search_zip") or
    radius_miles != st.session_state.get("last_radius")):
    if not run:   # don't clear if user just clicked Find My Match
        st.session_state["results"]        = None
        st.session_state["recommendation"] = None

# Render from session state so results survive reruns
if st.session_state.get("results") is not None:
    results        = st.session_state["results"]
    all_results    = st.session_state.get("all_results") or results
    recommendation = st.session_state.get("recommendation")
    sz  = st.session_state.get("search_zip", search_zip)
    rm  = st.session_state.get("radius_miles", radius_miles)
    nc  = st.session_state.get("nearby_count", 0)
    SCORE_THRESHOLD = 0.28   # minimum similarity to show in "more" results

    # ── Empty state ────────────────────────────────────────────────────────
    if not results:
        st.divider()
        st.markdown("### 🔍 No matches found")
        st.info(
            f"No pets found within **{rm} miles** of **{sz}**. "
            "Try expanding your radius to 25 or 50 miles, "
            "searching from a nearby ZIP code, "
            "or check back tomorrow when the index refreshes. "
            f"({nc} animals currently indexed in your area.)"
        )
        st.stop()

    st.caption(f"📍 Showing matches within {rm} miles of {sz} · {nc} animals in range")

    st.subheader("🏆 Your Personalised Recommendation")
    st.markdown(f'<div class="rec-box">{recommendation}</div>', unsafe_allow_html=True)

    st.subheader("📋 Matched Pet Profiles")

    for rank, (doc, score) in enumerate(results, start=1):
        m            = doc.metadata
        pet_name     = m.get("name", "?")
        breed        = m.get("breed", "?")
        score_pct    = f"{score:.0%}"
        photo_url    = m.get("photo_url", "")
        adoption_url = _validate_adoption_url(m)
        fav_key      = _favorite_key(m)
        is_fav       = fav_key in st.session_state["favorites"]

        action_html  = _build_action_html(m, adoption_url)
        badges_html  = _build_badges(m)
        quality_badge = _bio_quality(doc.page_content)
        meta_line    = _meta_line(m)

        card_html = f"""
        <div class="match-card">
          <h4>#{rank} &middot; {pet_name} &mdash; {breed}</h4>
          <div class="meta">{meta_line} &middot; Match: {score_pct}</div>
          {badges_html} {quality_badge}
          {action_html}
        </div>
        """

        # ── Layout ──────────────────────────────────────────────────────────
        if photo_url:
            img_col, card_col = st.columns([1, 2])
            with img_col:
                st.image(photo_url, use_container_width=True)
            with card_col:
                st.markdown(card_html, unsafe_allow_html=True)
                star_label = "⭐ Saved" if is_fav else "☆ Save"
                if st.button(star_label, key=f"fav_{fav_key}_{rank}"):
                    _toggle_favorite(fav_key, m)
                    st.rerun()
        else:
            st.markdown(card_html, unsafe_allow_html=True)
            star_label = "⭐ Saved" if is_fav else "☆ Save"
            if st.button(star_label, key=f"fav_{fav_key}_{rank}"):
                _toggle_favorite(fav_key, m)
                st.rerun()

        with st.expander(f"Read {pet_name}'s full bio"):
            bio_start = doc.page_content.find("Bio: ")
            bio = doc.page_content[bio_start + 5:] if bio_start != -1 else doc.page_content
            st.write(bio)

        st.write("")

    # ── Show more button ──────────────────────────────────────────────────────
    if not st.session_state.get("show_more"):
        extra = [
            (doc, score) for doc, score in all_results[len(results):]
            if score >= SCORE_THRESHOLD
        ]
        if extra:
            st.divider()
            if st.button(
                f"Show {len(extra)} more matches",
                key="show_more_btn",
            ):
                st.session_state["show_more"] = True
                st.rerun()
    else:
        # Render the extra results
        extra = [
            (doc, score) for doc, score in all_results[len(results):]
            if score >= SCORE_THRESHOLD
        ]
        if extra:
            st.divider()
            st.subheader("📋 Additional Matches")
            base_rank = len(results) + 1
            for i, (doc, score) in enumerate(extra):
                m            = doc.metadata
                pet_name     = m.get("name", "?")
                breed        = m.get("breed", "?")
                score_pct    = f"{score:.0%}"
                photo_url    = m.get("photo_url", "")
                adoption_url = _validate_adoption_url(m)
                fav_key      = _favorite_key(m)
                is_fav       = fav_key in st.session_state["favorites"]

                action_html   = _build_action_html(m, adoption_url)
                badges_html   = _build_badges(m)
                quality_badge = _bio_quality(doc.page_content)
                meta_line     = _meta_line(m)

                card_html = f"""
                <div class="match-card">
                  <h4>#{base_rank + i} &middot; {pet_name} &mdash; {breed}</h4>
                  <div class="meta">{meta_line} &middot; Match: {score_pct}</div>
                  {badges_html} {quality_badge}
                  {action_html}
                </div>
                """

                if photo_url:
                    img_col, card_col = st.columns([1, 2])
                    with img_col:
                        st.image(photo_url, use_container_width=True)
                    with card_col:
                        st.markdown(card_html, unsafe_allow_html=True)
                        star_label = "⭐ Saved" if is_fav else "☆ Save"
                        if st.button(star_label, key=f"fav_{fav_key}_extra_{i}"):
                            _toggle_favorite(fav_key, m)
                            st.rerun()
                else:
                    st.markdown(card_html, unsafe_allow_html=True)
                    star_label = "⭐ Saved" if is_fav else "☆ Save"
                    if st.button(star_label, key=f"fav_{fav_key}_extra_{i}"):
                        _toggle_favorite(fav_key, m)
                        st.rerun()

                with st.expander(f"Read {pet_name}'s full bio"):
                    bio_start = doc.page_content.find("Bio: ")
                    bio = doc.page_content[bio_start + 5:] if bio_start != -1 else doc.page_content
                    st.write(bio)
                st.write("")

            if st.button("Show fewer", key="show_less_btn"):
                st.session_state["show_more"] = False
                st.rerun()

st.divider()
st.caption(
    "PawMatch is an AI-powered tool — recommendations are suggestions only. "
    "Always visit the shelter to meet your potential pet. "
    "Pet availability changes daily.  |  "
    "[Privacy Policy](/privacy)"
)