"""
src/ingestion/rescuegroups_transformer.py

Transforms raw RescueGroups v5 API animal records into the normalised
format used by the rest of the pipeline (LangChain Documents + SQLite).

RescueGroups has a richer schema than Petfinder:
  - energyLevel, activityLevel, exerciseNeeds
  - isYardRequired, fenceNeeds
  - isCatsOk, isDogsOk, isKidsOk, isSeniorsOk
  - obedienceTraining, ownerExperience
  - vocalLevel, sheddingLevel, groomingNeeds
  - descriptionText (clean, no HTML)

All of these feed directly into the embedding content, making
retrieval quality significantly better than the mock CSV.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def _safe(value: Any, fallback: str = "Unknown") -> str:
    if value is None:
        return fallback
    s = str(value).strip()
    return s if s else fallback


def _bool_label(value: Optional[bool], true_str: str, false_str: str) -> str:
    if value is True:
        return true_str
    if value is False:
        return false_str
    return "Unknown"


def _build_bio(a: Dict[str, Any]) -> str:
    """
    Build a rich biography string from RescueGroups attributes.
    Falls back gracefully when fields are missing.
    """
    parts: List[str] = []
    name = _safe(a.get("name"), "This animal")
    species = _safe(a.get("_species") or a.get("species"), "pet")
    breed = _safe(a.get("_breed_primary") or a.get("breedString"), "Mixed breed")
    secondary = a.get("_breed_secondary") or a.get("breedSecondary")
    mixed = a.get("isBreedMixed", False)

    breed_str = breed
    if mixed and secondary:
        breed_str = f"{breed}/{secondary} mix"
    elif mixed:
        breed_str = f"{breed} mix"

    age_group = _safe(a.get("ageGroup"), "")
    sex = _safe(a.get("sex"), "")
    size = _safe(a.get("sizeGroup"), "")

    # ── Opener ────────────────────────────────────────────────────────────
    opener_parts = [p for p in [age_group.lower(), sex.lower(), breed_str, species.lower()] if p and p != "unknown"]
    parts.append(f"{name} is a {' '.join(opener_parts)}.")

    # ── Energy & activity ─────────────────────────────────────────────────
    energy = a.get("energyLevel")
    activity = a.get("activityLevel")
    exercise = a.get("exerciseNeeds")
    if any([energy, activity, exercise]):
        notes = []
        if energy:    notes.append(f"{energy.lower()} energy")
        if activity:  notes.append(f"{activity.lower()} activity level")
        if exercise:  notes.append(f"{exercise.lower()} exercise needs")
        parts.append(f"{name} has {', '.join(notes)}.")

    # ── Living situation ───────────────────────────────────────────────────
    yard = a.get("isYardRequired")
    fence = a.get("fenceNeeds")
    indoor = a.get("indoorOutdoor")
    living_notes = []
    if yard is True:   living_notes.append("requires a yard")
    if yard is False:  living_notes.append("does not require a yard — suitable for apartment living")
    if fence and fence != "Not required":
        living_notes.append(f"needs a {fence.lower()} fence")
    if indoor:         living_notes.append(f"is {indoor.lower()}")
    if living_notes:
        parts.append(f"{name} {', '.join(living_notes)}.")

    # ── Compatibility ──────────────────────────────────────────────────────
    compat = []
    if a.get("isKidsOk") is True:    compat.append("good with children")
    if a.get("isKidsOk") is False:   compat.append("not recommended for homes with children")
    if a.get("isDogsOk") is True:    compat.append("good with other dogs")
    if a.get("isDogsOk") is False:   compat.append("should be the only dog in the home")
    if a.get("isCatsOk") is True:    compat.append("good with cats")
    if a.get("isCatsOk") is False:   compat.append("not suitable for homes with cats")
    if a.get("isSeniorsOk") is True: compat.append("great with seniors")
    if compat:
        parts.append(f"{name} is {', '.join(compat)}.")

    # ── Training & experience ──────────────────────────────────────────────
    training = a.get("obedienceTraining")
    experience = a.get("ownerExperience")
    if training:   parts.append(f"Training: {training}.")
    if experience and experience != "None":
        parts.append(f"Best suited for an owner with {experience.lower()} experience.")

    # ── Health & care ─────────────────────────────────────────────────────
    health = []
    if a.get("isAltered"):              health.append("spayed/neutered")
    if a.get("isCurrentVaccinations"):  health.append("current on vaccinations")
    if a.get("isMicrochipped"):         health.append("microchipped")
    if a.get("isHousetrained"):         health.append("house-trained")
    if a.get("isSpecialNeeds"):         health.append("has special needs")
    if health:
        parts.append(f"{name} is {', '.join(health)}.")

    # ── Personality traits ─────────────────────────────────────────────────
    vocal = a.get("vocalLevel")
    shedding = a.get("sheddingLevel")
    grooming = a.get("groomingNeeds")
    coat = a.get("coatLength")
    personality = []
    if vocal and vocal != "Some":   personality.append(f"{vocal.lower()} vocal")
    if shedding:                    personality.append(f"{shedding.lower()} shedding")
    if grooming:                    personality.append(f"{grooming.lower()} grooming needs")
    if coat:                        personality.append(f"{coat.lower()} coat")
    if personality:
        parts.append(f"Traits: {', '.join(personality)}.")

    # ── Official description (best content) ───────────────────────────────
    desc = (a.get("descriptionText") or a.get("summary") or "").strip()
    if desc:
        parts.append(desc)

    return " ".join(parts)


def animal_to_document(a: Dict[str, Any]) -> Optional[Document]:
    """
    Convert a RescueGroups animal attributes dict into a LangChain Document.
    Returns None if the record is missing critical fields.
    """
    animal_id = a.get("_id")
    name = (a.get("name") or "").strip()

    if not animal_id or not name:
        return None

    species  = _safe(a.get("_species") or a.get("species"), "Unknown")
    breed    = _safe(a.get("_breed_primary") or a.get("breedString"), "Mixed")
    mixed    = a.get("isBreedMixed", False)
    breed_label = f"{breed} mix" if mixed else breed

    age_group = _safe(a.get("ageGroup"), "Unknown")
    size      = _safe(a.get("sizeGroup"), "Unknown")
    sex       = _safe(a.get("sex"), "Unknown")
    energy    = _safe(a.get("energyLevel"), "Unknown")

    # Location from org
    org = a.get("_org", {})
    city     = _safe(org.get("city"), "")
    state    = _safe(org.get("state"), "")
    postcode = _safe(org.get("postalcode"), "")
    org_name = _safe(org.get("name"), "")
    org_url  = org.get("adoptionUrl") or org.get("url") or ""

    # Photo — first picture's "large" URL
    photos = a.get("_photos", [])
    photo_url = photos[0].get("large", "") if photos else (
        a.get("pictureThumbnailUrl") or ""
    )

    # Adoption URL
    adoption_url = a.get("url") or org_url

    bio = _build_bio(a)

    page_content = (
        f"Name: {name}. "
        f"Species: {species}. "
        f"Breed: {breed_label}. "
        f"Age group: {age_group}. "
        f"Size: {size}. "
        f"Sex: {sex}. "
        f"Energy level: {energy}. "
        f"Good with kids: {a.get('isKidsOk')}. "
        f"Good with dogs: {a.get('isDogsOk')}. "
        f"Good with cats: {a.get('isCatsOk')}. "
        f"Requires yard: {a.get('isYardRequired')}. "
        f"House trained: {a.get('isHousetrained')}. "
        f"Special needs: {a.get('isSpecialNeeds')}. "
        f"Activity level: {a.get('activityLevel')}. "
        f"Exercise needs: {a.get('exerciseNeeds')}. "
        f"Owner experience needed: {a.get('ownerExperience')}. "
        f"Location: {city}, {state} {postcode}. "
        f"Shelter: {org_name}. "
        f"Bio: {bio}"
    )

    metadata = {
        "rescuegroups_id": str(animal_id),
        "org_id":          _safe(a.get("_org_id"), ""),
        "org_name":        org_name,
        "name":            name,
        "species":         species,
        "breed":           breed_label,
        "age_group":       age_group,
        "size":            size,
        "sex":             sex,
        "energy_level":    energy,
        "activity_level":  _safe(a.get("activityLevel"), ""),
        "good_with_kids":  a.get("isKidsOk"),
        "good_with_dogs":  a.get("isDogsOk"),
        "good_with_cats":  a.get("isCatsOk"),
        "is_seniors_ok":   a.get("isSeniorsOk"),
        "requires_yard":   a.get("isYardRequired"),
        "fence_needs":     _safe(a.get("fenceNeeds"), ""),
        "house_trained":   a.get("isHousetrained"),
        "is_altered":      a.get("isAltered"),
        "special_needs":   a.get("isSpecialNeeds"),
        "shots_current":   a.get("isCurrentVaccinations"),
        "microchipped":    a.get("isMicrochipped"),
        "photo_url":       photo_url,
        "adoption_url":    adoption_url,
        "city":            city,
        "state":           state,
        "postcode":        postcode,
        "status":          "adoptable",
    }

    return Document(page_content=page_content, metadata=metadata)


def animals_to_documents(animals: List[Dict[str, Any]]) -> List[Document]:
    """Batch-convert raw animal records, skipping invalid ones."""
    docs = []
    skipped = 0
    for animal in animals:
        doc = animal_to_document(animal)
        if doc:
            docs.append(doc)
        else:
            skipped += 1
    if skipped:
        logger.warning("Skipped %s invalid animal records.", skipped)
    return docs