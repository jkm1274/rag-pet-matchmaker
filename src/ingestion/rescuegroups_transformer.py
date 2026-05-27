"""
src/ingestion/rescuegroups_transformer.py

Transforms raw HTTP/JSON API animal records into LangChain Documents.
Field names follow the HTTP/JSON API v2 schema (animalXxx, orgXxx).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def _s(value: Any, fallback: str = "") -> str:
    if value is None:
        return fallback
    s = str(value).strip()
    return s if s else fallback


def _yes(value: Any) -> Optional[bool]:
    """Convert Yes/No/None string to bool."""
    if value is None:
        return None
    v = str(value).strip().lower()
    if v == "yes": return True
    if v == "no":  return False
    return None


def _build_bio(a: Dict[str, Any], org: Dict[str, Any]) -> str:
    """Build a rich biography from HTTP/JSON API animal attributes."""
    parts: List[str] = []
    name    = _s(a.get("animalName"), "This animal")
    species = _s(a.get("animalSpecies"), "pet")
    breed   = _s(a.get("animalBreed") or a.get("animalPrimaryBreed"), "Mixed breed")
    mixed   = str(a.get("animalMixedBreed", "")).lower() == "yes"
    breed_str = f"{breed} mix" if mixed else breed
    age     = _s(a.get("animalGeneralAge") or a.get("animalAgeString"))
    sex     = _s(a.get("animalSex"))
    size    = _s(a.get("animalGeneralSizePotential") or a.get("animalSizeCurrent"))

    # Opener
    parts_opener = [p for p in [age.lower(), sex.lower(), breed_str, species.lower()] if p]
    parts.append(f"{name} is a {' '.join(parts_opener)}.")

    # Energy & activity
    energy   = a.get("animalEnergyLevel")
    activity = a.get("animalActivityLevel")
    exercise = a.get("animalExerciseNeeds")
    notes = []
    if energy:   notes.append(f"{energy.lower()} energy")
    if activity: notes.append(f"{activity.lower()} activity")
    if exercise: notes.append(f"{exercise.lower()} exercise needs")
    if notes: parts.append(f"{name} has {', '.join(notes)}.")

    # Living situation
    yard    = _yes(a.get("animalYardRequired"))
    apt     = _yes(a.get("animalApartment"))
    indoor  = a.get("animalIndoorOutdoor")
    fence   = a.get("animalFence")
    living  = []
    if yard is True:            living.append("requires a yard")
    if yard is False or apt is True: living.append("suitable for apartment living")
    if fence and fence not in ("", "No"): living.append(f"needs a {fence.lower()} fence")
    if indoor:                  living.append(f"is {indoor.lower()}")
    if living: parts.append(f"{name} {', '.join(living)}.")

    # Compatibility
    compat = []
    if _yes(a.get("animalOKWithKids")) is True:    compat.append("good with children")
    if _yes(a.get("animalOKWithKids")) is False:   compat.append("not recommended for homes with children")
    if _yes(a.get("animalOKWithDogs")) is True:    compat.append("good with dogs")
    if _yes(a.get("animalOKWithDogs")) is False:   compat.append("should be the only dog")
    if _yes(a.get("animalOKWithCats")) is True:    compat.append("good with cats")
    if _yes(a.get("animalOKWithCats")) is False:   compat.append("not suitable with cats")
    if _yes(a.get("animalOKForSeniors")) is True:  compat.append("great with seniors")
    if compat: parts.append(f"{name} is {', '.join(compat)}.")

    # Training
    training    = a.get("animalObedienceTraining")
    experience  = a.get("animalOwnerExperience")
    if training:  parts.append(f"Training: {training}.")
    if experience and experience not in ("", "None"):
        parts.append(f"Best for an owner with {experience.lower()} experience.")

    # Health
    health = []
    if _yes(a.get("animalAltered")) is True:   health.append("spayed/neutered")
    if _yes(a.get("animalUptodate")) is True:  health.append("up to date on vaccinations")
    if _yes(a.get("animalHousetrained")) is True: health.append("house-trained")
    if _yes(a.get("animalSpecialneeds")) is True: health.append("has special needs")
    if health: parts.append(f"{name} is {', '.join(health)}.")

    # Traits
    traits = []
    vocal    = a.get("animalVocal")
    shedding = a.get("animalShedding")
    grooming = a.get("animalGroomingNeeds")
    coat     = a.get("animalCoatLength")
    hypo     = _yes(a.get("animalHypoallergenic"))
    if vocal:    traits.append(f"{vocal.lower()} vocal")
    if shedding: traits.append(f"{shedding.lower()} shedding")
    if grooming: traits.append(f"{grooming.lower()} grooming needs")
    if coat:     traits.append(f"{coat.lower()} coat")
    if hypo:     traits.append("hypoallergenic")
    if traits: parts.append(f"Traits: {', '.join(traits)}.")

    # Org location context
    org_name = _s(org.get("orgName"))
    org_city = _s(org.get("orgCity"))
    org_state= _s(org.get("orgState"))
    if org_name: parts.append(f"Available at {org_name}" + (f" in {org_city}, {org_state}." if org_city else "."))

    # Official description
    desc = _s(a.get("animalDescriptionPlain") or a.get("animalSummary"))
    if desc: parts.append(desc)

    return " ".join(parts)


def animal_to_document(a: Dict[str, Any]) -> Optional[Document]:
    """Convert a raw HTTP/JSON API animal dict into a LangChain Document."""
    animal_id = a.get("_id") or a.get("animalID")
    name      = _s(a.get("animalName"))
    if not animal_id or not name:
        return None

    org     = a.get("_org", {})
    species = _s(a.get("animalSpecies"))
    breed   = _s(a.get("animalBreed") or a.get("animalPrimaryBreed"), "Mixed")
    mixed   = str(a.get("animalMixedBreed", "")).lower() == "yes"
    breed_label = f"{breed} mix" if mixed else breed

    age     = _s(a.get("animalGeneralAge") or a.get("animalAgeString"))
    size    = _s(a.get("animalGeneralSizePotential") or a.get("animalSizeCurrent"))
    sex     = _s(a.get("animalSex"))
    energy  = _s(a.get("animalEnergyLevel"))
    activity= _s(a.get("animalActivityLevel"))

    org_id   = _s(a.get("animalOrgID"))
    org_name = _s(org.get("orgName"))
    org_city = _s(org.get("orgCity"))
    org_state= _s(org.get("orgState"))
    org_zip  = _s(org.get("orgPostalcode"))
    org_phone= _s(org.get("orgPhone"))
    org_email= _s(org.get("orgEmail"))

    location_str = _s(a.get("animalLocationCitystate"))
    city  = org_city or (location_str.split(",")[0].strip() if "," in location_str else "")
    state = org_state or (location_str.split(",")[1].strip() if "," in location_str else "")

    photo_url    = _s(a.get("animalThumbnailUrl"))
    adoption_url = _s(a.get("animalUrl"))
    org_url      = _s(org.get("orgUrl") or org.get("orgFacebookUrl"))

    bio = _build_bio(a, org)

    page_content = (
        f"Name: {name}. Species: {species}. Breed: {breed_label}. "
        f"Age: {age}. Size: {size}. Sex: {sex}. "
        f"Energy level: {energy}. Activity level: {activity}. "
        f"Good with kids: {a.get('animalOKWithKids')}. "
        f"Good with dogs: {a.get('animalOKWithDogs')}. "
        f"Good with cats: {a.get('animalOKWithCats')}. "
        f"Good with seniors: {a.get('animalOKForSeniors')}. "
        f"Requires yard: {a.get('animalYardRequired')}. "
        f"Apartment suitable: {a.get('animalApartment')}. "
        f"House trained: {a.get('animalHousetrained')}. "
        f"Special needs: {a.get('animalSpecialneeds')}. "
        f"Hypoallergenic: {a.get('animalHypoallergenic')}. "
        f"Owner experience: {a.get('animalOwnerExperience')}. "
        f"Location: {city}, {state}. "
        f"Shelter: {org_name}. "
        f"Bio: {bio}"
    )

    metadata = {
        "rescuegroups_id": str(animal_id),
        "org_id":          org_id,
        "org_name":        org_name,
        "org_phone":       org_phone,
        "org_email":       org_email,
        "org_url":         org_url,
        "name":            name,
        "species":         species,
        "breed":           breed_label,
        "age_group":       age,
        "size":            size,
        "sex":             sex,
        "energy_level":    energy,
        "activity_level":  activity,
        "good_with_kids":  _yes(a.get("animalOKWithKids")),
        "good_with_dogs":  _yes(a.get("animalOKWithDogs")),
        "good_with_cats":  _yes(a.get("animalOKWithCats")),
        "is_seniors_ok":   _yes(a.get("animalOKForSeniors")),
        "requires_yard":   _yes(a.get("animalYardRequired")),
        "apartment_ok":    _yes(a.get("animalApartment")),
        "house_trained":   _yes(a.get("animalHousetrained")),
        "is_altered":      _yes(a.get("animalAltered")),
        "special_needs":   _yes(a.get("animalSpecialneeds")),
        "shots_current":   _yes(a.get("animalUptodate")),
        "hypoallergenic":  _yes(a.get("animalHypoallergenic")),
        "needs_foster":    _yes(a.get("animalNeedsFoster")),
        "photo_url":       photo_url,
        "adoption_url":    adoption_url,
        "city":            city,
        "state":           state,
        "postcode":        org_zip,
        "status":          "adoptable",
        "slug":            _s(a.get("animalRescueID")),
    }

    return Document(page_content=page_content, metadata=metadata)


def animals_to_documents(animals: List[Dict[str, Any]]) -> List[Document]:
    docs, skipped = [], 0
    for animal in animals:
        doc = animal_to_document(animal)
        if doc: docs.append(doc)
        else:   skipped += 1
    if skipped:
        logger.warning("Skipped %d invalid records.", skipped)
    return docs