"""
src/ingestion/rescuegroups_client.py

RescueGroups HTTP/JSON API (v2) client.

Key facts:
  - Single endpoint: POST https://api.rescuegroups.org/http/v2.json
  - API key goes in the request body (not a header)
  - All requests are POST with JSON body
  - Two-step approach for org data:
      1. Search animals → collect animalOrgID values
      2. Batch lookup orgs by ID → merge into animal records
  - location fields (locationName etc.) are empty on animal records;
    org data must be fetched separately via objectType: orgs
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, Generator, List, Optional, Set

import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

API_URL      = "https://api.rescuegroups.org/http/v2.json"
REQUEST_DELAY = 0.5  # seconds between requests

# All valid animal fields we want
ANIMAL_FIELDS = [
    "animalID", "animalOrgID", "animalName",
    "animalSpecies", "animalBreed", "animalPrimaryBreed",
    "animalSecondaryBreed", "animalMixedBreed",
    "animalGeneralAge", "animalAgeString", "animalBirthdate",
    "animalSex", "animalGeneralSizePotential", "animalSizeCurrent",
    "animalEnergyLevel", "animalActivityLevel", "animalExerciseNeeds",
    "animalDescriptionPlain", "animalSummary",
    "animalThumbnailUrl", "animalUrl",
    # Compatibility
    "animalOKWithKids", "animalOKWithDogs", "animalOKWithCats",
    "animalOKForSeniors", "animalYardRequired", "animalApartment",
    "animalHousetrained", "animalAltered", "animalUptodate",
    "animalSpecialneeds", "animalSpecialneedsDescription",
    "animalHypoallergenic", "animalDeclawed",
    "animalObedienceTraining", "animalOwnerExperience",
    "animalIndoorOutdoor", "animalFence",
    "animalNeedsFoster", "animalCourtesy",
    # Health / traits
    "animalShedding", "animalGroomingNeeds", "animalCoatLength",
    "animalVocal", "animalNewPeople",
    # Location
    "animalLocationCitystate", "animalLocationDistance",
    "animalLocationState",
]

# Valid org fields (confirmed from API define)
ORG_FIELDS = [
    "orgID", "orgName", "orgCity", "orgState",
    "orgPostalcode", "orgPhone", "orgEmail", "orgFacebookUrl",
]


class RescueGroupsClient:
    """HTTP/JSON API v2 client for RescueGroups.org."""

    def __init__(self, api_key: Optional[str] = None) -> None:
        self.api_key = api_key or os.environ.get("RESCUEGROUPS_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "RescueGroups API key not found. "
                "Set RESCUEGROUPS_API_KEY in your .env file."
            )
        self._org_cache: Dict[str, Dict] = {}

    def _post(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """POST to the API and return the JSON response."""
        time.sleep(REQUEST_DELAY)
        resp = requests.post(
            API_URL,
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=20,
        )
        resp.raise_for_status()
        return resp.json()

    def _fetch_orgs(self, org_ids: List[str]) -> Dict[str, Dict]:
        """Batch fetch org details for a list of org IDs.
        Results are cached so each org is only fetched once per sync.
        """
        uncached = [oid for oid in org_ids if oid not in self._org_cache]

        if not uncached:
            return {oid: self._org_cache[oid] for oid in org_ids}

        # Fetch in batches of 50 to avoid overly large requests
        batch_size = 50
        for i in range(0, len(uncached), batch_size):
            batch = uncached[i:i + batch_size]
            data = self._post({
                "apikey": self.api_key,
                "objectType": "orgs",
                "objectAction": "publicSearch",
                "search": {
                    "calcFoundRows": "No",
                    "resultStart": 0,
                    "resultLimit": batch_size,
                    "fields": ORG_FIELDS,
                    "filters": [
                        {"fieldName": "orgID", "operation": "equals", "criteria": batch},
                    ],
                },
            })

            orgs = data.get("data", {})
            for org_id, attrs in orgs.items():
                self._org_cache[org_id] = attrs
                logger.debug("Cached org %s: %s", org_id, attrs.get("orgName"))

            logger.info("Fetched %d orgs (batch %d)", len(orgs), i // batch_size + 1)

        return {oid: self._org_cache.get(oid, {}) for oid in org_ids}

    def fetch_animals_by_location(
        self,
        postal_code: str,
        distance_miles: int = 25,
        limit: int = 100,
        max_pages: Optional[int] = None,
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Yield animal dicts enriched with org data, near a postal code.

        Uses pagination via resultStart. For each page:
          1. Fetches animals with location filter
          2. Batch-fetches org data for unique orgIDs on that page
          3. Merges org data into each animal record
          4. Yields each enriched animal dict
        """
        result_start = 0
        total_found  = None
        page         = 0

        while True:
            page += 1
            logger.info(
                "Fetching HTTP/JSON page %d (start=%d, ZIP=%s, radius=%d mi)",
                page, result_start, postal_code, distance_miles,
            )

            data = self._post({
                "apikey": self.api_key,
                "objectType": "animals",
                "objectAction": "publicSearch",
                "search": {
                    "calcFoundRows": "Yes",
                    "resultStart": result_start,
                    "resultLimit": limit,
                    "fields": ANIMAL_FIELDS,
                    "filters": [
                        {"fieldName": "animalStatus",
                         "operation": "equals",   "criteria": "Available"},
                        {"fieldName": "animalLocation",
                         "operation": "equals",   "criteria": postal_code},
                        {"fieldName": "animalLocationDistance",
                         "operation": "radius",   "criteria": str(distance_miles)},
                    ],
                },
            })

            status  = data.get("status")
            animals = data.get("data", {})

            if total_found is None:
                total_found = int(data.get("foundRows", 0))
                logger.info("Total animals: %d", total_found)

            if status == "error" or not animals:
                logger.warning("No animals on page %d (status=%s)", page, status)
                break

            # ── Batch fetch org data for this page ─────────────────────────
            org_ids = list({
                a.get("animalOrgID")
                for a in animals.values()
                if a.get("animalOrgID")
            })
            if org_ids:
                self._fetch_orgs(org_ids)

            # ── Yield each animal enriched with org data ───────────────────
            for animal_id, attrs in animals.items():
                attrs["_id"]  = animal_id
                org_id        = attrs.get("animalOrgID", "")
                attrs["_org"] = self._org_cache.get(org_id, {})
                yield attrs

            result_start += limit

            if result_start >= total_found:
                break
            if max_pages and page >= max_pages:
                logger.info("Reached max_pages limit (%d), stopping.", max_pages)
                break

    def test_connection(self) -> bool:
        """Quick check that the API key is valid."""
        try:
            data = self._post({
                "apikey": self.api_key,
                "objectType": "animals",
                "objectAction": "define",
            })
            return data.get("status") == "ok"
        except Exception as e:
            logger.error("Connection test failed: %s", e)
            return False