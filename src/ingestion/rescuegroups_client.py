"""
src/ingestion/rescuegroups_client.py

RescueGroups.org v5 API client.
Handles auth, pagination, and radius-based animal search.

API docs: https://test1-api.rescuegroups.org/v5/public/docs
Base URL:  https://api.rescuegroups.org/v5

Key differences from Petfinder:
  - Auth is a simple API key header (no OAuth)
  - Search uses POST with a JSON body (not GET query params)
  - Radius search uses a filterRadius object in the POST body
  - Response follows JSON:API spec: data under data[].attributes,
    photos under included[], org info under included[]
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, Generator, List, Optional

import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

BASE_URL = "https://api.rescuegroups.org/v5"
REQUEST_DELAY = 1.0  # seconds between paginated requests


class RescueGroupsClient:
    """Thin wrapper around the RescueGroups v5 REST API."""

    def __init__(self, api_key: Optional[str] = None) -> None:
        self.api_key = api_key or os.environ.get("RESCUEGROUPS_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "RescueGroups API key not found. "
                "Set RESCUEGROUPS_API_KEY in your .env file."
            )

    def _headers(self) -> Dict[str, str]:
        return {
            "Content-Type": "application/vnd.api+json",
            "Authorization": self.api_key,
        }

    def _post(self, endpoint: str, body: Dict[str, Any]) -> Dict[str, Any]:
        """Make a POST request and return the JSON response."""
        time.sleep(REQUEST_DELAY)
        url = f"{BASE_URL}{endpoint}"
        resp = requests.post(url, headers=self._headers(), json=body, timeout=15)

        if resp.status_code == 429:
            logger.warning("Rate limited — waiting 30s before retry")
            time.sleep(30)
            resp = requests.post(url, headers=self._headers(), json=body, timeout=15)

        resp.raise_for_status()
        return resp.json()

    def _get(self, endpoint: str) -> Dict[str, Any]:
        """Make a GET request and return the JSON response."""
        time.sleep(REQUEST_DELAY)
        url = f"{BASE_URL}{endpoint}"
        resp = requests.get(url, headers=self._headers(), timeout=15)
        resp.raise_for_status()
        return resp.json()

    # ── Public API ────────────────────────────────────────────────────────────

    def fetch_animals_by_location(
        self,
        postal_code: str,
        distance_miles: int = 25,
        limit: int = 100,
        max_pages: Optional[int] = None,
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Yield individual animal records near a postal code.

        Uses the RescueGroups v5 radius search (POST to search/available)
        with pictures and org info included in each response.

        Args:
            postal_code:    US ZIP code, e.g. "08817" for Edison NJ
            distance_miles: Search radius in miles
            limit:          Animals per page (max 250)
            max_pages:      Safety cap on pages. None = fetch all.
        """
        page = 1
        total_pages = None

        while True:
            body = {
                "data": {
                    "filterRadius": {
                        "postalcode": postal_code,
                        "miles": distance_miles,
                    }
                },
            }

            endpoint = (
                f"/public/animals/search/available/"
                f"?limit={limit}&page={page}"
                f"&include[]=pictures&include[]=orgs&include[]=breeds&include[]=species"
                f"&sort=distance"
            )

            logger.info(
                "Fetching RescueGroups page %s (ZIP=%s, radius=%s mi)",
                page, postal_code, distance_miles,
            )

            data = self._post(endpoint, body)
            animals: List[Dict[str, Any]] = data.get("data", [])
            included: List[Dict[str, Any]] = data.get("included", [])
            meta: Dict[str, Any] = data.get("meta", {})

            if total_pages is None:
                total_pages = meta.get("pages", 1)
                logger.info(
                    "Total animals: %s across %s pages",
                    meta.get("count", "?"),
                    total_pages,
                )

            # Build lookup maps for included data
            pictures_by_animal: Dict[str, List[Dict]] = {}
            orgs_by_id: Dict[str, Dict] = {}
            breeds_by_id: Dict[str, str] = {}
            species_by_id: Dict[str, str] = {}

            for item in included:
                item_type = item.get("type")
                item_id = item.get("id")
                attrs = item.get("attributes", {})

                if item_type == "pictures":
                    # Pictures link back to their animal via relationships
                    animal_rel = (
                        item.get("relationships", {})
                        .get("animals", {})
                        .get("data", {})
                    )
                    if isinstance(animal_rel, dict):
                        aid = animal_rel.get("id")
                        if aid:
                            pictures_by_animal.setdefault(aid, []).append(attrs)

                elif item_type == "orgs":
                    orgs_by_id[item_id] = attrs

                elif item_type == "breeds":
                    breeds_by_id[item_id] = attrs.get("name", "")

                elif item_type == "species":
                    species_by_id[item_id] = attrs.get("singular", "")

            # Attach included data to each animal and yield
            for animal in animals:
                animal_id = animal.get("id")
                attrs = animal.get("attributes", {})
                relationships = animal.get("relationships", {})

                # Attach photos
                attrs["_photos"] = pictures_by_animal.get(animal_id, [])

                # Attach org info
                org_rel = relationships.get("orgs", {}).get("data", {})
                if isinstance(org_rel, dict):
                    org_id = org_rel.get("id")
                    attrs["_org"] = orgs_by_id.get(org_id, {})
                    attrs["_org_id"] = org_id
                elif isinstance(org_rel, list) and org_rel:
                    org_id = org_rel[0].get("id")
                    attrs["_org"] = orgs_by_id.get(org_id, {})
                    attrs["_org_id"] = org_id
                else:
                    attrs["_org"] = {}
                    attrs["_org_id"] = ""

                # Attach primary breed name
                breed_rel = relationships.get("breeds", {}).get("data", [])
                if isinstance(breed_rel, list) and breed_rel:
                    attrs["_breed_primary"] = breeds_by_id.get(
                        breed_rel[0].get("id", ""), ""
                    )
                    if len(breed_rel) > 1:
                        attrs["_breed_secondary"] = breeds_by_id.get(
                            breed_rel[1].get("id", ""), ""
                        )
                else:
                    attrs["_breed_primary"] = attrs.get("breedPrimary", "")
                    attrs["_breed_secondary"] = attrs.get("breedSecondary", "")

                # Attach species name
                species_rel = relationships.get("species", {}).get("data", {})
                if isinstance(species_rel, dict):
                    attrs["_species"] = species_by_id.get(
                        species_rel.get("id", ""), ""
                    )
                else:
                    attrs["_species"] = ""

                attrs["_id"] = animal_id
                yield attrs

            if page >= total_pages:
                break
            if max_pages and page >= max_pages:
                logger.info("Reached max_pages limit (%s), stopping.", max_pages)
                break

            page += 1

    def test_connection(self) -> bool:
        """Quick check that the API key is valid. Returns True if OK."""
        try:
            data = self._get("/public/animals/species/")
            return "data" in data
        except Exception as e:
            logger.error("RescueGroups connection test failed: %s", e)
            return False