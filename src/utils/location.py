"""
src/utils/location.py

Lightweight ZIP code distance utilities using the Haversine formula.
No external geocoding API needed — uses a hardcoded lat/lng table for
the ZIP codes we know are in the index, with a free API fallback for unknowns.
"""

from __future__ import annotations

import math
import logging
from typing import Dict, List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)

# ── Pre-seeded coords for ZIPs we know are in the index ───────────────────────
ZIP_COORDS: Dict[str, Tuple[float, float]] = {
    # NJ
    "07067": (40.5776, -74.2846),  # Colonia
    "07747": (40.4220, -74.2438),  # Matawan
    "07748": (40.4123, -74.2624),  # Matawan alt
    "07718": (40.4151, -74.1052),  # Belford
    "07930": (40.7784, -74.6965),  # Chester
    "07762": (40.1484, -74.0318),  # Spring Lake
    "08618": (40.2326, -74.7546),  # Trenton
    "07016": (40.6568, -74.3029),  # Cranford
    "07751": (40.3832, -74.2857),  # Morganville
    "08872": (40.4590, -74.3271),  # Sayreville
    "08861": (40.5068, -74.2677),  # Perth Amboy
    "08850": (40.5576, -74.4341),  # Milltown
    "07721": (40.4432, -74.2274),  # Cliffwood
    "07006": (40.8565, -74.2985),  # Caldwell
    "07755": (40.1710, -74.0129),  # Oakhurst
    "08817": (40.5187, -74.4121),  # Edison
    "08901": (40.4774, -74.4435),  # New Brunswick
    "08540": (40.3573, -74.6672),  # Princeton
    "07001": (40.5790, -74.2807),  # Avenel
    "07302": (40.7178, -74.0431),  # Jersey City
    "07030": (40.7440, -74.0324),  # Hoboken
    "07960": (40.7968, -74.4814),  # Morristown
    "07728": (40.2593, -74.2735),  # Freehold
    "08753": (39.9537, -74.1979),  # Toms River
    "08002": (39.9348, -75.0163),  # Cherry Hill
    "08401": (39.3643, -74.4229),  # Atlantic City
    "07102": (40.7357, -74.1724),  # Newark
    # NY
    "11202": (40.6892, -73.9442),  # Brooklyn
    "11229": (40.6012, -73.9389),  # Brooklyn
    "11214": (40.6198, -74.0093),  # Brooklyn Bath Beach
    "11224": (40.5754, -74.0001),  # Brooklyn Coney Island
    "10302": (40.6295, -74.1195),  # Staten Island
    # GA — out of area, will be filtered by distance
    "30470": (34.5987, -83.9060),  # Waleska GA
    # NY additions
    "11201": (40.6892, -73.9931),  # Brooklyn Heights
    "10001": (40.7484, -74.0040),  # Manhattan (Chelsea)
    "10451": (40.8176, -73.9256),  # Bronx
    "11354": (40.7677, -73.8330),  # Queens (Flushing)
    "10301": (40.6298, -74.0943),  # Staten Island
    "10601": (41.0340, -73.7629),  # White Plains
    "10701": (40.9312, -73.8988),  # Yonkers
    # PA additions
    "19103": (39.9526, -75.1652),  # Philadelphia
    "19401": (40.1215, -75.3397),  # Norristown
    "19406": (40.0918, -75.3852),  # King of Prussia
    "08002": (39.9348, -75.0163),  # Cherry Hill NJ
    "08401": (39.3643, -74.4229),  # Atlantic City NJ
    "08096": (39.8318, -75.1527),  # Woodbury NJ
}


def haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distance in miles between two lat/lng points."""
    R = 3958.8
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi    = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return 2 * R * math.asin(math.sqrt(a))


def get_zip_coords(zipcode: str) -> Optional[Tuple[float, float]]:
    """
    Return (lat, lng) for a ZIP. Checks local cache first,
    then falls back to the free Zippopotam.us API (no key required).
    """
    zipcode = zipcode.strip()
    if zipcode in ZIP_COORDS:
        return ZIP_COORDS[zipcode]

    try:
        resp = requests.get(
            f"https://api.zippopotam.us/us/{zipcode}",
            timeout=5,
        )
        if resp.ok:
            place = resp.json()["places"][0]
            lat, lng = float(place["latitude"]), float(place["longitude"])
            ZIP_COORDS[zipcode] = (lat, lng)
            logger.info("Geocoded ZIP %s → (%.4f, %.4f)", zipcode, lat, lng)
            return (lat, lng)
    except Exception as e:
        logger.warning("ZIP lookup failed for %s: %s", zipcode, e)

    return None


def filter_postcodes_by_radius(
    all_postcodes: List[str],
    center_zip: str,
    radius_miles: int,
) -> List[str]:
    """Return postcodes within radius_miles of center_zip."""
    center = get_zip_coords(center_zip)
    if not center:
        return all_postcodes   # can't geocode — return everything

    clat, clng = center
    result = []
    for pc in set(all_postcodes):
        coords = get_zip_coords(pc)
        if coords and haversine_miles(clat, clng, coords[0], coords[1]) <= radius_miles:
            result.append(pc)
    return result


# Tri-state ZIP suggestions for the UI dropdown
ZIP_SUGGESTIONS = {
    # ── New Jersey ────────────────────────────────────────────────────────
    "Edison, NJ (08817)":          "08817",
    "Newark, NJ (07102)":          "07102",
    "Jersey City, NJ (07302)":     "07302",
    "Hoboken, NJ (07030)":         "07030",
    "Princeton, NJ (08540)":       "08540",
    "New Brunswick, NJ (08901)":   "08901",
    "Trenton, NJ (08618)":         "08618",
    "Morristown, NJ (07960)":      "07960",
    "Freehold, NJ (07728)":        "07728",
    "Toms River, NJ (08753)":      "08753",
    "Cherry Hill, NJ (08002)":     "08002",
    "Atlantic City, NJ (08401)":   "08401",
    # ── New York ──────────────────────────────────────────────────────────
    "Brooklyn, NY (11201)":        "11201",
    "Manhattan, NY (10001)":       "10001",
    "Queens, NY (11354)":          "11354",
    "Staten Island, NY (10301)":   "10301",
    "Bronx, NY (10451)":           "10451",
    "White Plains, NY (10601)":    "10601",
    "Yonkers, NY (10701)":         "10701",
    # ── Pennsylvania ─────────────────────────────────────────────────────
    "Philadelphia, PA (19103)":    "19103",
    "King of Prussia, PA (19406)": "19406",
    "Norristown, PA (19401)":      "19401",
}

# Keep old name for backward compatibility
NJ_ZIP_SUGGESTIONS = ZIP_SUGGESTIONS