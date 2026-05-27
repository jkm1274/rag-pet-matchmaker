"""
debug_httpjson4.py — print the raw response for one animal
to see exactly what fields are populated vs empty.
Run: python debug_httpjson4.py
"""
import os, json, requests
from dotenv import load_dotenv
load_dotenv()

key = os.getenv("RESCUEGROUPS_API_KEY")
URL = "https://api.rescuegroups.org/http/v2.json"

def post(payload):
    resp = requests.post(URL, headers={"Content-Type": "application/json"}, json=payload, timeout=20)
    resp.raise_for_status()
    return resp.json()

# ── Get one animal, all fields, print raw ─────────────────────────────────────
data = post({
    "apikey": key,
    "objectType": "animals",
    "objectAction": "publicSearch",
    "search": {
        "calcFoundRows": "No",
        "resultStart": 0,
        "resultLimit": 1,
        "fields": [
            "animalID", "animalOrgID", "animalName",
            "animalSpecies", "animalBreed",
            "animalThumbnailUrl", "animalUrl",
            "locationName", "locationCity", "locationState",
            "locationPostalcode", "locationPhone", "locationUrl",
            "animalLocationCitystate", "animalLocationDistance",
        ],
        "filters": [
            {"fieldName": "animalStatus",           "operation": "equals", "criteria": "Available"},
            {"fieldName": "animalLocation",         "operation": "equals", "criteria": "08817"},
            {"fieldName": "animalLocationDistance", "operation": "radius", "criteria": "25"},
        ],
    },
})

animals = data.get("data", {})
if animals:
    first_id, first = list(animals.items())[0]
    print(f"Animal ID: {first_id}")
    print(json.dumps(first, indent=2))
else:
    print("No animals")
    print(json.dumps(data, indent=2)[:500])

# ── Now try org lookup with the orgID ─────────────────────────────────────────
print("\n" + "─"*60)
print("Org lookup")
print("─"*60)

if animals:
    org_id = list(animals.values())[0].get("animalOrgID")
    print(f"animalOrgID: {org_id}")

    if org_id:
        org_data = post({
            "apikey": key,
            "objectType": "orgs",
            "objectAction": "publicSearch",
            "search": {
                "calcFoundRows": "No",
                "resultStart": 0,
                "resultLimit": 1,
                "fields": [
                    "orgID", "orgName", "orgCity", "orgState",
                    "orgPostalcode", "orgPhone", "orgEmail",
                    "orgUrl", "orgFacebookUrl",
                ],
                "filters": [
                    {"fieldName": "orgID", "operation": "equals", "criteria": org_id},
                ],
            },
        })
        print(json.dumps(org_data, indent=2)[:1000])