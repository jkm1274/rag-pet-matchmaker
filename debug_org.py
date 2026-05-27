"""
debug_org.py
Tests different RescueGroups API include combinations to diagnose
why org/location data isn't coming through.

Run from project root:
    python debug_org.py
"""
import os, json, requests
from dotenv import load_dotenv
load_dotenv()

key = os.getenv("RESCUEGROUPS_API_KEY")
if not key:
    print("❌ RESCUEGROUPS_API_KEY not set in .env")
    exit(1)

headers = {"Content-Type": "application/vnd.api+json", "Authorization": key}
body = {"data": {"filterRadius": {"postalcode": "08817", "miles": 25}}}

print("=" * 60)
print("  Testing include[] combinations")
print("=" * 60)

tests = [
    ("orgs only",        ["orgs"]),
    ("locations only",   ["locations"]),
    ("pictures only",    ["pictures"]),
    ("breeds only",      ["breeds"]),
    ("species only",     ["species"]),
    ("orgs + locations", ["orgs", "locations"]),
    ("all",              ["orgs", "locations", "pictures", "breeds", "species"]),
]

for label, includes in tests:
    params = {"limit": 2, "page": 1, "include[]": includes}
    resp = requests.post(
        "https://api.rescuegroups.org/v5/public/animals/search/available/",
        headers=headers, json=body, params=params, timeout=20
    )
    data = resp.json()
    included = data.get("included", [])
    types = list(set(i.get("type") for i in included))
    status = "✅" if types else "⚠️ "
    print(f"{status} {label:30} → {types or 'EMPTY'} ({len(included)} items)")

print("\n" + "=" * 60)
print("  Testing direct org fetch")
print("=" * 60)

for org_id in ["3077", "1000003077"]:
    resp = requests.get(
        f"https://api.rescuegroups.org/v5/public/orgs/{org_id}",
        headers=headers, timeout=15
    )
    if resp.ok:
        attrs = resp.json().get("data", {}).get("attributes", {})
        print(f"✅ GET /public/orgs/{org_id} → 200")
        print(f"   name={attrs.get('name')} | city={attrs.get('city')} | phone={attrs.get('phone')}")
    else:
        print(f"❌ GET /public/orgs/{org_id} → {resp.status_code}")

print("\n" + "=" * 60)
print("  Testing public orgs search")
print("=" * 60)

resp = requests.post(
    "https://api.rescuegroups.org/v5/public/orgs/search/",
    headers=headers,
    json={"data": {"filterRadius": {"postalcode": "08817", "miles": 25}}},
    params={"limit": 3},
    timeout=20
)
if resp.ok:
    data = resp.json()
    orgs = data.get("data", [])
    print(f"✅ Org search returned {len(orgs)} orgs")
    for org in orgs:
        attrs = org.get("attributes", {})
        print(f"   {attrs.get('name')} | {attrs.get('city')}, {attrs.get('state')} | {attrs.get('phone')} | {attrs.get('url')}")
else:
    print(f"❌ Org search → {resp.status_code}: {resp.text[:200]}")