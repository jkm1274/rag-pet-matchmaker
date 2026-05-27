"""
data/generate_rg_synthetic.py

Generates 30 synthetic shelter pets in the exact RescueGroups v5 API
response shape — same field names, same nested structure, same types.

The transformer, ingest pipeline, and app work identically with this
data as they will with real API responses. When your RescueGroups key
arrives, simply run sync.py and the real data replaces this seamlessly.

Run:
    python data/generate_rg_synthetic.py

Output:
    data/rg_synthetic.json   (30 animals in RescueGroups API shape)
"""

import json
import random
from pathlib import Path

random.seed(42)  # reproducible output

# ── Reference data ─────────────────────────────────────────────────────────────

NJ_SHELTERS = [
    {"id": "ORG001", "name": "Middlesex County Animal Shelter",
     "city": "Milltown", "state": "NJ", "postalcode": "08850",
     "url": "https://www.co.middlesex.nj.us/animalcontrol"},
    {"id": "ORG002", "name": "Edison Animal Shelter",
     "city": "Edison", "state": "NJ", "postalcode": "08817",
     "url": "https://www.edisontnj.org/animal-shelter"},
    {"id": "ORG003", "name": "Old Bridge Animal Shelter",
     "city": "Old Bridge", "state": "NJ", "postalcode": "08857",
     "url": "https://www.oldbridge.com/animal-control"},
    {"id": "ORG004", "name": "SAVE Animal Rescue",
     "city": "Princeton", "state": "NJ", "postalcode": "08540",
     "url": "https://saverescue.org"},
    {"id": "ORG005", "name": "Paws 4 A Cure Rescue",
     "city": "Piscataway", "state": "NJ", "postalcode": "08854",
     "url": "https://paws4acure.org"},
]

DOG_BREEDS   = ["Labrador Retriever", "German Shepherd", "Golden Retriever",
                "Beagle", "Bulldog", "Poodle", "Rottweiler", "Boxer",
                "Chihuahua", "Dachshund", "Shih Tzu", "Greyhound",
                "Border Collie", "Australian Shepherd", "Basset Hound"]

CAT_BREEDS   = ["Domestic Shorthair", "Domestic Longhair", "Siamese",
                "Maine Coon", "Ragdoll", "Bengal", "British Shorthair",
                "Russian Blue", "Persian", "Tabby"]

RABBIT_BREEDS = ["Holland Lop", "Mini Rex", "Lionhead", "Dutch"]

AGE_GROUPS   = ["Baby", "Young", "Adult", "Senior"]
SIZE_GROUPS  = ["Small", "Medium", "Large", "Extra Large"]
ENERGY       = ["Low", "Medium", "High", "Very High"]
ACTIVITY     = ["Couch Potato", "Low Activity", "Moderate Activity",
                "High Activity", "Very High Activity"]
EXERCISE     = ["Minimal", "Low", "Moderate", "High", "Very High"]
VOCAL        = ["Quiet", "Some", "Vocal", "Very Vocal"]
SHEDDING     = ["None", "Minimal", "Moderate", "High"]
GROOMING     = ["Minimal", "Low", "Moderate", "High"]
COAT         = ["Short", "Medium", "Long", "Hairless", "Curly", "Wavy"]
FENCE        = ["Not required", "Any", "4 feet", "5 feet", "6 feet"]
EXPERIENCE   = ["None", "Some", "Experienced"]
TRAINING     = ["None", "Basic", "Intermediate", "Advanced"]
INDOOR       = ["Indoor Only", "Indoor/Outdoor", "Outdoor Only"]
COLORS       = ["Black", "White", "Brown", "Tan", "Gray", "Orange",
                "Cream", "Brindle", "Merle", "Spotted", "Tabby"]

DESCRIPTIONS = [
    "{name} came to us as a stray and has blossomed into a true sweetheart. "
    "Staff describe them as one of the most affectionate animals in the shelter. "
    "They love cuddle sessions and would thrive in a calm, loving home.",

    "{name} is a playful and curious companion who lights up every room. "
    "They've been with us since being surrendered by a family who moved abroad. "
    "Despite the upheaval, {name} remains trusting and eager to please.",

    "Don't let {name}'s shy start fool you — once they warm up, there's no "
    "stopping the love they have to give. A quiet household would help "
    "{name} come out of their shell at their own pace.",

    "{name} arrived as part of a rescue from an overcrowded shelter downstate. "
    "They are fully vetted, social with humans, and ready for their forever home. "
    "Shelter volunteers say {name} is one of the most rewarding animals to work with.",

    "{name} is an energetic and intelligent animal who needs an owner ready to "
    "invest in training and enrichment. In return, {name} offers unwavering loyalty "
    "and an eagerness to learn that experienced owners will love.",

    "Gentle, quiet, and endlessly patient — that's {name} in three words. "
    "They've lived with volunteers for the past month and been wonderful "
    "with everyone they've met. A true hidden gem waiting for the right family.",

    "{name} was surrendered when their owner entered a care facility. "
    "They miss having a person to follow around and would do best with someone "
    "home during the day. {name} gives the best morning greetings.",

    "{name} is a senior who still has plenty of life left. They move at an "
    "unhurried pace, enjoy gentle play, and ask for very little beyond a warm "
    "spot and someone who appreciates quiet companionship.",

    "Young and full of promise, {name} is still learning the ropes. They pick "
    "up new skills quickly and respond beautifully to positive reinforcement. "
    "The right owner will find {name} to be an incredibly rewarding companion.",

    "{name} has been with us longer than we'd like — not because anything is "
    "wrong, but because the right person hasn't found them yet. If you're reading "
    "this, maybe that person is you.",
]


def _rb(true_weight: float = 0.6) -> bool:
    """Random bool with configurable probability of True."""
    return random.random() < true_weight


def _maybe(value, probability: float = 0.8):
    """Return value with given probability, else None."""
    return value if random.random() < probability else None


def make_animal(idx: int) -> dict:
    """Generate one synthetic animal in RescueGroups v5 API shape."""
    species_roll = random.random()
    if species_roll < 0.50:
        species = "Dog"
        breed   = random.choice(DOG_BREEDS)
    elif species_roll < 0.85:
        species = "Cat"
        breed   = random.choice(CAT_BREEDS)
    else:
        species = "Rabbit"
        breed   = random.choice(RABBIT_BREEDS)

    org    = random.choice(NJ_SHELTERS)
    age    = random.choice(AGE_GROUPS)
    sex    = random.choice(["Male", "Female"])
    size   = random.choice(SIZE_GROUPS)
    energy = random.choice(ENERGY)
    mixed  = _rb(0.45)
    color  = random.choice(COLORS)

    is_senior      = age == "Senior"
    is_young       = age in ["Baby", "Young"]
    high_energy    = energy in ["High", "Very High"]

    # Realistic compatibility logic
    is_kids_ok     = _maybe(_rb(0.65 if not high_energy else 0.45))
    is_dogs_ok     = _maybe(_rb(0.55 if species == "Dog" else 0.40))
    is_cats_ok     = _maybe(_rb(0.60 if species == "Cat" else 0.45))
    is_seniors_ok  = _maybe(_rb(0.70 if not high_energy else 0.35))
    is_yard_req    = _maybe(_rb(0.55 if species == "Dog" and high_energy else 0.25))

    name_pool = [
        "Biscuit", "Maple", "Ranger", "Cleo", "Duke", "Nala", "Jasper",
        "Willow", "Titan", "Rosie", "Archie", "Zara", "Bruno", "Lily",
        "Chester", "Ivy", "Rex", "Penny", "Gus", "Stella", "Finn",
        "Ruby", "Max", "Daisy", "Leo", "Coco", "Bear", "Lola", "Milo",
        "Sandy", "Oscar", "Bella", "Charlie", "Luna", "Rocky", "Molly",
    ]
    name = name_pool[idx % len(name_pool)]

    desc_template = random.choice(DESCRIPTIONS)
    description   = desc_template.format(name=name)

    animal_id = f"RG{10000 + idx}"

    return {
        # ── Core identity ──────────────────────────────────────────────────
        "_id":            animal_id,
        "name":           name,
        "_species":       species,
        "_breed_primary": breed,
        "_breed_secondary": random.choice(DOG_BREEDS) if mixed and species == "Dog" else None,
        "isBreedMixed":   mixed,
        "colorPrimary":   color,

        # ── Demographics ───────────────────────────────────────────────────
        "ageGroup":       age,
        "sizeGroup":      size,
        "sex":            sex,

        # ── Energy & exercise ──────────────────────────────────────────────
        "energyLevel":    energy,
        "activityLevel":  random.choice(ACTIVITY),
        "exerciseNeeds":  random.choice(EXERCISE),

        # ── Living situation ───────────────────────────────────────────────
        "isYardRequired": is_yard_req,
        "fenceNeeds":     random.choice(FENCE) if is_yard_req else "Not required",
        "indoorOutdoor":  random.choice(INDOOR),

        # ── Compatibility ──────────────────────────────────────────────────
        "isKidsOk":       is_kids_ok,
        "isDogsOk":       is_dogs_ok,
        "isCatsOk":       is_cats_ok,
        "isSeniorsOk":    is_seniors_ok,

        # ── Health & care ──────────────────────────────────────────────────
        "isAltered":             _rb(0.80),
        "isCurrentVaccinations": _rb(0.85),
        "isMicrochipped":        _rb(0.70),
        "isHousetrained":        _rb(0.55 if is_young else 0.80),
        "isSpecialNeeds":        _rb(0.12),

        # ── Training ───────────────────────────────────────────────────────
        "obedienceTraining": random.choice(TRAINING),
        "ownerExperience":   random.choice(EXPERIENCE),

        # ── Appearance & care ──────────────────────────────────────────────
        "coatLength":    random.choice(COAT),
        "vocalLevel":    random.choice(VOCAL),
        "sheddingLevel": random.choice(SHEDDING),
        "groomingNeeds": random.choice(GROOMING),

        # ── Description ────────────────────────────────────────────────────
        "descriptionText": description,

        # ── URLs ───────────────────────────────────────────────────────────
        "url": f"https://rescuegroups.org/animals/{animal_id}",

        # ── Photos (synthetic placeholder — no real image URLs) ────────────
        "_photos": [],

        # ── Org (shelter) ──────────────────────────────────────────────────
        "_org":    org,
        "_org_id": org["id"],

        # ── RescueGroups internal ──────────────────────────────────────────
        "status":   "Available",
        "priority": _rb(0.15),          # ~15% are priority/urgent
        "isNeedingFoster": _rb(0.10),   # ~10% need foster
    }


def generate(n: int = 30, output_path: str = "data/rg_synthetic.json") -> None:
    animals = [make_animal(i) for i in range(n)]
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(animals, f, indent=2, default=str)
    print(f"Generated {n} synthetic RescueGroups animals → {output_path}")

    # Quick summary
    species_counts = {}
    for a in animals:
        s = a["_species"]
        species_counts[s] = species_counts.get(s, 0) + 1
    for s, c in sorted(species_counts.items()):
        print(f"  {s}: {c}")
    priority = sum(1 for a in animals if a.get("priority"))
    print(f"  Priority/urgent: {priority}")


if __name__ == "__main__":
    generate(30)