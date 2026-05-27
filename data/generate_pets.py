"""
generate_pets.py
Generates a mock CSV dataset of 15 shelter pets with rich biographies.
Run: python data/generate_pets.py
"""

import csv
import os

PETS = [
    {
        "id": "PET001",
        "name": "Biscuit",
        "species": "Dog",
        "breed": "Golden Retriever Mix",
        "age_years": 3,
        "energy_level": "High",
        "size": "Large",
        "good_with_kids": True,
        "good_with_dogs": True,
        "good_with_cats": False,
        "requires_yard": True,
        "hypoallergenic": False,
        "bio": (
            "Biscuit is an exuberant three-year-old Golden Retriever mix who lives life at full volume. "
            "He wakes up ready to run, play fetch, and explore every trail he can find. Biscuit would "
            "thrive in an active household — ideally with a fenced yard where he can burn off his boundless "
            "energy. He loves children and will happily follow a toddler around all day, but his enthusiasm "
            "can be overwhelming for cats. He has completed basic obedience training and knows sit, stay, and "
            "come reliably. Biscuit's ideal owner is a runner, hiker, or outdoor enthusiast who wants a "
            "four-legged adventure partner. He does not do well in apartments and gets anxious without "
            "significant daily exercise. In return, he offers unconditional loyalty and a wagging tail "
            "that never seems to stop."
        ),
    },
    {
        "id": "PET002",
        "name": "Luna",
        "species": "Cat",
        "breed": "Domestic Shorthair",
        "age_years": 5,
        "energy_level": "Low",
        "size": "Medium",
        "good_with_kids": True,
        "good_with_dogs": False,
        "good_with_cats": True,
        "requires_yard": False,
        "hypoallergenic": False,
        "bio": (
            "Luna is a serene five-year-old tortoiseshell cat who was born to be a lap companion. "
            "She spends most of her day finding the warmest patch of sunlight in the room and napping "
            "with distinguished grace. Luna is the perfect pet for someone who works from home or "
            "enjoys quiet evenings — she will sit beside you for hours without demanding attention, "
            "yet purrs loudly the moment you reach out to stroke her. She is gentle with older children "
            "who respect her boundaries but dislikes dogs and can become stressed in chaotic environments. "
            "Luna is fully litter-trained, up to date on vaccinations, and spayed. She would flourish "
            "in a calm apartment or house with at least one other mellow cat for company, or as an "
            "only pet with a patient, gentle owner."
        ),
    },
    {
        "id": "PET003",
        "name": "Mango",
        "species": "Dog",
        "breed": "Chihuahua",
        "age_years": 2,
        "energy_level": "Medium",
        "size": "Small",
        "good_with_kids": False,
        "good_with_dogs": True,
        "good_with_cats": True,
        "requires_yard": False,
        "hypoallergenic": False,
        "bio": (
            "Mango is a spirited two-year-old Chihuahua with a personality three times his body size. "
            "He is alert, curious, and surprisingly adaptable — happy in a studio apartment as long as "
            "he gets a couple of 20-minute walks each day. Mango has formed strong bonds with other "
            "small dogs and tolerates cats well, but he is not recommended for homes with young children "
            "who might accidentally hurt him or trigger his defensive snapping. He loves to burrow under "
            "blankets, ride in tote bags, and survey his kingdom from elevated perches. Despite his bold "
            "exterior, Mango is deeply affectionate with his chosen person and will follow you from room "
            "to room with devoted loyalty. He is neutered, microchipped, and house-trained. Ideal for "
            "a single professional or couple living in an urban environment."
        ),
    },
    {
        "id": "PET004",
        "name": "Oreo",
        "species": "Dog",
        "breed": "Border Collie Mix",
        "age_years": 4,
        "energy_level": "Very High",
        "size": "Medium",
        "good_with_kids": True,
        "good_with_dogs": True,
        "good_with_cats": False,
        "requires_yard": True,
        "hypoallergenic": False,
        "bio": (
            "Oreo is a four-year-old Border Collie mix whose brain never shuts off. He needs not just "
            "physical exercise but serious mental stimulation — puzzle toys, agility courses, scent work, "
            "or learning new tricks on a daily basis. Without this engagement, he becomes restless and "
            "may redirect his energy into destructive behavior. Oreo is an outstanding candidate for "
            "competitive dog sports or an owner committed to structured training. He herds anything that "
            "moves, including cats, so a cat-free home is essential. With kids who can keep up with his "
            "energy, he is a playful and protective companion. Oreo is one of the most intelligent dogs "
            "in our shelter and will reward a dedicated owner with spectacular obedience and an unbreakable bond."
        ),
    },
    {
        "id": "PET005",
        "name": "Cleo",
        "species": "Cat",
        "breed": "Siamese Mix",
        "age_years": 7,
        "energy_level": "Medium",
        "size": "Medium",
        "good_with_kids": True,
        "good_with_dogs": True,
        "good_with_cats": True,
        "requires_yard": False,
        "hypoallergenic": False,
        "bio": (
            "Cleo is a seven-year-old Siamese mix who is simultaneously the most vocal and most charming "
            "cat you will ever meet. She announces her opinions loudly, greets guests at the door, and "
            "insists on being part of every family activity. Unlike many cats, Cleo genuinely enjoys "
            "the company of calm dogs and gets along beautifully with other cats. She is the ideal pet "
            "for a family looking for an interactive, engaging companion rather than an aloof observer. "
            "Cleo learned to walk on a harness and enjoys supervised outdoor exploration. She does best "
            "in an environment where someone is home frequently — prolonged alone time makes her vocal "
            "and anxious. She is spayed, vaccinated, and has a clean bill of health from our veterinarian."
        ),
    },
    {
        "id": "PET006",
        "name": "Winston",
        "species": "Dog",
        "breed": "Basset Hound",
        "age_years": 6,
        "energy_level": "Low",
        "size": "Large",
        "good_with_kids": True,
        "good_with_dogs": True,
        "good_with_cats": True,
        "requires_yard": False,
        "hypoallergenic": False,
        "bio": (
            "Winston is a six-year-old Basset Hound who has perfected the art of the dignified nap. "
            "He requires two gentle daily walks — enough to keep him healthy — and spends the rest of "
            "his time lounging with aristocratic contentment. Winston is remarkably tolerant: he accepts "
            "children tugging his ears with patient grace, coexists happily with cats, and welcomes "
            "other dogs as napping partners. His soulful eyes and velvet ears make him irresistible to "
            "anyone who meets him. Winston is an outstanding choice for first-time dog owners, seniors, "
            "or anyone who values a calm companion over an athletic one. He adapts well to apartment "
            "living provided he gets his daily constitutional. Fair warning: his baying howl is operatic "
            "in volume, so thin-walled apartments may not be the ideal setting."
        ),
    },
    {
        "id": "PET007",
        "name": "Pepper",
        "species": "Cat",
        "breed": "Maine Coon Mix",
        "age_years": 3,
        "energy_level": "Medium",
        "size": "Large",
        "good_with_kids": True,
        "good_with_dogs": True,
        "good_with_cats": True,
        "requires_yard": False,
        "hypoallergenic": False,
        "bio": (
            "Pepper is a magnificent three-year-old Maine Coon mix with a luxurious tufted coat and the "
            "temperament of a gentle giant. He weighs 14 pounds and has the confident, dog-like personality "
            "that Maine Coons are famous for — he fetches crinkle balls, comes when called, and will "
            "supervise your cooking from a respectful distance. Pepper is exceptional with children, "
            "patient with dogs, and sociable with other cats. He needs daily interactive play to stay "
            "mentally engaged but is equally content to spend an evening on a lap. His coat requires "
            "weekly brushing to prevent matting. Pepper is an excellent choice for a family or active "
            "individual who wants a cat that acts more like a dog in the best possible way."
        ),
    },
    {
        "id": "PET008",
        "name": "Daisy",
        "species": "Dog",
        "breed": "Cavalier King Charles Spaniel",
        "age_years": 4,
        "energy_level": "Low",
        "size": "Small",
        "good_with_kids": True,
        "good_with_dogs": True,
        "good_with_cats": True,
        "requires_yard": False,
        "hypoallergenic": False,
        "bio": (
            "Daisy is a four-year-old Cavalier King Charles Spaniel who was quite literally bred to be "
            "a companion dog, and she takes that role seriously. She thrives on human contact, adapts "
            "effortlessly to the pace of her household, and is equally happy on a leisurely walk or "
            "curled up on the sofa watching television. Daisy is exceptionally gentle — she has lived "
            "successfully with a toddler, two cats, and another dog and remained unfailingly sweet "
            "throughout. She is an outstanding match for retirees, remote workers, or families with "
            "young children who want a calm, affectionate dog. Daisy does experience separation anxiety "
            "and should not be left alone for more than four hours. She is spayed, vaccinated, and "
            "up to date on all preventative care."
        ),
    },
    {
        "id": "PET009",
        "name": "Ziggy",
        "species": "Dog",
        "breed": "Jack Russell Terrier Mix",
        "age_years": 2,
        "energy_level": "Very High",
        "size": "Small",
        "good_with_kids": True,
        "good_with_dogs": False,
        "good_with_cats": False,
        "requires_yard": True,
        "hypoallergenic": False,
        "bio": (
            "Ziggy is a two-year-old Jack Russell Terrier mix who is essentially a compressed ball of "
            "pure kinetic energy. Despite his small size, he needs vigorous daily exercise — think "
            "long runs, fetch marathons, or agility training rather than a stroll around the block. "
            "He has a strong prey drive and will chase cats relentlessly, and he can be reactive with "
            "other dogs. Ziggy does best as an only pet in a home with active adults or older kids "
            "who can match his intensity. He is whip-smart, learns commands quickly, and loves "
            "obstacle courses. Ziggy's foster family describes him as 'exhausting and absolutely worth it.' "
            "He needs a securely fenced yard and an owner who will channel his energy into productive "
            "activities. In the right home, he is endlessly entertaining and fiercely loyal."
        ),
    },
    {
        "id": "PET010",
        "name": "Mocha",
        "species": "Cat",
        "breed": "Ragdoll Mix",
        "age_years": 2,
        "energy_level": "Low",
        "size": "Large",
        "good_with_kids": True,
        "good_with_dogs": True,
        "good_with_cats": True,
        "requires_yard": False,
        "hypoallergenic": False,
        "bio": (
            "Mocha is a two-year-old Ragdoll mix who fully embodies her breed's reputation for going "
            "limp with contentment when picked up. She is a deeply relaxed cat who meets the world "
            "with unhurried calm. Mocha actively seeks out laps, greets strangers with curiosity "
            "rather than fear, and has never shown aggression toward children, dogs, or other cats. "
            "She is ideal for a busy family that wants a low-maintenance but warm and loving companion. "
            "Mocha is not demanding — she will not wake you at dawn or knock items off shelves for "
            "attention — but she will find you every evening for her scheduled cuddle session. Her "
            "semi-long coat needs brushing twice a week. She is spayed, microchipped, and has received "
            "all recommended vaccinations."
        ),
    },
    {
        "id": "PET011",
        "name": "Atlas",
        "species": "Dog",
        "breed": "German Shepherd Mix",
        "age_years": 5,
        "energy_level": "High",
        "size": "Large",
        "good_with_kids": True,
        "good_with_dogs": False,
        "good_with_cats": False,
        "requires_yard": True,
        "hypoallergenic": False,
        "bio": (
            "Atlas is a five-year-old German Shepherd mix with the bearing of a dog who takes his "
            "responsibilities seriously. He is confident, protective, and deeply bonded to his chosen "
            "humans. Atlas was a working dog in a previous life and retains that focused, task-oriented "
            "temperament — he excels at advanced obedience, tracking, and protective work. He does not "
            "get along well with other dogs or cats and should be the sole pet in the household. Atlas "
            "requires significant daily exercise and mental engagement, and he needs an experienced "
            "owner who can provide clear, consistent leadership. In return, he offers unmatched loyalty "
            "and a calm, steady presence that makes him an exceptional family guardian. He is wonderful "
            "with children in his family once he has accepted them as part of his pack."
        ),
    },
    {
        "id": "PET012",
        "name": "Hazel",
        "species": "Dog",
        "breed": "Greyhound",
        "age_years": 4,
        "energy_level": "Low",
        "size": "Large",
        "good_with_kids": True,
        "good_with_dogs": True,
        "good_with_cats": False,
        "requires_yard": False,
        "hypoallergenic": False,
        "bio": (
            "Hazel is a four-year-old retired racing Greyhound who has discovered that life off the "
            "track is very, very comfortable. Contrary to popular belief, Greyhounds are couch potatoes "
            "who need only a couple of short walks per day. Hazel spends approximately 18 hours a day "
            "asleep on the softest available surface and is an ideal apartment dog despite her size. "
            "She is gentle, quiet, and surprisingly sensitive — she responds beautifully to soft voices "
            "and positive reinforcement. Hazel has a strong instinct to chase small animals, so cats "
            "are not safe in her home, but she coexists gracefully with other calm dogs. She would "
            "flourish with a gentle family or individual who appreciates a dignified, unhurried companion "
            "who asks for very little and gives quiet affection in return."
        ),
    },
    {
        "id": "PET013",
        "name": "Pebbles",
        "species": "Cat",
        "breed": "British Shorthair",
        "age_years": 6,
        "energy_level": "Low",
        "size": "Medium",
        "good_with_kids": True,
        "good_with_dogs": False,
        "good_with_cats": True,
        "requires_yard": False,
        "hypoallergenic": False,
        "bio": (
            "Pebbles is a six-year-old British Shorthair with a plush coat, round copper eyes, and the "
            "unflappable composure of someone who has seen everything twice. She is not a lap cat — "
            "she prefers to sit nearby rather than on top of you — but she is consistently present, "
            "watchful, and quietly affectionate. Pebbles would suit a home where her independence "
            "is respected: she does not like being picked up but will headbutt your hand for pets on "
            "her own terms. She tolerates children who understand cat body language and gets along with "
            "other calm cats, but dislikes dogs. Pebbles is a wonderful choice for someone who travels "
            "occasionally, works long hours, or simply wants a companion who won't demand constant "
            "attention. She is perfectly self-sufficient — just make sure her food bowl is always full."
        ),
    },
    {
        "id": "PET014",
        "name": "Finn",
        "species": "Dog",
        "breed": "Labrador Retriever",
        "age_years": 1,
        "energy_level": "Very High",
        "size": "Large",
        "good_with_kids": True,
        "good_with_dogs": True,
        "good_with_cats": True,
        "requires_yard": True,
        "hypoallergenic": False,
        "bio": (
            "Finn is a one-year-old Labrador Retriever who is, at this moment, the most enthusiastic "
            "living creature on this planet. Everything is the best thing that has ever happened to him: "
            "breakfast, walks, strangers, puddles, sticks, and especially you. Finn is in the full "
            "bloom of Lab puppyhood — all paws, all energy, all love. He needs substantial daily "
            "exercise, patient training, and a household that can handle an exuberant young dog who "
            "has not yet learned his own strength. He is excellent with kids, tolerant with cats, and "
            "joyfully social with other dogs. Finn has completed puppy kindergarten and knows basic "
            "commands, but he still needs consistent reinforcement. He would be an outstanding first "
            "dog for an active family committed to training. He promises to become the most "
            "loyal, joyful companion you have ever had — he just needs a year or two to find his calm."
        ),
    },
    {
        "id": "PET015",
        "name": "Sage",
        "species": "Cat",
        "breed": "Russian Blue Mix",
        "age_years": 4,
        "energy_level": "Medium",
        "size": "Small",
        "good_with_kids": False,
        "good_with_dogs": False,
        "good_with_cats": True,
        "requires_yard": False,
        "hypoallergenic": True,
        "bio": (
            "Sage is a four-year-old Russian Blue mix with a silvery coat and the reserved elegance "
            "that the breed is famous for. She is a one-person cat in the most devoted sense: once "
            "she has chosen you, she will follow you everywhere, sleep beside your pillow, and watch "
            "your every move with intelligent green eyes. With strangers, however, she is cautious "
            "and needs time to warm up. Sage is not suitable for children or dogs — she finds them "
            "too unpredictable — but she has lived harmoniously with another quiet, gentle cat. "
            "She is the ideal companion for a single adult or couple who want a deeply bonded, "
            "low-allergen cat. Her short plush coat sheds minimally and she produces lower levels "
            "of the Fel d1 allergen, making her a good option for mild cat allergy sufferers. "
            "Sage is spayed, vaccinated, and deeply ready to become someone's devoted shadow."
        ),
    },
]


def generate_csv(output_path: str) -> None:
    fieldnames = [
        "id", "name", "species", "breed", "age_years", "energy_level",
        "size", "good_with_kids", "good_with_dogs", "good_with_cats",
        "requires_yard", "hypoallergenic", "bio",
    ]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(PETS)

    print(f"Generated {len(PETS)} pet profiles → {output_path}")


if __name__ == "__main__":
    generate_csv("data/shelter_pets.csv")