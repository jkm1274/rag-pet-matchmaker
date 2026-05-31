"""
src/llm/chain.py
Builds the RAG chain: takes user input + retrieved context and returns
a personalised, conversational pet recommendation using an LLM.
"""

from __future__ import annotations
from typing import Dict, List, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

SYSTEM_PROMPT = """You are PawMatch, a knowledgeable and warm pet adoption counselor.
Your job is to read shelter pet profiles and explain clearly why each one is
or isn't a good fit for this specific person.

Rules:
- Address the person directly. Reference specific details they mentioned
  (apartment, work schedule, kids, other pets, allergies, energy preference).
- For each pet lead with ONE concrete reason it fits (or doesn't) — energy level,
  size, compatibility, temperament. Be specific, not generic.
- If the person applied filters (species, age, size), acknowledge them briefly.
- If the search location was provided, mention it naturally.
- Rank clearly: best match first, then briefly note the others.
- Do NOT use filler phrases like "I hope this helps", "Best of luck",
  "Happy adopting", "feel free to reach out", or "wonderful journey".
- Do NOT repeat the pet's name more than twice per paragraph.
- Keep the full response under 300 words.
- End with one practical next step (e.g. "Call the shelter to arrange a meet")."""

USER_PROMPT = """Adopter's lifestyle description:
"{user_query}"

Search context:
- Location: {location}
- Radius: {radius} miles
- Active filters: {filters}

Retrieved pet profiles:
{context}

Write a personalised recommendation. Start directly with the best match — no preamble."""


def build_chain(model: str = "gpt-4o-mini", temperature: float = 0.3):
    llm    = ChatOpenAI(model=model, temperature=temperature)
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human",  USER_PROMPT),
    ])
    return prompt | llm


def generate_recommendation(
    user_query: str,
    context: str,
    location: str = "Not specified",
    radius: int = 25,
    active_filters: Optional[Dict[str, List[str]]] = None,
    model: str = "gpt-4o-mini",
) -> str:
    """Run the chain and return the LLM recommendation."""

    # Build a human-readable filter summary
    if active_filters:
        parts = []
        species = active_filters.get("species", [])
        ages    = active_filters.get("ages",    [])
        sizes   = active_filters.get("sizes",   [])
        # Only mention filters that are actually restricting something
        if species and len(species) < 3:
            parts.append(f"species: {', '.join(species)}")
        if ages and len(ages) < 4:
            parts.append(f"age: {', '.join(ages)}")
        if sizes and len(sizes) < 4:
            parts.append(f"size: {', '.join(sizes)}")
        filters_str = ", ".join(parts) if parts else "none"
    else:
        filters_str = "none"

    chain = build_chain(model=model)
    response = chain.invoke({
        "user_query": user_query,
        "context":    context,
        "location":   location,
        "radius":     radius,
        "filters":    filters_str,
    })
    return response.content