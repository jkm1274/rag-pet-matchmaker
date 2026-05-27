"""
src/llm/chain.py
Builds the RAG chain: takes user input + retrieved context and returns
a personalised, conversational pet recommendation using an LLM.
"""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

SYSTEM_PROMPT = """You are PawMatch, a warm and knowledgeable pet adoption counselor.
Your job is to review shelter pet profiles retrieved from a database and explain —
in a friendly, conversational tone — why each pet is or isn't a good fit for the
person's lifestyle.

Rules:
- Address the person directly and personally.
- Mention specific details from their description to show you listened.
- For each pet, give one or two sentences explaining the match, referencing concrete
  traits (energy level, size, temperament).
- Rank them clearly: start with the best match.
- Be warm and encouraging — adoption is an emotional decision.
- Keep the response under 350 words."""

USER_PROMPT = """A potential adopter described their lifestyle as follows:

"{user_query}"

The following shelter pets were retrieved as potential matches:

{context}

Please write a personalised recommendation explaining which pet is the best match
and why, followed by brief notes on the second and third options."""


def build_chain(model: str = "gpt-4o-mini", temperature: float = 0.4):
    """Return a runnable LangChain chain."""
    llm = ChatOpenAI(model=model, temperature=temperature)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", USER_PROMPT),
        ]
    )

    return prompt | llm


def generate_recommendation(
    user_query: str,
    context: str,
    model: str = "gpt-4o-mini",
) -> str:
    """Run the chain and return the LLM's recommendation as a string."""
    chain = build_chain(model=model)
    response = chain.invoke({"user_query": user_query, "context": context})
    return response.content