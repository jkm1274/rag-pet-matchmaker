"""
src/ingestion/ingest.py
Loads the shelter pets CSV, builds LangChain Documents, and stores
embeddings in ChromaDB. Run this once (or whenever data changes).

Usage:
    python -m src.ingestion.ingest
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings


DATA_PATH = Path(__file__).parents[2] / "data" / "shelter_pets.csv"
CHROMA_DIR = Path(__file__).parents[2] / ".chroma_db"
COLLECTION_NAME = "shelter_pets"


def load_pet_documents(csv_path: Path = DATA_PATH) -> List[Document]:
    """Read the CSV and return one Document per pet.

    The page_content contains the biography (what gets embedded).
    All structured fields go into metadata for filtering / display.
    """
    docs: List[Document] = []

    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            # Enrich the bio with structured context so the embedding
            # captures breed, size, energy, etc. alongside the narrative.
            content = (
                f"Name: {row['name']}. "
                f"Species: {row['species']}. "
                f"Breed: {row['breed']}. "
                f"Age: {row['age_years']} year(s). "
                f"Size: {row['size']}. "
                f"Energy level: {row['energy_level']}. "
                f"Good with kids: {row['good_with_kids']}. "
                f"Good with dogs: {row['good_with_dogs']}. "
                f"Good with cats: {row['good_with_cats']}. "
                f"Requires yard: {row['requires_yard']}. "
                f"Hypoallergenic: {row['hypoallergenic']}. "
                f"Bio: {row['bio']}"
            )

            metadata = {
                "id": row["id"],
                "name": row["name"],
                "species": row["species"],
                "breed": row["breed"],
                "age_years": int(row["age_years"]),
                "energy_level": row["energy_level"],
                "size": row["size"],
                "good_with_kids": row["good_with_kids"] == "True",
                "good_with_dogs": row["good_with_dogs"] == "True",
                "good_with_cats": row["good_with_cats"] == "True",
                "requires_yard": row["requires_yard"] == "True",
                "hypoallergenic": row["hypoallergenic"] == "True",
            }

            docs.append(Document(page_content=content, metadata=metadata))

    return docs


def build_vector_store(
    docs: List[Document],
    persist_dir: Path = CHROMA_DIR,
    collection: str = COLLECTION_NAME,
) -> Chroma:
    """Embed documents and persist them to ChromaDB."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    vector_store = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=collection,
        persist_directory=str(persist_dir),
    )

    print(f"Indexed {len(docs)} pet profiles into ChromaDB at {persist_dir}")
    return vector_store


def load_vector_store(
    persist_dir: Path = CHROMA_DIR,
    collection: str = COLLECTION_NAME,
) -> Chroma:
    """Load an existing ChromaDB collection."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return Chroma(
        collection_name=collection,
        embedding_function=embeddings,
        persist_directory=str(persist_dir),
    )


if __name__ == "__main__":
    documents = load_pet_documents()
    print(f"Loaded {len(documents)} documents from CSV.")
    build_vector_store(documents)