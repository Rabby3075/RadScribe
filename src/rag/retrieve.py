from __future__ import annotations

import argparse
import json
import os
from typing import Any

from src.rag.common import (
    CHROMA_COLLECTION,
    EMBEDDING_MODEL,
    INDEX_DIR,
    LOCAL_CHROMA_COLLECTION,
    LOCAL_EMBEDDING_MODEL,
)


def load_env() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ModuleNotFoundError:
        pass


def embed_openai(texts: list[str], model: str = EMBEDDING_MODEL) -> list[list[float]]:
    from openai import OpenAI

    client = OpenAI()
    response = client.embeddings.create(model=model, input=texts)
    return [item.embedding for item in response.data]


def embed_local(texts: list[str], model: str = LOCAL_EMBEDDING_MODEL) -> list[list[float]]:
    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer(model)
    embeddings = encoder.encode(texts, batch_size=32, normalize_embeddings=True)
    return embeddings.tolist()


def parse_json_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if not isinstance(value, str):
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return [value]
    if isinstance(parsed, list):
        return [str(item) for item in parsed]
    return [str(parsed)]


def get_collection(provider: str = "openai"):
    import chromadb

    collection_name = CHROMA_COLLECTION if provider == "openai" else LOCAL_CHROMA_COLLECTION
    client = chromadb.PersistentClient(path=str(INDEX_DIR))
    return client.get_collection(collection_name)


def retrieve(query: str, k: int = 4, provider: str = "openai") -> list[dict[str, Any]]:
    """Return KB passages for a query.

    The score is cosine similarity from the Chroma cosine index, so higher is better.
    """
    if provider not in {"openai", "local"}:
        raise ValueError("provider must be 'openai' or 'local'")

    if provider == "openai":
        load_env()
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set.")
        query_embedding = embed_openai([query])[0]
    else:
        query_embedding = embed_local([query])[0]

    collection = get_collection(provider=provider)
    result = collection.query(query_embeddings=[query_embedding], n_results=k)

    rows = []
    for rank, i in enumerate(range(len(result["ids"][0])), start=1):
        metadata = result["metadatas"][0][i]
        distance = float(result["distances"][0][i])
        rows.append(
            {
                "rank": rank,
                "chunk_id": result["ids"][0][i],
                "text": result["documents"][0][i],
                "finding": metadata["finding"],
                "score": 1 - distance,
                "distance": distance,
                "provider": provider,
                "source_ids": parse_json_list(metadata.get("source_ids")),
                "source_titles": parse_json_list(metadata.get("source_titles")),
                "source_urls": parse_json_list(metadata.get("source_urls")),
                "licences": parse_json_list(metadata.get("licences")),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieve RadScribe KB passages.")
    parser.add_argument("query")
    parser.add_argument("-k", type=int, default=4)
    parser.add_argument("--provider", choices=["openai", "local"], default="openai")
    args = parser.parse_args()

    for row in retrieve(args.query, k=args.k, provider=args.provider):
        print(f"{row['rank']}. {row['finding']} score={row['score']:.3f} id={row['chunk_id']}")
        print(row["text"])
        print("sources:", "; ".join(row["source_titles"]))
        print()


if __name__ == "__main__":
    main()
