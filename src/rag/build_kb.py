from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import pandas as pd

from src.rag.common import (
    APPROVED_LICENCES,
    CHROMA_COLLECTION,
    CHUNKS_PATH,
    EMBEDDING_MODEL,
    INDEX_DIR,
    KB_DIR,
    LOCAL_CHROMA_COLLECTION,
    LOCAL_EMBEDDING_MODEL,
    PROCESSED_DIR,
    PROJECT_ROOT,
    SOURCES_PATH,
)


REQUIRED_SOURCE_COLUMNS = {
    "id",
    "finding",
    "title",
    "url",
    "source_name",
    "licence",
    "licence_url",
    "raw_text_path",
    "accessed_date",
    "licence_ok",
    "notes",
}


def project_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_sources(path: Path = SOURCES_PATH) -> pd.DataFrame:
    sources = pd.read_csv(path)
    missing = REQUIRED_SOURCE_COLUMNS - set(sources.columns)
    if missing:
        raise ValueError(f"Missing source columns: {sorted(missing)}")

    bad_licences = sorted(set(sources["licence"]) - APPROVED_LICENCES)
    if bad_licences:
        raise ValueError(f"Unapproved licences found: {bad_licences}")

    if not sources["licence_ok"].astype(bool).all():
        raise ValueError("Every source must have licence_ok=True before building the KB.")

    missing_files = [
        str(path)
        for path in sources["raw_text_path"].unique()
        if not project_path(path).exists()
    ]
    if missing_files:
        raise FileNotFoundError(f"Missing raw KB files: {missing_files}")

    return sources


def clean_doc_text(text: str) -> str:
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            continue
        # Attribution belongs in metadata, not in embedded retrieval text.
        if stripped.lower().startswith("sources:") or stripped.lower().startswith("source:"):
            continue
        if "rewritten for this project" in stripped.lower():
            continue
        lines.append(stripped)
    return " ".join(lines)


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    return [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", text) if sentence.strip()]


def safe_chunk_prefix(finding: str) -> str:
    return (
        finding.lower()
        .replace(" / ", "_")
        .replace("/", "_")
        .replace(" ", "_")
        .replace("-", "_")
    )


def make_chunks(
    sources: pd.DataFrame,
    sentences_per_chunk: int = 4,
    overlap: int = 1,
) -> pd.DataFrame:
    if overlap >= sentences_per_chunk:
        raise ValueError("overlap must be smaller than sentences_per_chunk")

    rows = []
    grouped = sources.groupby("raw_text_path", sort=True)
    for raw_text_path, source_rows in grouped:
        findings = source_rows["finding"].drop_duplicates().tolist()
        if len(findings) != 1:
            raise ValueError(f"{raw_text_path} maps to multiple findings: {findings}")

        finding = findings[0]
        text = clean_doc_text(project_path(raw_text_path).read_text(encoding="utf-8"))
        sentences = split_sentences(text)
        step = sentences_per_chunk - overlap
        prefix = safe_chunk_prefix(finding)

        for chunk_number, start in enumerate(range(0, len(sentences), step)):
            chunk_sentences = sentences[start : start + sentences_per_chunk]
            if not chunk_sentences:
                continue
            chunk_text = " ".join(chunk_sentences).strip()
            if not chunk_text:
                continue

            rows.append(
                {
                    "chunk_id": f"{prefix}_{chunk_number:03d}",
                    "finding": finding,
                    "text": chunk_text,
                    "raw_text_path": raw_text_path,
                    "source_ids": json.dumps(source_rows["id"].tolist()),
                    "source_titles": json.dumps(source_rows["title"].tolist()),
                    "source_urls": json.dumps(source_rows["url"].tolist()),
                    "licences": json.dumps(source_rows["licence"].tolist()),
                    "source_names": json.dumps(source_rows["source_name"].tolist()),
                }
            )

    chunks = pd.DataFrame(rows)
    if chunks.empty:
        raise ValueError("No chunks were created.")
    if chunks["text"].str.strip().eq("").any():
        raise ValueError("Empty chunk text found.")
    if chunks["chunk_id"].duplicated().any():
        raise ValueError("Duplicate chunk_id values found.")
    return chunks


def embed_openai(texts: list[str], model: str = EMBEDDING_MODEL) -> list[list[float]]:
    try:
        from dotenv import load_dotenv

        load_dotenv(PROJECT_ROOT / ".env")
    except ModuleNotFoundError:
        pass

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set. Add it to .env or set it in your terminal.")

    from openai import OpenAI

    client = OpenAI()
    response = client.embeddings.create(model=model, input=texts)
    return [item.embedding for item in response.data]


def embed_local(texts: list[str], model: str = LOCAL_EMBEDDING_MODEL) -> list[list[float]]:
    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer(model)
    embeddings = encoder.encode(
        texts,
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    return embeddings.tolist()


def build_chroma_index(chunks: pd.DataFrame, provider: str = "openai") -> int:
    import chromadb

    if provider == "openai":
        collection_name = CHROMA_COLLECTION
        embeddings = embed_openai(chunks["text"].tolist())
    elif provider == "local":
        collection_name = LOCAL_CHROMA_COLLECTION
        embeddings = embed_local(chunks["text"].tolist())
    else:
        raise ValueError(f"Unknown provider: {provider}")

    client = chromadb.PersistentClient(path=str(INDEX_DIR))
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass

    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    metadatas = []
    for row in chunks.to_dict("records"):
        metadatas.append(
            {
                "finding": row["finding"],
                "raw_text_path": row["raw_text_path"],
                "source_ids": row["source_ids"],
                "source_titles": row["source_titles"],
                "source_urls": row["source_urls"],
                "licences": row["licences"],
                "source_names": row["source_names"],
            }
        )

    collection.add(
        ids=chunks["chunk_id"].tolist(),
        documents=chunks["text"].tolist(),
        embeddings=embeddings,
        metadatas=metadatas,
    )
    return collection.count()


def build_kb(rebuild_index: bool = True, provider: str = "openai") -> pd.DataFrame:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    sources = load_sources()
    chunks = make_chunks(sources)
    chunks.to_parquet(CHUNKS_PATH, index=False)

    if rebuild_index:
        providers = ["openai", "local"] if provider == "both" else [provider]
        for current_provider in providers:
            build_chroma_index(chunks, provider=current_provider)

    return chunks


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the RadScribe retrieval knowledge base.")
    parser.add_argument(
        "--chunks-only",
        action="store_true",
        help="Only write chunks.parquet; do not call OpenAI or rebuild Chroma.",
    )
    parser.add_argument(
        "--provider",
        choices=["openai", "local", "both"],
        default="openai",
        help="Which embedding index to build.",
    )
    args = parser.parse_args()

    chunks = build_kb(rebuild_index=not args.chunks_only, provider=args.provider)
    print(f"chunks: {len(chunks)}")
    print(f"saved: {CHUNKS_PATH}")
    if args.chunks_only:
        print("index: skipped")
    else:
        print(f"provider: {args.provider}")


if __name__ == "__main__":
    main()
