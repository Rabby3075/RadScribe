from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

KB_DIR = PROJECT_ROOT / "data" / "kb"
RAW_DIR = KB_DIR / "raw"
PROCESSED_DIR = KB_DIR / "processed"
INDEX_DIR = KB_DIR / "chroma"
SOURCES_PATH = KB_DIR / "sources.csv"
CHUNKS_PATH = PROCESSED_DIR / "chunks.parquet"
EVAL_QUERIES_PATH = KB_DIR / "eval_queries.csv"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "rag"

CHROMA_COLLECTION = "radscribe_kb_openai"
LOCAL_CHROMA_COLLECTION = "radscribe_kb_local"
EMBEDDING_MODEL = "text-embedding-3-small"
LOCAL_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
APPROVED_LICENCES = {"Public-Domain", "CC-BY-4.0", "CC-BY-SA-4.0"}
