# Phase 3 Optional: Local MiniLM vs OpenAI Retrieval

This optional check compares two embedding models on the same RadScribe knowledge base and the same evaluation queries.

## Why This Exists

The main Phase 3 retriever uses OpenAI `text-embedding-3-small`.

This optional comparison adds a local baseline using `sentence-transformers/all-MiniLM-L6-v2`.

The goal is simple:

- check if OpenAI retrieval is better than a free local model
- keep the same chunks, same Chroma cosine setup, and same eval queries
- report the result honestly

## Models

| Provider | Embedding model | Notes |
| --- | --- | --- |
| OpenAI | `text-embedding-3-small` | Uses API key and costs a tiny amount |
| Local | `sentence-transformers/all-MiniLM-L6-v2` | Runs locally after the model is downloaded |

## Commands

Build the OpenAI index:

```bash
python -m src.rag.build_kb --provider openai
```

Build the local MiniLM index:

```bash
python -m src.rag.build_kb --provider local
```

Run both evaluations and save the comparison table:

```bash
python -m src.rag.evaluate --provider both
```

## Output Files

The comparison writes:

```text
outputs/rag/retrieval_provider_comparison.csv
outputs/rag/retrieval_metrics_openai.csv
outputs/rag/retrieval_metrics_local.csv
outputs/rag/retrieval_hits_openai.csv
outputs/rag/retrieval_hits_local.csv
```

## What To Report

Use this format after running the commands:

```text
OpenAI vs MiniLM retrieval comparison:
OpenAI recall@1 = ___, recall@3 = ___, MRR@3 = ___.
MiniLM recall@1 = ___, recall@3 = ___, MRR@3 = ___.
OpenAI/MiniLM had the better score separation for out-of-scope queries: ___.
```

## Honest Limit

This is a small evaluation set over 21 chunks, so the comparison is not a final benchmark. It is a useful sanity check for whether the paid embedding model gives better retrieval quality than a local embedding model on this project.
