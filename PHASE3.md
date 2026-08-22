# RadScribe Phase 3: Knowledge Base + Retrieval

Phase 3 adds a small retrieval tool for the six chest X-ray findings used by the vision model. The goal is not to create a large medical search engine. The goal is to retrieve short, licensed reference passages that a later report-writing agent can cite and stay grounded in.

## Knowledge Base

The knowledge base contains one short markdown document per finding:

- Cardiomegaly
- Atelectasis
- Consolidation / Pneumonia
- Pleural Effusion
- Edema
- Pneumothorax

Each document is written in plain language and focuses on definition, cause/meaning, chest X-ray appearance, and clinical significance.

The final KB has:

- Raw documents: 6
- Source rows: 10
- Embedded chunks: 21
- Retriever output contract: `retrieve(query, k) -> passages with text, finding, score, and source metadata`

## Licence Discipline

All KB sources are tracked in `data/kb/sources.csv`.

Allowed licences:

- `Public-Domain`
- `CC-BY-4.0`
- `CC-BY-SA-4.0`

The build step checks that every source has an approved licence, `licence_ok=True`, and a real raw markdown file before it builds the index. This prevents an unapproved source from silently entering the retriever.

The KB uses public-domain government sources and Wikipedia pages with CC-BY-SA attribution. Copyrighted or non-commercial/no-derivatives sources are not used as KB text.

## Retrieval Setup

The main retriever uses:

- Embedding model: OpenAI `text-embedding-3-small`
- Vector store: Chroma
- Distance space: cosine
- Main collection: `radscribe_kb_openai`

Cosine distance is important because the score is used later for abstention. With cosine space, the retriever score is meaningful: higher means more relevant.

## Retrieval Metrics

Evaluation used 24 in-scope queries and 4 out-of-scope queries.

Final OpenAI retrieval results:

| Metric | Value |
| --- | ---: |
| Recall@1 | 0.708 |
| Recall@3 | 0.917 |
| Recall@5 | 0.958 |
| MRR@1 | 0.708 |
| MRR@3 | 0.798 |
| MRR@5 | 0.809 |

The correct finding is retrieved in the top 3 for about 92% of in-scope queries.

## Score Separation

The out-of-scope test checks whether unrelated queries, such as kidney stones or skin rash, receive lower retrieval scores than real chest X-ray finding queries.

After switching Chroma to cosine space:

| Query Type | Mean Top Score |
| --- | ---: |
| In-scope | 0.543 |
| Out-of-scope | 0.311 |

A threshold around `0.38` separates most relevant queries from nonsense queries. This is useful for Phase 4: if retrieval is weak, the agent should not pretend it has evidence.

There is a small overlap: the weakest in-scope hit is around `0.337`, while the strongest out-of-scope hit is around `0.349`. So the threshold is useful, but not perfect. Borderline scores should be treated as low-confidence rather than hard truth.

## Provider Comparison

An optional comparison tested OpenAI embeddings against a local MiniLM model on the same chunks and queries.

| Provider | Model | Recall@1 | Recall@3 | MRR@3 | Score Gap |
| --- | --- | ---: | ---: | ---: | ---: |
| OpenAI | `text-embedding-3-small` | 0.708 | 0.917 | 0.798 | 0.233 |
| Local | `sentence-transformers/all-MiniLM-L6-v2` | 0.708 | 0.917 | 0.792 | 0.228 |

Both models performed almost the same on recall. OpenAI was slightly better on MRR@3 and score separation, so OpenAI remains the default retriever. MiniLM is a strong local fallback when API cost or offline use matters.

## Limitations

This is a small KB: 6 documents and 21 chunks. The evaluation set is also small. The results are useful for project-level validation, not a medical benchmark.

The main known weakness is wording overlap. Queries about fluid can confuse edema, pleural effusion, and sometimes pneumothorax. Phase 4 should use retrieved evidence carefully and abstain or mark low confidence when scores are near the threshold.

## Commands

Build the default OpenAI KB index:

```bash
python -m src.rag.build_kb --provider openai
```

Evaluate the default retriever:

```bash
python -m src.rag.evaluate --provider openai
```

Run the optional OpenAI vs MiniLM comparison:

```bash
python -m src.rag.build_kb --provider local
python -m src.rag.evaluate --provider both
```
