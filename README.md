# RadScribe

**An educational, multimodal, agentic chest X-ray reporting assistant.** RadScribe reads a chest X-ray, retrieves reference evidence, drafts a cautious report grounded in that evidence, checks its own claims, and stays silent when it is not confident. It was built end to end: dataset, vision model, retrieval, agent, and a full quantitative evaluation.

> **Not a medical device. Not for diagnosis or patient care.** RadScribe is a research and learning project built on public, de-identified data. Every output carries a disclaimer, and the system is designed to defer to a human rather than make decisions.

Live demo (local): a FastAPI app with an upload page that shows the report, the findings, the confidence, and the evidence behind each call.

---

## What it does

```
chest X-ray  ->  guardrail  ->  vision model (findings + confidence)
             ->  gate: report only findings the model is confident about
             ->  retrieve reference evidence (cited)
             ->  draft a cautious report, grounded in that evidence
             ->  critic checks every claim against the evidence
             ->  report, or abstain when confidence is low
```

The point of the project is not a high accuracy number. It is a system that knows the limits of its own model, cites what it says, and refuses when the evidence is weak. The final evaluation is reported honestly, including where it falls short.

---

## Headline results

**Vision (Phase 2), study-level test set.** A domain-pretrained baseline was benchmarked against a fine-tuned DenseNet-121. The baseline edges the fine-tune overall on limited data, and the fine-tune wins on Atelectasis average precision.

| Finding | Baseline AUROC | Fine-tuned AUROC | Note |
|---|---:|---:|---|
| Pleural Effusion | 0.931 | 0.906 | supported |
| Cardiomegaly | 0.896 | 0.888 | supported |
| Consolidation / Pneumonia | 0.895 | 0.878 | supported |
| Atelectasis | 0.816 | 0.826 | supported |
| Edema | 0.553 | 0.772 | data-limited (2 test cases) |
| Pneumothorax | 0.329 | 0.603 | data-limited (3 test cases) |

Macro AUROC on the four supported findings: baseline ~0.885, fine-tuned ~0.874.

**Retrieval (Phase 3).** A licence-checked knowledge base of six findings, embedded in Chroma (cosine). Recall@3 ~0.92, MRR ~0.80, with a ~0.38 relevance threshold that separates in-scope from out-of-scope queries. Local MiniLM and OpenAI embeddings tied on recall.

**End-to-end agent (Phase 5), 550 test studies.**

| Metric | Value |
|---|---:|
| Sensitivity | 0.617 |
| Specificity | 0.869 |
| False-report rate | 0.131 |
| Retrieval-OK on drafted cases | 1.000 |
| Disclaimer on every output | 1.000 |

The honest takeaway: the safety wrapper works well (grounding, abstention, self-check, disclaimer), and the bottleneck is the vision model. When the classifier is confidently wrong, the agent can still draft the wrong finding. Better wording around a weak model does not fix a weak model.

---

## How it is built

| Phase | What it delivers |
|---|---|
| 1. Dataset | Curated 3,791 chest X-rays / 3,666 studies from the Indiana University set. Patient-level splits with zero leakage, a datasheet, and a manual label audit (~90% agreement). |
| 2. Vision | Fine-tuned DenseNet-121 vs a pretrained baseline, per-class study-level AUROC/AP, honest data-limited flags. `predict_findings(image)` tool. |
| 3. Retrieval | Licence-gated knowledge base from public-domain and CC-BY-SA sources, Chroma cosine index, `retrieve(query, k)` tool, measured relevance threshold. |
| 4. Agent | A LangGraph agent (guardrail, vision, retrieve, draft, critic, gate) that grounds claims, abstains, catches an induced hallucination, and logs a trace per run. |
| 5. Evaluation | The full agent scored across 550 studies: sensitivity, specificity, false-report rate, per-finding precision/recall, and a traced diagnosis of the bottleneck. |
| 6. Deploy | A FastAPI service and a small upload UI, packaged for local Docker. Disclaimer visible on every output. |

Each phase has its own write-up: `datasheet.md`, `PHASE2.md`, `PHASE3.md`, `PHASE4.md`, `PHASE5.md`, `PHASE6.md`.

---

## Tech stack

Python, PyTorch, torchxrayvision, scikit-learn, LangGraph, LangChain, Chroma, sentence-transformers, OpenAI API (gpt-4o-mini), FastAPI, Docker, pandas.

---

## Run it

Analyze one image from the command line:

```bash
python -m src.models.predict data/processed/images_224/1_IM-0001-4001.dcm.png
```

Run the full agent on an image:

```bash
python -m src.agent.run data/processed/images_224/797_IM-2332-1001.dcm.png
```

Start the web app:

```bash
python -m uvicorn src.api.main:app --reload
# open http://127.0.0.1:8000/
```

Or with Docker:

```bash
docker compose up --build
# open http://127.0.0.1:8000/   (set OPENAI_API_KEY in a .env file first)
```

The vision tool returns probabilities for six findings: Cardiomegaly, Atelectasis, Consolidation / Pneumonia, Pleural Effusion, Edema, and Pneumothorax. `No Finding` and `Other` are derived, not model outputs.

---

## Repository layout

```
src/data/    Phase 1 dataset pipeline (parse, label, dedupe, split, manifest)
src/models/  Phase 2 vision (baseline, train, evaluate, predict)
src/rag/     Phase 3 knowledge base + retrieval
src/agent/   Phase 4 LangGraph agent (state, nodes, graph, run)
src/api/     Phase 6 FastAPI app
web/         upload UI
notebooks/   per-phase notebooks
outputs/     saved metrics, traces, and figures
```

---

## Limitations

RadScribe is a local demo, not a clinical or production system. The labels are rule-derived (about 90% agreement), disease prevalence in the test set is low (~19%), and two findings are too rare to measure. There is no authentication, database, rate limiting, DICOM/PHI handling, or robust out-of-domain image detector. A real clinical tool would need all of those plus external validation. These limits are documented rather than hidden, because knowing where a system fails is part of building it well.