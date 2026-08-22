from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.rag.common import EVAL_QUERIES_PATH, OUTPUT_DIR
from src.rag.retrieve import retrieve


DEFAULT_EVAL_QUERIES = [
    {"query_id": "cardio_01", "query": "enlarged heart on chest x ray", "expected_finding": "Cardiomegaly", "scope": "in_scope"},
    {"query_id": "cardio_02", "query": "heart shadow is too large", "expected_finding": "Cardiomegaly", "scope": "in_scope"},
    {"query_id": "cardio_03", "query": "cardiothoracic ratio greater than fifty percent", "expected_finding": "Cardiomegaly", "scope": "in_scope"},
    {"query_id": "cardio_04", "query": "portable AP film makes the heart look enlarged", "expected_finding": "Cardiomegaly", "scope": "in_scope"},
    {"query_id": "atel_01", "query": "volume loss in part of the lung", "expected_finding": "Atelectasis", "scope": "in_scope"},
    {"query_id": "atel_02", "query": "collapsed air spaces with elevated diaphragm", "expected_finding": "Atelectasis", "scope": "in_scope"},
    {"query_id": "atel_03", "query": "linear opacity from subsegmental collapse", "expected_finding": "Atelectasis", "scope": "in_scope"},
    {"query_id": "atel_04", "query": "loss of air and reduced lung volume", "expected_finding": "Atelectasis", "scope": "in_scope"},
    {"query_id": "cons_01", "query": "air spaces filled with inflammatory material", "expected_finding": "Consolidation / Pneumonia", "scope": "in_scope"},
    {"query_id": "cons_02", "query": "pneumonia causing dense white opacity", "expected_finding": "Consolidation / Pneumonia", "scope": "in_scope"},
    {"query_id": "cons_03", "query": "air bronchograms inside an opacity", "expected_finding": "Consolidation / Pneumonia", "scope": "in_scope"},
    {"query_id": "cons_04", "query": "alveoli are filled instead of air filled", "expected_finding": "Consolidation / Pneumonia", "scope": "in_scope"},
    {"query_id": "eff_01", "query": "fluid around the lung", "expected_finding": "Pleural Effusion", "scope": "in_scope"},
    {"query_id": "eff_02", "query": "blunting of the costophrenic angle", "expected_finding": "Pleural Effusion", "scope": "in_scope"},
    {"query_id": "eff_03", "query": "meniscus sign at the lung base", "expected_finding": "Pleural Effusion", "scope": "in_scope"},
    {"query_id": "eff_04", "query": "extra fluid in the pleural space", "expected_finding": "Pleural Effusion", "scope": "in_scope"},
    {"query_id": "edema_01", "query": "fluid buildup inside the lungs", "expected_finding": "Edema", "scope": "in_scope"},
    {"query_id": "edema_02", "query": "bat wing opacities and Kerley B lines", "expected_finding": "Edema", "scope": "in_scope"},
    {"query_id": "edema_03", "query": "vascular redistribution with pulmonary edema", "expected_finding": "Edema", "scope": "in_scope"},
    {"query_id": "edema_04", "query": "heart failure causing lung fluid", "expected_finding": "Edema", "scope": "in_scope"},
    {"query_id": "ptx_01", "query": "air in the pleural space", "expected_finding": "Pneumothorax", "scope": "in_scope"},
    {"query_id": "ptx_02", "query": "visible pleural line with no lung markings beyond it", "expected_finding": "Pneumothorax", "scope": "in_scope"},
    {"query_id": "ptx_03", "query": "collapsed lung from air outside the lung", "expected_finding": "Pneumothorax", "scope": "in_scope"},
    {"query_id": "ptx_04", "query": "tension pneumothorax is an emergency", "expected_finding": "Pneumothorax", "scope": "in_scope"},
    {"query_id": "oos_01", "query": "broken rib fracture after trauma", "expected_finding": "Out of scope", "scope": "out_of_scope"},
    {"query_id": "oos_02", "query": "kidney stone seen on CT abdomen", "expected_finding": "Out of scope", "scope": "out_of_scope"},
    {"query_id": "oos_03", "query": "brain MRI shows stroke", "expected_finding": "Out of scope", "scope": "out_of_scope"},
    {"query_id": "oos_04", "query": "patient has a skin rash on the arm", "expected_finding": "Out of scope", "scope": "out_of_scope"},
]


def ensure_eval_queries(path=EVAL_QUERIES_PATH) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    eval_df = pd.DataFrame(DEFAULT_EVAL_QUERIES)
    eval_df.to_csv(path, index=False)
    return eval_df


def run_retrieval(eval_df: pd.DataFrame, k: int = 5, provider: str = "openai") -> pd.DataFrame:
    rows = []
    for query_row in eval_df.to_dict("records"):
        for hit in retrieve(query_row["query"], k=k, provider=provider):
            rows.append({**query_row, **hit})
    return pd.DataFrame(rows)


def metrics_at_k(hits: pd.DataFrame, k: int) -> dict[str, float]:
    in_scope = hits[hits["scope"] == "in_scope"].copy()
    per_query = []

    for _, group in in_scope.groupby("query_id"):
        group = group.sort_values("rank").head(k)
        expected = group["expected_finding"].iloc[0]
        matches = group["finding"].eq(expected).to_numpy()

        reciprocal_rank = 0.0
        if matches.any():
            reciprocal_rank = 1.0 / float(np.where(matches)[0][0] + 1)

        per_query.append(
            {
                "recall": float(matches.any()),
                "precision": float(matches.mean()),
                "reciprocal_rank": reciprocal_rank,
            }
        )

    per_query_df = pd.DataFrame(per_query)
    return {
        "k": k,
        "recall": per_query_df["recall"].mean(),
        "precision": per_query_df["precision"].mean(),
        "mrr": per_query_df["reciprocal_rank"].mean(),
    }


def summarize_per_query(hits: pd.DataFrame) -> pd.DataFrame:
    rows = []
    in_scope = hits[hits["scope"] == "in_scope"].copy()
    for query_id, group in in_scope.groupby("query_id"):
        group = group.sort_values("rank")
        expected = group["expected_finding"].iloc[0]
        top_findings = group.head(5)["finding"].tolist()
        first_correct_rank = next(
            (i + 1 for i, finding in enumerate(top_findings) if finding == expected),
            None,
        )
        rows.append(
            {
                "query_id": query_id,
                "query": group["query"].iloc[0],
                "expected_finding": expected,
                "top1_finding": top_findings[0],
                "first_correct_rank": first_correct_rank,
                "top5_findings": top_findings,
            }
        )
    return pd.DataFrame(rows)


def evaluate(k: int = 5, provider: str = "openai") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    eval_df = ensure_eval_queries()
    hits = run_retrieval(eval_df, k=k, provider=provider)
    metrics = pd.DataFrame([metrics_at_k(hits, n) for n in [1, 3, 5]])
    per_query = summarize_per_query(hits)

    metrics["provider"] = provider
    metrics.to_csv(OUTPUT_DIR / f"retrieval_metrics_{provider}.csv", index=False)
    per_query.to_csv(OUTPUT_DIR / f"retrieval_per_query_{provider}.csv", index=False)
    hits.to_csv(OUTPUT_DIR / f"retrieval_hits_{provider}.csv", index=False)

    if provider == "openai":
        metrics.to_csv(OUTPUT_DIR / "retrieval_metrics.csv", index=False)
        per_query.to_csv(OUTPUT_DIR / "retrieval_per_query.csv", index=False)
        hits.to_csv(OUTPUT_DIR / "retrieval_hits.csv", index=False)

    return metrics, per_query, hits


def compare_providers(k: int = 5) -> pd.DataFrame:
    """Evaluate OpenAI and local MiniLM retrieval with the same query set."""
    rows = []
    for provider in ["openai", "local"]:
        metrics, _, hits = evaluate(k=k, provider=provider)
        top_hits = hits[hits["rank"] == 1].copy()
        score_summary = top_hits.groupby("scope")["score"].agg(["mean", "min", "max"])
        row = {"provider": provider}
        for _, metric_row in metrics.iterrows():
            current_k = int(metric_row["k"])
            row[f"recall@{current_k}"] = metric_row["recall"]
            row[f"precision@{current_k}"] = metric_row["precision"]
            row[f"mrr@{current_k}"] = metric_row["mrr"]
        row["in_scope_mean_score"] = score_summary.loc["in_scope", "mean"]
        row["in_scope_min_score"] = score_summary.loc["in_scope", "min"]
        row["in_scope_max_score"] = score_summary.loc["in_scope", "max"]
        row["out_of_scope_mean_score"] = score_summary.loc["out_of_scope", "mean"]
        row["out_of_scope_min_score"] = score_summary.loc["out_of_scope", "min"]
        row["out_of_scope_max_score"] = score_summary.loc["out_of_scope", "max"]
        row["score_gap"] = row["in_scope_mean_score"] - row["out_of_scope_mean_score"]
        rows.append(row)

    comparison = pd.DataFrame(rows)
    comparison.to_csv(OUTPUT_DIR / "retrieval_provider_comparison.csv", index=False)
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the RadScribe KB retriever.")
    parser.add_argument("-k", type=int, default=5)
    parser.add_argument("--provider", choices=["openai", "local", "both"], default="openai")
    args = parser.parse_args()

    if args.provider == "both":
        comparison = compare_providers(k=args.k)
        print(comparison.to_string(index=False))
        return

    metrics, _, hits = evaluate(k=args.k, provider=args.provider)
    print(metrics.to_string(index=False))

    top_hits = hits[hits["rank"] == 1].copy()
    score_summary = top_hits.groupby("scope")["score"].agg(["count", "mean", "min", "max"])
    print()
    print(score_summary.to_string())


if __name__ == "__main__":
    main()
