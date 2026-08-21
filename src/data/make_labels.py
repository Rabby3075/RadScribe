from __future__ import annotations

import pandas as pd

from common import LABELS_INTERIM, REPORTS_INTERIM, derive_labels, save_parquet


def make_labels() -> pd.DataFrame:
    reports = pd.read_parquet(REPORTS_INTERIM)
    reports["labels"] = reports.apply(derive_labels, axis=1)
    out = reports[["uid", "study_id", "patient_id", "labels"]].copy()
    save_parquet(out, LABELS_INTERIM)
    print(f"saved {LABELS_INTERIM} with {len(out)} labelled reports")
    print(out.explode("labels")["labels"].value_counts().to_string())
    return out


if __name__ == "__main__":
    make_labels()
