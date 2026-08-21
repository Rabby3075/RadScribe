from __future__ import annotations

import pandas as pd

from common import REPORTS_CSV, REPORTS_INTERIM, clean_text, ensure_dirs, save_parquet, stable_id


def parse_reports() -> pd.DataFrame:
    ensure_dirs()
    reports = pd.read_csv(REPORTS_CSV)
    reports["study_id"] = reports["uid"].map(stable_id)
    reports["patient_id"] = reports["study_id"]
    reports["report_findings"] = reports["findings"].map(clean_text)
    reports["report_impression"] = reports["impression"].map(clean_text)
    reports["report_indication"] = reports["indication"].map(clean_text)
    reports["report_text"] = reports["report_findings"].where(
        reports["report_findings"] != "",
        reports["report_impression"],
    )

    keep = [
        "uid",
        "study_id",
        "patient_id",
        "report_indication",
        "report_findings",
        "report_impression",
        "report_text",
        "MeSH",
        "Problems",
    ]
    out = reports[keep].copy()
    save_parquet(out, REPORTS_INTERIM)
    print(f"saved {REPORTS_INTERIM} with {len(out)} reports")
    return out


if __name__ == "__main__":
    parse_reports()
