from __future__ import annotations

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from common import (
    DATASET_SUMMARY_PATH,
    DEDUPED_INTERIM,
    LABELS_INTERIM,
    MANIFEST_PATH,
    REPORTS_INTERIM,
    SPLIT_COUNTS_PATH,
    save_parquet,
)


def normalize_labels(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, tuple):
        return [str(item) for item in value]
    if hasattr(value, "tolist"):
        items = value.tolist()
        if isinstance(items, list):
            return [str(item) for item in items]
    return ["Other"]


def add_splits(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
    train_idx, temp_idx = next(gss.split(df, groups=df["patient_id"]))

    train = df.iloc[train_idx]
    temp = df.iloc[temp_idx]

    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=42)
    val_idx, test_idx = next(gss2.split(temp, groups=temp["patient_id"]))

    df.loc[train.index, "split"] = "train"
    df.loc[temp.iloc[val_idx].index, "split"] = "val"
    df.loc[temp.iloc[test_idx].index, "split"] = "test"

    per_patient_splits = df.groupby("patient_id")["split"].nunique()
    assert per_patient_splits.max() == 1, "LEAK: one patient appears in multiple splits"
    return df


def build_final_manifest() -> pd.DataFrame:
    reports = pd.read_parquet(REPORTS_INTERIM)
    labels = pd.read_parquet(LABELS_INTERIM)
    images = pd.read_parquet(DEDUPED_INTERIM)

    df = images[images["keep_image"]].merge(
        reports[
            [
                "uid",
                "study_id",
                "patient_id",
                "report_findings",
                "report_impression",
                "report_text",
                "report_indication",
            ]
        ],
        on=["uid", "study_id", "patient_id"],
        how="left",
    )
    df = df.merge(labels, on=["uid", "study_id", "patient_id"], how="left")
    df["labels"] = df["labels"].apply(normalize_labels)
    df = df[df["report_text"].fillna("").str.len() > 0].copy()
    df = add_splits(df)

    columns = [
        "image_id",
        "image_path",
        "patient_id",
        "study_id",
        "report_findings",
        "report_impression",
        "labels",
        "split",
        "projection",
        "report_indication",
        "report_text",
    ]
    manifest = df[columns].sort_values(["split", "study_id", "image_id"]).reset_index(drop=True)
    manifest["image_path"] = manifest["image_path"].astype(str).str.replace("\\", "/", regex=False)
    save_parquet(manifest, MANIFEST_PATH)

    counts = (
        manifest.explode("labels")
        .groupby(["split", "labels"])
        .size()
        .reset_index(name="count")
        .sort_values(["split", "labels"])
    )
    counts.to_csv(SPLIT_COUNTS_PATH, index=False)

    overlap = manifest.groupby("patient_id")["split"].nunique().max()
    summary = [
        "MediScope Phase 1 Dataset Summary",
        f"rows/images: {len(manifest)}",
        f"patients/studies: {manifest['patient_id'].nunique()}",
        f"splits: {manifest['split'].value_counts().to_dict()}",
        f"max splits per patient: {overlap}",
        "",
        "label counts:",
        manifest.explode("labels")["labels"].value_counts().to_string(),
    ]
    DATASET_SUMMARY_PATH.write_text("\n".join(summary), encoding="utf-8")

    print(f"saved {MANIFEST_PATH} with {len(manifest)} rows")
    print(f"saved {SPLIT_COUNTS_PATH}")
    print(f"saved {DATASET_SUMMARY_PATH}")
    print("zero patient leakage check passed")
    return manifest


if __name__ == "__main__":
    build_final_manifest()
