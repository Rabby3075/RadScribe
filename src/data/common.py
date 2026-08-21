from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pandas as pd


DATA_DIR = Path("data")
SOURCE_IMAGES_DIR = DATA_DIR / "images" / "images_normalized"
REPORTS_CSV = DATA_DIR / "indiana_reports.csv"
PROJECTIONS_CSV = DATA_DIR / "indiana_projections.csv"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_IMAGES_DIR = PROCESSED_DIR / "images_224"

REPORTS_INTERIM = INTERIM_DIR / "reports.parquet"
IMAGES_INTERIM = INTERIM_DIR / "images.parquet"
LABELS_INTERIM = INTERIM_DIR / "labels.parquet"
DEDUPED_INTERIM = INTERIM_DIR / "deduped.parquet"
MANIFEST_PATH = PROCESSED_DIR / "manifest.parquet"
SPLIT_COUNTS_PATH = PROCESSED_DIR / "split_label_counts.csv"
DATASET_SUMMARY_PATH = PROCESSED_DIR / "dataset_summary.txt"

IMAGE_SIZE = 224

FINDING_LABELS = [
    "No Finding",
    "Cardiomegaly",
    "Pleural Effusion",
    "Consolidation / Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Edema",
    "Other",
]

LABEL_RULES = {
    "Cardiomegaly": ["cardiomegaly", "enlarged heart"],
    "Pleural Effusion": ["effusion", "pleural fluid"],
    "Consolidation / Pneumonia": [
        "consolidation",
        "pneumonia",
        "airspace disease",
        "airspace opacity",
        "infiltrate",
    ],
    "Atelectasis": ["atelectasis", "atelectatic"],
    "Pneumothorax": ["pneumothorax"],
    "Edema": ["edema", "pulmonary edema", "vascular congestion"],
}

NEGATION_WORDS = [
    "no",
    "not",
    "without",
    "negative for",
    "free of",
    "clear of",
    "absence of",
]

XXXX_RE = re.compile(r"\bX{2,}\b")
WS_RE = re.compile(r"\s+")


def ensure_dirs() -> None:
    for path in [INTERIM_DIR, PROCESSED_DIR, PROCESSED_IMAGES_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def clean_text(value: object) -> str:
    if not isinstance(value, str):
        return ""
    value = XXXX_RE.sub(" ", value)
    value = WS_RE.sub(" ", value)
    return value.strip()


def stable_id(value: object) -> str:
    return str(value).strip()


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_terms(value: object) -> list[str]:
    if not isinstance(value, str):
        return []
    return [term.strip().lower() for term in value.split(";") if term.strip()]


def is_normal_only(terms: list[str]) -> bool:
    return len(terms) == 1 and terms[0] == "normal"


def term_matches(term: str, keyword: str) -> bool:
    pattern = rf"\b{re.escape(keyword)}\b"
    return re.search(pattern, term, flags=re.IGNORECASE) is not None


def has_negation_near(text: str, start: int, window: int = 45) -> bool:
    before = text[max(0, start - window) : start].lower()
    for word in NEGATION_WORDS:
        pattern = rf"\b{re.escape(word)}\b"
        if re.search(pattern, before):
            return True
    return False


def keyword_present(text: str, keyword: str) -> bool:
    pattern = rf"\b{re.escape(keyword)}\b"
    for match in re.finditer(pattern, text, flags=re.IGNORECASE):
        if not has_negation_near(text, match.start()):
            return True
    return False


def labels_from_terms(terms: list[str]) -> list[str]:
    if not terms:
        return []
    if is_normal_only(terms):
        return ["No Finding"]

    labels: list[str] = []
    for label, keywords in LABEL_RULES.items():
        if any(term_matches(term, keyword) for term in terms for keyword in keywords):
            labels.append(label)

    if labels:
        return labels
    return ["Other"]


def labels_from_report_text(row: pd.Series) -> list[str]:
    text = " ".join(
        [
            str(row.get("report_findings", "") or ""),
            str(row.get("report_impression", "") or ""),
        ]
    ).lower()
    if not text.strip():
        return []

    labels = []
    for label, keywords in LABEL_RULES.items():
        if any(keyword_present(text, keyword) for keyword in keywords):
            labels.append(label)
    return labels


def derive_labels(row: pd.Series) -> list[str]:
    problem_terms = split_terms(row.get("Problems", ""))
    mesh_terms = split_terms(row.get("MeSH", ""))

    # The radiologist-curated fields are the main source. Report text is only a fallback.
    if problem_terms:
        return labels_from_terms(problem_terms)

    if mesh_terms:
        return labels_from_terms(mesh_terms)

    text_labels = labels_from_report_text(row)
    if text_labels:
        return text_labels

    report_text = " ".join(
        [
            str(row.get("report_findings", "") or ""),
            str(row.get("report_impression", "") or ""),
        ]
    ).lower()
    normal_markers = ["normal chest", "no acute cardiopulmonary", "no active disease"]
    if any(marker in report_text for marker in normal_markers):
        return ["No Finding"]

    return ["Other"]


def save_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
