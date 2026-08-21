# RadScribe / MediScope

This project is an educational chest X-ray reporting assistant.

Important: this is **not a medical device**. It is **not for diagnosis** and must not be used for real patient care. The goal is to learn data work, multimodal AI, RAG, agents, and evaluation using public, de-identified data.

## Phase 1 Status

Phase 1 builds the dataset.

The final dataset file is:

```text
data/processed/manifest.parquet
```

It has one row per cleaned frontal chest X-ray image.

Current Phase 1 output:

- 3,791 cleaned frontal images
- 3,666 patient/study groups
- train / validation / test split
- zero patient overlap across splits
- basic rule-based labels
- dataset summary files

Split sizes:

| split | rows |
|---|---:|
| train | 2,653 |
| val | 574 |
| test | 564 |

Labels are made from the radiologist-curated `Problems` field first. If that is missing, the code falls back to `MeSH`, then report text. These labels are useful for learning and first experiments, but they still need a manual hand-check before serious model training.

## Manifest Columns

The manifest has these main columns:

- `image_id`
- `image_path`
- `patient_id`
- `study_id`
- `report_findings`
- `report_impression`
- `labels`
- `split`

Extra helper columns:

- `projection`
- `report_indication`
- `report_text`

## How To Rebuild Phase 1

Run this from the project root:

```bash
python src/data/build_manifest.py
```

If you are using the local virtual environment on Windows:

```bash
.venv\Scripts\python.exe src\data\build_manifest.py
```

The pipeline runs these steps:

1. Parse the reports.
2. Clean and resize frontal images to 224 x 224.
3. Create basic labels.
4. Remove exact and near duplicate images.
5. Create train / val / test splits with zero patient leakage.

## Dataset Card

Source dataset: Indiana University Chest X-ray Collection from Open-i / NLM, using the Kaggle-style CSV and PNG layout already present in `data/`.

This dataset has chest X-ray images and report text. In this project, `uid` is used as both `patient_id` and `study_id` because the provided CSV files do not give a separate patient identifier. This is conservative for Phase 1 because all images from one report stay in only one split.

Known limits:

- single dataset source
- frontal images only for now
- labels come from source terms plus simple rules, not a full medical labeler
- no demographic data in the current manifest
- not for clinical use

## Useful Files

- `src/data/build_manifest.py`: runs the full Phase 1 pipeline
- `src/data/download.py`: checks that the manual download is in place
- `src/data/parse_reports.py`: cleans report text
- `src/data/clean_images.py`: creates 224 x 224 cleaned images
- `src/data/make_labels.py`: creates simple finding labels
- `src/data/dedupe.py`: removes duplicate images
- `src/data/split.py`: creates the final manifest and split checks
- `datasheet.md`: dataset documentation
- `LICENSES.md`: dataset licence notes
- `notebooks/01_eda.ipynb`: starter EDA notebook
