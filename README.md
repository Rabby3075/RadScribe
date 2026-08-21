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

## Phase 2 Vision Files

Phase 2 builds a chest X-ray vision model that predicts six disease probabilities:

- `Cardiomegaly`
- `Atelectasis`
- `Consolidation / Pneumonia`
- `Pleural Effusion`
- `Edema`
- `Pneumothorax`

The model does not directly output `No Finding` or `Other`.

- `src/models/dataset.py`: loads the manifest, images, and 6-label targets
- `src/models/baseline.py`: runs the torchxrayvision pretrained baseline
- `src/models/model.py`: builds DenseNet-121 with a 6-output head
- `src/models/train.py`: overfit-20 check and optional full fine-tuning
- `src/models/evaluate.py`: study-level metrics and validation thresholds
- `src/models/predict.py`: `predict_findings(image_path)` tool
- `PHASE2.md`: final Phase 2 result report

Useful commands:

```bash
python -m src.models.baseline
python -m src.models.train
python -m src.models.train --full-train
python -m src.models.predict data/processed/images_224/1_IM-0001-4001.dcm.png
```

Final Phase 2 result:

- baseline macro AUROC on supported labels: about `0.885`
- fine-tuned macro AUROC on supported labels: about `0.874`
- fine-tune improved Atelectasis AP: `0.305` to `0.408`
- Edema and Pneumothorax are data-limited because the test split has very few positives

Short conclusion: the domain-pretrained baseline is slightly stronger overall, but the fine-tuned model is competitive and gives a useful honest benchmark.
