# MediScope Phase 1 Datasheet

## Motivation

This dataset was built for an educational chest X-ray AI project. The goal is to connect an X-ray image with report text and simple finding labels.

The dataset supports later work on multimodal AI, RAG, agent workflows, and evaluation. It is not meant for real medical use.

## Composition

The final manifest has 3,791 rows. One row means one cleaned frontal chest X-ray image with its report text and labels.

The dataset has 3,666 patient/study groups. The CSV files do not give a separate patient ID, so `uid` is used as both `patient_id` and `study_id`.

Labels:

- No Finding
- Cardiomegaly
- Pleural Effusion
- Consolidation / Pneumonia
- Atelectasis
- Pneumothorax
- Edema
- Other

## Collection

The source is the Indiana University Chest X-ray Collection from Open-i / NLM, using local Kaggle-style CSV and PNG files.

The local files are:

- `data/indiana_reports.csv`
- `data/indiana_projections.csv`
- `data/images/images_normalized/`

Licence notes are recorded in `LICENSES.md`.

## Preprocessing And Cleaning

Reports were cleaned by removing de-identification tokens like `XXXX`, removing extra spaces, and keeping findings and impression text.

Images were filtered to frontal views only. Each image was converted to grayscale, resized with padding to 224 x 224, and saved in `data/processed/images_224/`.

Labels are made from the radiologist-curated `Problems` field first. If `Problems` is missing, the code uses `MeSH`. Report text is only used as a fallback. Negation matching uses word boundaries, so `no` is not accidentally matched inside words like `normal` or `cannot`.

The broad word `opacity` is not mapped directly to Consolidation / Pneumonia, because it is too non-specific.

Duplicate checking was done with SHA-256 file hashes and perceptual hashes. The pipeline removed 2 exact duplicate images and 3 near duplicate images.

Hand-check status: not done yet. The next learning task should be to manually check 30 reports and write the rough label accuracy here.

## Splits

The final split is:

| split | rows |
|---|---:|
| train | 2,653 |
| val | 574 |
| test | 564 |

Splitting is grouped by `patient_id`. Since `patient_id` comes from `uid`, all images from the same report stay in one split.

The code asserts that no patient/study appears in more than one split. The check passed.

Per-label counts are saved in:

```text
data/processed/split_label_counts.csv
```

## Known Limitations And Biases

This is a small dataset from one source. It may not represent all hospitals, machines, patient groups, or clinical styles.

The labels are created from source terms and simple rules, not by a full medical labeler. This means some labels will still be wrong. Negation is hard, and medical wording can be subtle.

Only frontal images are used in this first version. Lateral images are excluded for simplicity.

## Uses And Out Of Scope

Good uses:

- learning data curation
- building a portfolio project
- testing simple vision and report pipelines
- practicing evaluation

Out of scope:

- real diagnosis
- clinical decision-making
- patient care
- commercial medical use

## Maintenance

This dataset should be versioned when the manifest or labels change.

Suggested next version tasks:

- manually check 30 labels
- improve negation handling
- add lateral image support
- add stronger labeler comparison
- add more EDA charts
