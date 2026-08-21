# RadScribe / MediScope

RadScribe is an educational chest X-ray AI project that builds a clean dataset pipeline, trains/evaluates vision baselines, and exposes a small `predict_findings(image_path)` tool for disease probabilities.

Important: this is **not a medical device**. It is **not for diagnosis** and must not be used for real patient care. The goal is to learn data engineering, multimodal AI, and honest model evaluation using public, de-identified data.

## Phase 2 Headline

The strongest result is the Phase 2 vision benchmark. A domain-pretrained torchxrayvision baseline slightly beats the local fine-tuned model overall, but the fine-tuned model is competitive and improves Atelectasis Average Precision.

Study-level test results:

| label | positives | baseline AUROC | baseline AP | fine-tuned AUROC | fine-tuned AP | note |
|---|---:|---:|---:|---:|---:|---|
| Cardiomegaly | 44 | 0.896 | 0.424 | 0.888 | 0.421 | supported |
| Atelectasis | 44 | 0.816 | 0.305 | 0.826 | 0.408 | supported |
| Consolidation / Pneumonia | 28 | 0.895 | 0.294 | 0.878 | 0.213 | supported |
| Pleural Effusion | 22 | 0.931 | 0.706 | 0.906 | 0.530 | supported |
| Edema | 2 | 0.553 | 0.169 | 0.772 | 0.032 | data-limited |
| Pneumothorax | 3 | 0.329 | 0.006 | 0.603 | 0.011 | data-limited |

Macro AUROC on the four supported labels:

- baseline: about `0.885`
- fine-tuned: about `0.874`

Edema and Pneumothorax are marked data-limited because the test split has only 2 and 3 positive studies.

## Quick Run

Run prediction on one cleaned image:

```bash
python -m src.models.predict data/processed/images_224/1_IM-0001-4001.dcm.png
```

The tool returns probabilities for six disease labels:

- `Cardiomegaly`
- `Atelectasis`
- `Consolidation / Pneumonia`
- `Pleural Effusion`
- `Edema`
- `Pneumothorax`

`No Finding` and `Other` are not model outputs.

## Project Files

- `src/data/build_manifest.py`: runs the full Phase 1 dataset pipeline
- `src/models/baseline.py`: runs the torchxrayvision pretrained baseline
- `src/models/train.py`: runs the overfit-20 check and optional full fine-tuning
- `src/models/predict.py`: final `predict_findings(image_path)` tool
- `notebooks/01_eda.ipynb`: Phase 1 EDA notebook
- `notebooks/02_vision_baseline.ipynb`: pretrained baseline notebook
- `notebooks/02_vision_finetune.ipynb`: fine-tuning notebook
- `notebooks/02_vision_eval.ipynb`: final comparison notebook
- `datasheet.md`: dataset documentation
- `PHASE2.md`: Phase 2 result report
- `LICENSES.md`: dataset licence notes

## Phase 1 Dataset

Phase 1 creates:

- 3,791 cleaned frontal chest X-ray images
- 3,666 patient/study groups
- train / validation / test split
- zero patient overlap across splits
- rule-based labels from source terms and report text

Final dataset:

```text
data/processed/manifest.parquet
```

Split sizes:

| split | rows |
|---|---:|
| train | 2,653 |
| val | 574 |
| test | 564 |

Rebuild Phase 1:

```bash
python src/data/build_manifest.py
```

On Windows with the local virtual environment:

```bash
.venv\Scripts\python.exe src\data\build_manifest.py
```

## More Detail

For the dataset card, see `datasheet.md`.

For the vision benchmark writeup, see `PHASE2.md`.
