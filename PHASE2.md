# Phase 2 Vision Report

Phase 2 turns a cleaned chest X-ray image into disease probabilities.

The final prediction function is:

```python
predict_findings(image_path)
```

It returns probabilities for these six disease labels:

- Cardiomegaly
- Atelectasis
- Consolidation / Pneumonia
- Pleural Effusion
- Edema
- Pneumothorax

`No Finding` and `Other` are not model outputs. They are dataset labels, not disease heads.

## What Was Built

This phase has two vision models:

- A pretrained torchxrayvision DenseNet baseline.
- A fine-tuned torchvision DenseNet-121 model with a 6-output sigmoid head.

The fine-tuned model uses:

- multi-label targets
- `BCEWithLogitsLoss`
- `pos_weight` computed from the train split only
- validation thresholds chosen on the validation split only
- study-level test evaluation

## Test Results

These numbers are evaluated at study level on the held-out test split.

| label | positives | baseline AUROC | baseline AP | fine-tuned AUROC | fine-tuned AP | note |
|---|---:|---:|---:|---:|---:|---|
| Cardiomegaly | 44 | 0.896 | 0.424 | 0.888 | 0.421 | supported |
| Atelectasis | 44 | 0.816 | 0.305 | 0.826 | 0.408 | supported |
| Consolidation / Pneumonia | 28 | 0.895 | 0.294 | 0.878 | 0.213 | supported |
| Pleural Effusion | 22 | 0.931 | 0.706 | 0.906 | 0.530 | supported |
| Edema | 2 | 0.553 | 0.169 | 0.772 | 0.032 | data-limited |
| Pneumothorax | 3 | 0.329 | 0.006 | 0.603 | 0.011 | data-limited |

Macro AUROC on the four supported labels:

- baseline: about 0.885
- fine-tuned: about 0.874

## Conclusion

The pretrained torchxrayvision baseline is slightly stronger overall. This makes sense because it was trained on a much larger chest X-ray collection before this project used it.

The fine-tuned model is still useful. It comes close to the baseline with much less data, and it improves Average Precision for Atelectasis.

Edema and Pneumothorax should not be treated as reliable results yet because the test split has only 2 and 3 positive studies for those labels.

The honest Phase 2 finding is:

> A strong domain-pretrained baseline slightly beats the local fine-tune overall, but the fine-tune is competitive and improves Atelectasis AP. Rare classes need more data.

## Smoke Test

The final prediction function was tested on three real test images.

Example output shape:

```text
{
  "Cardiomegaly": probability,
  "Atelectasis": probability,
  "Consolidation / Pneumonia": probability,
  "Pleural Effusion": probability,
  "Edema": probability,
  "Pneumothorax": probability
}
```

The function loaded the saved checkpoint and returned six probabilities for each image.
