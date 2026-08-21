from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchxrayvision as xrv
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.models.dataset import DISEASE_LABELS, PROJECT_ROOT, image_path_from_row, load_manifest, safe_name
from src.models.evaluate import compute_metric_table, to_study_level


OUT_DIR = PROJECT_ROOT / "outputs" / "vision"

XRV_LABEL_MAP = {
    "Cardiomegaly": ["Cardiomegaly"],
    "Atelectasis": ["Atelectasis"],
    "Consolidation / Pneumonia": ["Consolidation"],
    "Pleural Effusion": ["Effusion"],
    "Edema": ["Edema"],
    "Pneumothorax": ["Pneumothorax"],
}


class XRVTestDataset(Dataset):
    def __init__(self, frame: pd.DataFrame) -> None:
        self.frame = frame.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int) -> dict:
        row = self.frame.iloc[idx]
        image = Image.open(image_path_from_row(row)).convert("L")
        image = np.array(image).astype(np.float32)
        image = xrv.datasets.normalize(image, 255)
        image = image[None, :, :]

        target = np.array(row["target"], dtype=np.float32)
        return {
            "image": torch.from_numpy(image),
            "target": torch.from_numpy(target),
            "study_id": str(row["study_id"]),
            "image_id": row["image_id"],
        }


def xrv_indices(model) -> dict[str, list[int]]:
    pathologies = list(model.pathologies)
    return {
        our_label: [pathologies.index(xrv_label) for xrv_label in xrv_labels]
        for our_label, xrv_labels in XRV_LABEL_MAP.items()
    }


def run_baseline(batch_size: int = 16) -> tuple[pd.DataFrame, pd.DataFrame]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_manifest()
    test_df = df[df["split"] == "test"].copy()

    loader = DataLoader(XRVTestDataset(test_df), batch_size=batch_size, shuffle=False)
    model = xrv.models.DenseNet(weights="densenet121-res224-all")
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    index_map = xrv_indices(model)

    rows = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            outputs = model(images)
            targets = batch["target"].numpy()

            probs_by_label = {}
            for label, indices in index_map.items():
                selected = outputs[:, indices]
                probs_by_label[label] = selected.max(dim=1).values.cpu().numpy()

            for i in range(len(targets)):
                row = {"study_id": batch["study_id"][i], "image_id": batch["image_id"][i]}
                for j, label in enumerate(DISEASE_LABELS):
                    name = safe_name(label)
                    row[f"true_{name}"] = targets[i, j]
                    row[f"prob_{name}"] = probs_by_label[label][i]
                rows.append(row)

    image_preds = pd.DataFrame(rows)
    study_preds = to_study_level(image_preds)
    metrics = compute_metric_table(study_preds)

    study_preds.to_csv(OUT_DIR / "baseline_test_predictions.csv", index=False)
    metrics.to_csv(OUT_DIR / "baseline_test_metrics.csv", index=False)
    return study_preds, metrics


if __name__ == "__main__":
    _, metric_table = run_baseline()
    print(metric_table.to_string(index=False))
