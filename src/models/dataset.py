from __future__ import annotations

from pathlib import Path
from typing import Callable

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = PROJECT_ROOT / "data" / "processed" / "manifest.parquet"

DISEASE_LABELS = [
    "Cardiomegaly",
    "Atelectasis",
    "Consolidation / Pneumonia",
    "Pleural Effusion",
    "Edema",
    "Pneumothorax",
]

SUPPORTED_LABELS = [
    "Cardiomegaly",
    "Atelectasis",
    "Consolidation / Pneumonia",
    "Pleural Effusion",
]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def safe_name(label: str) -> str:
    return label.replace(" / ", "_").replace(" ", "_")


def make_target(labels: list[str]) -> list[int]:
    return [1 if label in labels else 0 for label in DISEASE_LABELS]


def load_manifest(path: Path = MANIFEST_PATH) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df = df.copy()
    df["target"] = df["labels"].apply(make_target)
    return df


def split_manifest(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()
    return train_df, val_df, test_df


def image_path_from_row(row: pd.Series, project_root: Path = PROJECT_ROOT) -> Path:
    return project_root / Path(str(row["image_path"]).replace("\\", "/"))


def train_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomRotation(degrees=5),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def eval_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


class CXRDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        transform: Callable | None = None,
        project_root: Path = PROJECT_ROOT,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.transform = transform
        self.project_root = project_root

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int) -> dict:
        row = self.frame.iloc[idx]
        image = Image.open(image_path_from_row(row, self.project_root)).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        return {
            "image": image,
            "target": torch.tensor(row["target"], dtype=torch.float32),
            "study_id": str(row["study_id"]),
            "image_id": row["image_id"],
        }


def compute_pos_weight(train_df: pd.DataFrame) -> torch.Tensor:
    targets = torch.tensor(train_df["target"].tolist(), dtype=torch.float32)
    pos_counts = targets.sum(dim=0)
    neg_counts = len(targets) - pos_counts
    return neg_counts / pos_counts.clamp(min=1)
