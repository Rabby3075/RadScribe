from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from src.models.dataset import (
    DISEASE_LABELS,
    CXRDataset,
    compute_pos_weight,
    eval_transform,
    load_manifest,
    safe_name,
    split_manifest,
    train_transform,
)
from src.models.evaluate import (
    choose_thresholds,
    compute_metric_table,
    macro_auroc_supported,
    thresholds_to_dict,
    to_study_level,
)
from src.models.model import build_densenet121


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "outputs" / "vision"


def train_one_epoch(model, loader, criterion, optimizer, device) -> float:
    model.train()
    total_loss = 0.0

    for batch in loader:
        images = batch["image"].to(device)
        targets = batch["target"].to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

    return total_loss / len(loader.dataset)


def predict_loader(model, loader, device) -> pd.DataFrame:
    model.eval()
    rows = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            targets = batch["target"].cpu().numpy()
            probs = torch.sigmoid(model(images)).cpu().numpy()

            for i in range(len(probs)):
                row = {"study_id": batch["study_id"][i], "image_id": batch["image_id"][i]}
                for j, label in enumerate(DISEASE_LABELS):
                    name = safe_name(label)
                    row[f"true_{name}"] = targets[i, j]
                    row[f"prob_{name}"] = probs[i, j]
                rows.append(row)

    return pd.DataFrame(rows)


def run_overfit_check(train_df: pd.DataFrame, pos_weight: torch.Tensor, device) -> list[float]:
    overfit_ds = Subset(CXRDataset(train_df, transform=train_transform()), list(range(20)))
    overfit_loader = DataLoader(overfit_ds, batch_size=4, shuffle=True, num_workers=0)

    model = build_densenet121(num_outputs=len(DISEASE_LABELS)).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    losses = []
    for epoch in range(1, 21):
        loss = train_one_epoch(model, overfit_loader, criterion, optimizer, device)
        losses.append(loss)
        print(f"overfit epoch {epoch:02d} loss={loss:.4f}")

    eval_loader = DataLoader(overfit_ds, batch_size=4, shuffle=False, num_workers=0)
    overfit_preds = predict_loader(model, eval_loader, device)
    overfit_preds.to_csv(OUT_DIR / "overfit20_predictions.csv", index=False)
    return losses


def run_full_training(epochs: int, batch_size: int, lr: float, device) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    np.random.seed(42)

    df = load_manifest()
    train_df, val_df, test_df = split_manifest(df)
    pos_weight = compute_pos_weight(train_df)

    train_loader = DataLoader(
        CXRDataset(train_df, transform=train_transform()),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        CXRDataset(val_df, transform=eval_transform()),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        CXRDataset(test_df, transform=eval_transform()),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = build_densenet121(num_outputs=len(DISEASE_LABELS)).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best_model_state = None
    best_val_macro = -1.0
    history = []

    for epoch in range(1, epochs + 1):
        start = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        val_image_preds = predict_loader(model, val_loader, device)
        val_study_preds = to_study_level(val_image_preds)
        val_metrics = compute_metric_table(val_study_preds)
        val_macro = macro_auroc_supported(val_metrics)

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_macro_AUROC_supported_4": val_macro,
                "seconds": time.time() - start,
            }
        )
        print(f"epoch {epoch:02d} train_loss={train_loss:.4f} val_macro={val_macro:.4f}")

        if val_macro > best_val_macro:
            best_val_macro = val_macro
            best_model_state = copy.deepcopy(model.state_dict())
            torch.save(best_model_state, OUT_DIR / "finetuned_densenet121_best.pt")

    pd.DataFrame(history).to_csv(OUT_DIR / "finetuned_train_history.csv", index=False)

    model.load_state_dict(torch.load(OUT_DIR / "finetuned_densenet121_best.pt", map_location=device, weights_only=True))

    val_image_preds = predict_loader(model, val_loader, device)
    val_study_preds = to_study_level(val_image_preds)
    thresholds_df = choose_thresholds(val_study_preds)
    thresholds_df.to_csv(OUT_DIR / "val_thresholds.csv", index=False)
    thresholds = thresholds_to_dict(thresholds_df)

    test_image_preds = predict_loader(model, test_loader, device)
    test_study_preds = to_study_level(test_image_preds)
    test_metrics = compute_metric_table(test_study_preds, thresholds=thresholds)

    test_study_preds.to_csv(OUT_DIR / "finetuned_test_predictions.csv", index=False)
    test_metrics.to_csv(OUT_DIR / "finetuned_test_metrics.csv", index=False)
    print("test macro AUROC supported 4:", macro_auroc_supported(test_metrics))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-train", action="store_true", help="Run full fine-tuning after overfit check.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    df = load_manifest()
    train_df, _, _ = split_manifest(df)
    pos_weight = compute_pos_weight(train_df)
    losses = run_overfit_check(train_df, pos_weight, device)
    pd.DataFrame({"epoch": range(1, len(losses) + 1), "loss": losses}).to_csv(
        OUT_DIR / "overfit20_history.csv",
        index=False,
    )

    if args.full_train:
        run_full_training(args.epochs, args.batch_size, args.lr, device)
    else:
        print("Full training skipped. Pass --full-train when ready.")


if __name__ == "__main__":
    main()
