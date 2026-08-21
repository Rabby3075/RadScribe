from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image

from src.models.dataset import DISEASE_LABELS, PROJECT_ROOT, eval_transform
from src.models.model import build_densenet121


DEFAULT_CHECKPOINT = PROJECT_ROOT / "outputs" / "vision" / "finetuned_densenet121_best.pt"


def load_finetuned_model(checkpoint_path: Path = DEFAULT_CHECKPOINT, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_densenet121(num_outputs=len(DISEASE_LABELS), pretrained=False)
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, device


def predict_findings(image_path: str | Path, checkpoint_path: Path = DEFAULT_CHECKPOINT) -> dict[str, float]:
    model, device = load_finetuned_model(checkpoint_path)
    transform = eval_transform()
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        probs = torch.sigmoid(model(tensor)).squeeze(0).cpu().numpy()

    return {label: float(prob) for label, prob in zip(DISEASE_LABELS, probs)}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("image_path")
    args = parser.parse_args()

    print(predict_findings(args.image_path))
