from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image, ImageOps
from tqdm import tqdm

from common import (
    IMAGE_SIZE,
    IMAGES_INTERIM,
    PROCESSED_IMAGES_DIR,
    PROJECTIONS_CSV,
    SOURCE_IMAGES_DIR,
    clean_text,
    ensure_dirs,
    save_parquet,
    stable_id,
)


def clean_one_image(source_path: Path, output_path: Path) -> tuple[bool, str]:
    try:
        with Image.open(source_path) as img:
            img = img.convert("L")
            img.thumbnail((IMAGE_SIZE, IMAGE_SIZE))
            canvas = Image.new("L", (IMAGE_SIZE, IMAGE_SIZE), 0)
            left = (IMAGE_SIZE - img.width) // 2
            top = (IMAGE_SIZE - img.height) // 2
            canvas.paste(img, (left, top))
            canvas = ImageOps.autocontrast(canvas)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            canvas.save(output_path)
        return True, ""
    except Exception as exc:
        return False, str(exc)


def clean_images(frontal_only: bool = True) -> pd.DataFrame:
    ensure_dirs()
    projections = pd.read_csv(PROJECTIONS_CSV)
    projections["study_id"] = projections["uid"].map(stable_id)
    projections["patient_id"] = projections["study_id"]
    projections["projection"] = projections["projection"].map(clean_text)

    if frontal_only:
        projections = projections[projections["projection"].str.lower() == "frontal"].copy()

    rows = []
    for row in tqdm(projections.itertuples(index=False), total=len(projections), desc="clean images"):
        source_path = SOURCE_IMAGES_DIR / row.filename
        image_id = str(row.filename).replace(".png", "")
        processed_name = f"{image_id}.png"
        processed_path = PROCESSED_IMAGES_DIR / processed_name
        exists = source_path.exists()
        ok = False
        error = ""
        if exists:
            ok, error = clean_one_image(source_path, processed_path)

        rows.append(
            {
                "uid": row.uid,
                "study_id": row.study_id,
                "patient_id": row.patient_id,
                "image_id": image_id,
                "source_image_path": str(source_path),
                "image_path": str(processed_path),
                "projection": row.projection,
                "image_exists": exists,
                "image_cleaned": ok,
                "image_error": error,
            }
        )

    out = pd.DataFrame(rows)
    save_parquet(out, IMAGES_INTERIM)
    print(f"saved {IMAGES_INTERIM} with {len(out)} image rows")
    print(f"cleaned images: {int(out['image_cleaned'].sum())}")
    return out


if __name__ == "__main__":
    clean_images()
