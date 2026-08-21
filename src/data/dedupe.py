from __future__ import annotations

from pathlib import Path

import imagehash
import pandas as pd
from PIL import Image
from tqdm import tqdm

from common import DEDUPED_INTERIM, IMAGES_INTERIM, file_sha256, save_parquet


def perceptual_hash(path: Path) -> str:
    with Image.open(path) as img:
        return str(imagehash.phash(img))


def dedupe() -> pd.DataFrame:
    images = pd.read_parquet(IMAGES_INTERIM)
    rows = []
    for row in tqdm(images.itertuples(index=False), total=len(images), desc="hash images"):
        path = Path(row.image_path)
        sha = ""
        phash = ""
        if row.image_cleaned and path.exists():
            sha = file_sha256(path)
            phash = perceptual_hash(path)
        rows.append({**row._asdict(), "sha256": sha, "phash": phash})

    df = pd.DataFrame(rows)
    df["exact_duplicate"] = df.duplicated("sha256", keep="first") & df["sha256"].ne("")
    df["near_duplicate"] = df.duplicated("phash", keep="first") & df["phash"].ne("")
    df["keep_image"] = df["image_cleaned"] & ~df["exact_duplicate"] & ~df["near_duplicate"]

    save_parquet(df, DEDUPED_INTERIM)
    print(f"saved {DEDUPED_INTERIM} with {len(df)} rows")
    print(f"kept images: {int(df['keep_image'].sum())}")
    print(f"exact duplicates removed: {int(df['exact_duplicate'].sum())}")
    print(f"near duplicates removed: {int(df['near_duplicate'].sum())}")
    return df


if __name__ == "__main__":
    dedupe()
