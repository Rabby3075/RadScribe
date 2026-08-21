from __future__ import annotations

from common import PROJECTIONS_CSV, REPORTS_CSV, SOURCE_IMAGES_DIR


def check_manual_download() -> None:
    print("Manual data check for IU Chest X-ray dataset")
    print("This script does not download data automatically.")
    print("Use the official Open-i/NLM source or the Kaggle-style mirror, then place files here:")
    print(f"- reports CSV: {REPORTS_CSV}")
    print(f"- projections CSV: {PROJECTIONS_CSV}")
    print(f"- images folder: {SOURCE_IMAGES_DIR}")
    print()

    missing = []
    for path in [REPORTS_CSV, PROJECTIONS_CSV, SOURCE_IMAGES_DIR]:
        if not path.exists():
            missing.append(path)

    if missing:
        print("Missing:")
        for path in missing:
            print(f"- {path}")
        raise SystemExit(1)

    image_count = len(list(SOURCE_IMAGES_DIR.glob("*.png")))
    print("All required local data paths exist.")
    print(f"PNG images found: {image_count}")


if __name__ == "__main__":
    check_manual_download()
