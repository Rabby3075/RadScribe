from __future__ import annotations

from clean_images import clean_images
from dedupe import dedupe
from make_labels import make_labels
from parse_reports import parse_reports
from split import build_final_manifest


def main() -> None:
    parse_reports()
    clean_images(frontal_only=True)
    make_labels()
    dedupe()
    build_final_manifest()


if __name__ == "__main__":
    main()
