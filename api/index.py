from __future__ import annotations

import os
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_CHROMA = ROOT / "data" / "kb" / "chroma"
TMP_CHROMA = Path("/tmp/radscribe_chroma")

os.environ.setdefault("RADSCRIBE_TRACE_DIR", "/tmp/radscribe_traces")
os.environ.setdefault("RADSCRIBE_CHROMA_DIR", str(TMP_CHROMA))

if SOURCE_CHROMA.exists() and not TMP_CHROMA.exists():
    shutil.copytree(SOURCE_CHROMA, TMP_CHROMA)

from src.api.main import app
