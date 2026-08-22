from __future__ import annotations

import logging
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from PIL import UnidentifiedImageError

from src.agent.nodes import DISCLAIMER, with_disclaimer
from src.agent.run import run_agent
from src.api.schemas import AnalyzeResponse


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WEB_DIR = PROJECT_ROOT / "web"
logger = logging.getLogger(__name__)

app = FastAPI(
    title="RadScribe API",
    version="0.1.0",
    description="Small API wrapper around the RadScribe agent.",
)


@app.get("/")
def index() -> FileResponse:
    page = WEB_DIR / "index.html"
    if not page.exists():
        raise HTTPException(status_code=404, detail="UI file not found.")
    return FileResponse(page)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


def _safe_suffix(filename: str | None) -> str:
    suffix = Path(filename or "").suffix.lower()
    if suffix in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}:
        return suffix
    return ".upload"


def _response_from_state(state: dict) -> AnalyzeResponse:
    if state.get("guardrail_passed") is False:
        status = "refused"
    elif state.get("can_draft"):
        status = "drafted"
    else:
        status = "no_draft"

    return AnalyzeResponse(
        status=status,
        report=state.get("final_report", ""),
        disclaimer=DISCLAIMER,
        vision_status=state.get("vision_status"),
        main_findings=state.get("main_findings", []),
        borderline_findings=state.get("borderline_findings", []),
        vision_predictions=state.get("vision_predictions", []),
        best_retrieval_score=state.get("best_retrieval_score"),
        retrieval_ok=state.get("retrieval_ok"),
        critic_result=state.get("critic_result", {}),
        trace_path=state.get("trace_path"),
    )


def _refused_response(reason: str) -> AnalyzeResponse:
    report = with_disclaimer(f"Input refused: {reason}")
    return AnalyzeResponse(
        status="refused",
        report=report,
        disclaimer=DISCLAIMER,
        vision_status="input_refused",
        main_findings=[],
        borderline_findings=[],
        vision_predictions=[],
        best_retrieval_score=0.0,
        retrieval_ok=False,
        critic_result={
            "supported": True,
            "missing_evidence": [],
            "safety_note": "Input was refused before analysis.",
        },
        trace_path=None,
    )


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(image: UploadFile = File(...)) -> AnalyzeResponse:
    suffix = _safe_suffix(image.filename)

    try:
        with TemporaryDirectory(prefix="radscribe_") as tmp_dir:
            image_path = Path(tmp_dir) / f"upload{suffix}"
            with image_path.open("wb") as out_file:
                shutil.copyfileobj(image.file, out_file)

            state = run_agent(image_path)
            return _response_from_state(state)
    except (UnidentifiedImageError, OSError) as exc:
        logger.info("Upload refused during image validation: %s", exc)
        return _refused_response("Upload is not a readable image.")
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Analyze request failed")
        raise HTTPException(
            status_code=500,
            detail="Analyze failed inside the demo server. Check the server log for details.",
        ) from exc
