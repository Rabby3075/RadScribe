from __future__ import annotations

import os
from pathlib import Path

import requests
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse


ROOT = Path(__file__).resolve().parents[1]
DISCLAIMER = (
    "Educational prototype. Not a medical device, not a diagnosis. "
    "For research use only; consult a qualified radiologist."
)

app = FastAPI(title="RadScribe Vercel UI")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "mode": "vercel-ui"}


@app.get("/")
def index() -> HTMLResponse:
    return HTMLResponse((ROOT / "web" / "index.html").read_text(encoding="utf-8"))


@app.post("/analyze")
async def analyze(image: UploadFile = File(...)) -> JSONResponse:
    backend_url = os.getenv("RADSCRIBE_API_BASE_URL", "").rstrip("/")

    if not backend_url:
        return JSONResponse(
            content={
                "status": "backend_not_connected",
                "report": (
                    "The Vercel UI is running, but the ML backend is not connected. "
                    "Run the Docker/FastAPI backend and set RADSCRIBE_API_BASE_URL "
                    "to its public URL."
                    f"\n\n{DISCLAIMER}"
                ),
                "disclaimer": DISCLAIMER,
                "vision_status": "backend_not_connected",
                "main_findings": [],
                "borderline_findings": [],
                "vision_predictions": [],
                "best_retrieval_score": None,
                "retrieval_ok": False,
                "critic_result": {},
                "trace_path": None,
            },
        )

    content = await image.read()
    files = {
        "image": (
            image.filename or "upload.png",
            content,
            image.content_type or "application/octet-stream",
        )
    }

    try:
        response = requests.post(f"{backend_url}/analyze", files=files, timeout=90)
    except requests.RequestException:
        return JSONResponse(
            content={
                "status": "backend_unreachable",
                "report": f"The ML backend could not be reached.\n\n{DISCLAIMER}",
                "disclaimer": DISCLAIMER,
                "vision_status": "backend_unreachable",
                "main_findings": [],
                "borderline_findings": [],
                "vision_predictions": [],
                "best_retrieval_score": None,
                "retrieval_ok": False,
                "critic_result": {},
                "trace_path": None,
            },
        )

    return JSONResponse(status_code=response.status_code, content=response.json())
