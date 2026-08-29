from __future__ import annotations

import os
from pathlib import Path

import requests
from fastapi import File, UploadFile
from fastapi.responses import JSONResponse


ROOT = Path(__file__).resolve().parents[1]
DISCLAIMER = (
    "Educational prototype. Not a medical device, not a diagnosis. "
    "For research use only; consult a qualified radiologist."
)


def empty_response(status: str, report: str) -> dict:
    return {
        "status": status,
        "report": f"{report}\n\n{DISCLAIMER}",
        "disclaimer": DISCLAIMER,
        "vision_status": status,
        "main_findings": [],
        "borderline_findings": [],
        "vision_predictions": [],
        "best_retrieval_score": None,
        "retrieval_ok": False,
        "critic_result": {},
        "trace_path": None,
    }


async def proxy_analyze(image: UploadFile = File(...)) -> JSONResponse:
    backend_url = os.getenv("RADSCRIBE_API_BASE_URL", "").rstrip("/")

    if not backend_url:
        return JSONResponse(
            content=empty_response(
                "backend_not_connected",
                "The Vercel UI is running, but the ML backend is not connected. "
                "Run the Docker/FastAPI backend and set RADSCRIBE_API_BASE_URL "
                "to its public URL.",
            )
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
            content=empty_response(
                "backend_unreachable",
                "The ML backend could not be reached.",
            )
        )

    return JSONResponse(status_code=response.status_code, content=response.json())
