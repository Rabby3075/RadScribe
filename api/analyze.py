from __future__ import annotations

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

from api._shared import proxy_analyze


app = FastAPI(title="RadScribe Analyze Proxy")


@app.post("/analyze")
@app.post("/api/analyze.py")
async def analyze(image: UploadFile = File(...)) -> JSONResponse:
    return await proxy_analyze(image)
