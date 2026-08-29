from __future__ import annotations

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse

from api._shared import ROOT
from api._shared import proxy_analyze


app = FastAPI(title="RadScribe UI")


@app.get("/")
@app.get("/api/index.py")
def index() -> HTMLResponse:
    html = (ROOT / "web" / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(html)


@app.post("/analyze")
@app.post("/api/index.py")
async def analyze(image: UploadFile = File(...)) -> JSONResponse:
    return await proxy_analyze(image)
