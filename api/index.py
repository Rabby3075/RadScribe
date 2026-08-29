from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from api._shared import ROOT


app = FastAPI(title="RadScribe UI")


@app.get("/")
@app.get("/api/index.py")
def index() -> HTMLResponse:
    html = (ROOT / "web" / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(html)
