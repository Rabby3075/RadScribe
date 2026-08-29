from __future__ import annotations

from fastapi import FastAPI


app = FastAPI(title="RadScribe Health")


@app.get("/health")
@app.get("/api/health.py")
def health() -> dict[str, str]:
    return {"status": "ok", "mode": "vercel-ui"}
