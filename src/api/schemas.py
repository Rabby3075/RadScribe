from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class AnalyzeResponse(BaseModel):
    status: str
    report: str
    disclaimer: str
    vision_status: str | None = None
    main_findings: list[str] = Field(default_factory=list)
    borderline_findings: list[str] = Field(default_factory=list)
    vision_predictions: list[dict[str, Any]] = Field(default_factory=list)
    best_retrieval_score: float | None = None
    retrieval_ok: bool | None = None
    critic_result: dict[str, Any] = Field(default_factory=dict)
    trace_path: str | None = None
