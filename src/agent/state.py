from __future__ import annotations

from typing import Any, TypedDict


class AgentState(TypedDict, total=False):
    image_path: str
    question: str | None
    study_id: str | None
    true_labels: list[str]

    guardrail_passed: bool
    guardrail_reason: str

    vision_predictions: list[dict[str, Any]]
    main_findings: list[str]
    borderline_findings: list[str]
    retrieval_findings: list[str]
    vision_status: str

    evidence: list[dict[str, Any]]
    best_retrieval_score: float
    retrieval_ok: bool
    can_draft: bool

    draft_report: str
    critic_result: dict[str, Any]
    final_report: str
    trace_path: str
