from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openai import OpenAI
from PIL import Image, ImageStat, UnidentifiedImageError

from src.agent.state import AgentState
from src.models.predict import predict_findings
from src.rag.retrieve import retrieve


MAIN_VISION_THRESHOLD = 0.70
BORDERLINE_VISION_THRESHOLD = 0.50
MIN_RETRIEVAL_SCORE = 0.38
RETRIEVAL_K = 3
DRAFT_MODEL = "gpt-4o-mini"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRACE_DIR = PROJECT_ROOT / "outputs" / "agent" / "traces"
DISCLAIMER = (
    "Educational prototype. Not a medical device, not a diagnosis. "
    "For research use only; consult a qualified radiologist."
)


def load_env() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv(PROJECT_ROOT / ".env")
    except ModuleNotFoundError:
        pass


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_ready(v) for v in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def with_disclaimer(text: str) -> str:
    text = (text or "").strip()
    if DISCLAIMER in text:
        return text
    return f"{text}\n\n{DISCLAIMER}".strip()


def _looks_like_plain_radiograph(image_path: Path) -> tuple[bool, str]:
    with Image.open(image_path) as image:
        rgb = image.convert("RGB").resize((128, 128))

    hsv = rgb.convert("HSV")
    saturation = ImageStat.Stat(hsv).mean[1] / 255
    rgb_means = ImageStat.Stat(rgb).mean
    channel_gap = max(rgb_means) - min(rgb_means)

    if saturation > 0.12 or channel_gap > 18:
        return (
            False,
            "Input image looks too colorful for a chest X-ray. Please upload a grayscale chest radiograph.",
        )

    return True, "input image is readable and looks like a grayscale radiograph"


def guardrail_node(state: AgentState) -> AgentState:
    image_path = Path(state["image_path"])
    if not image_path.exists():
        return {
            **state,
            "guardrail_passed": False,
            "guardrail_reason": f"Input file does not exist: {image_path}",
        }

    if not image_path.is_file():
        return {
            **state,
            "guardrail_passed": False,
            "guardrail_reason": f"Input path is not a file: {image_path}",
        }

    try:
        with Image.open(image_path) as image:
            image.verify()
    except (UnidentifiedImageError, OSError) as exc:
        return {
            **state,
            "guardrail_passed": False,
            "guardrail_reason": f"Input is not a readable image: {exc}",
        }

    looks_ok, reason = _looks_like_plain_radiograph(image_path)
    if not looks_ok:
        return {
            **state,
            "guardrail_passed": False,
            "guardrail_reason": reason,
        }

    return {
        **state,
        "guardrail_passed": True,
        "guardrail_reason": reason,
    }


def route_after_guardrail(state: AgentState) -> str:
    if state.get("guardrail_passed"):
        return "vision"
    return "input_refusal"


def input_refusal_node(state: AgentState) -> AgentState:
    reason = state.get("guardrail_reason", "Input failed the guardrail.")
    draft = f"Input refused: {reason}"
    final = with_disclaimer(draft)
    return {
        **state,
        "vision_predictions": [],
        "main_findings": [],
        "borderline_findings": [],
        "retrieval_findings": [],
        "vision_status": "input_refused",
        "evidence": [],
        "best_retrieval_score": 0.0,
        "retrieval_ok": False,
        "can_draft": False,
        "draft_report": draft,
        "critic_result": {
            "supported": True,
            "missing_evidence": [],
            "safety_note": "Input was refused before vision inference.",
        },
        "final_report": final,
    }


def vision_node(state: AgentState) -> AgentState:
    probs = predict_findings(state["image_path"])
    predictions = sorted(
        [{"finding": finding, "probability": float(prob)} for finding, prob in probs.items()],
        key=lambda row: row["probability"],
        reverse=True,
    )

    main_findings = [
        row["finding"] for row in predictions if row["probability"] >= MAIN_VISION_THRESHOLD
    ]
    borderline_findings = [
        row["finding"]
        for row in predictions
        if BORDERLINE_VISION_THRESHOLD <= row["probability"] < MAIN_VISION_THRESHOLD
    ]

    if main_findings:
        vision_status = "high_confidence"
    elif borderline_findings:
        vision_status = "borderline"
    else:
        vision_status = "no_finding_above_threshold"

    return {
        **state,
        "vision_predictions": predictions,
        "main_findings": main_findings,
        "borderline_findings": borderline_findings,
        "retrieval_findings": main_findings + borderline_findings,
        "vision_status": vision_status,
    }


def route_after_vision(state: AgentState) -> str:
    if not state.get("retrieval_findings"):
        return "no_draft"
    return "retrieve"


def retrieve_node(state: AgentState) -> AgentState:
    evidence = []
    probabilities = {
        row["finding"]: float(row["probability"]) for row in state.get("vision_predictions", [])
    }

    for finding in state.get("retrieval_findings", []):
        group = "main" if finding in state.get("main_findings", []) else "borderline"
        query = f"chest xray finding {finding} appearance"
        for hit in retrieve(query, k=RETRIEVAL_K):
            evidence.append(
                {
                    "query_finding": finding,
                    "confidence_group": group,
                    "vision_probability": probabilities.get(finding, 0.0),
                    **hit,
                }
            )

    best_score = max((float(row["score"]) for row in evidence), default=0.0)
    retrieval_ok = bool(best_score >= MIN_RETRIEVAL_SCORE)

    return {
        **state,
        "evidence": evidence,
        "best_retrieval_score": best_score,
        "retrieval_ok": retrieval_ok,
        "can_draft": bool(state.get("main_findings")) and retrieval_ok,
    }


def route_after_retrieval(state: AgentState) -> str:
    if state.get("can_draft"):
        return "draft"
    return "no_draft"


def no_draft_node(state: AgentState) -> AgentState:
    if state.get("borderline_findings") and not state.get("main_findings"):
        draft = "Only borderline model findings were present. No main disease-specific report was generated."
    elif state.get("retrieval_findings") and not state.get("retrieval_ok"):
        draft = "Low confidence: the model selected a finding, but retrieval did not find enough supporting evidence."
    else:
        draft = "No finding above the model confidence threshold. No disease-specific draft was generated."

    final = with_disclaimer(draft)
    return {
        **state,
        "draft_report": draft,
        "critic_result": {
            "supported": True,
            "missing_evidence": [],
            "safety_note": "No disease-specific claim was made.",
        },
        "final_report": final,
    }


def short_evidence_block(evidence: list[dict[str, Any]], max_rows: int = 6) -> str:
    if not evidence:
        return "No retrieved evidence."

    lines = []
    for row in evidence[:max_rows]:
        lines.append(
            f"- [{row['chunk_id']}] finding={row['finding']} "
            f"score={row['score']:.3f}: {row['text']}"
        )
    return "\n".join(lines)


def make_draft_prompt(state: AgentState) -> str:
    vision_rows = []
    for row in state.get("vision_predictions", []):
        vision_rows.append(f"- {row['finding']}: {row['probability']:.3f}")

    main = ", ".join(state.get("main_findings", [])) or "None"
    borderline = ", ".join(state.get("borderline_findings", [])) or "None"
    question = state.get("question") or "No extra question."

    return f"""
Draft a concise chest X-ray report from the information below.

Use the vision findings as model output, not as a final diagnosis.
Use the retrieved passages only for grounding.
Do not add findings that are not listed or supported.
Main findings are the only findings allowed in the main impression.
Borderline findings may be mentioned only as lower-confidence observations.
Never state a finding as confirmed. Use wording like "model suggests", "possible", or "may represent".
Do not use "demonstrates", "shows", "is present", or "there is" for model-selected findings.

Question:
{question}

All vision probabilities:
{chr(10).join(vision_rows)}

Vision status:
{state.get('vision_status')}

Main model findings, probability >= {MAIN_VISION_THRESHOLD:.2f}:
{main}

Borderline model findings, probability {BORDERLINE_VISION_THRESHOLD:.2f}-{MAIN_VISION_THRESHOLD:.2f}:
{borderline}

Best retrieval score:
{state.get('best_retrieval_score', 0.0):.3f}

Retrieved evidence:
{short_evidence_block(state.get('evidence', []))}

Write:
Findings: short paragraph.
Impression: 1-2 short bullets.
Evidence: cite chunk ids in parentheses.
""".strip()


def draft_node(state: AgentState) -> AgentState:
    load_env()
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set.")

    prompt = make_draft_prompt(state)
    client = OpenAI()
    response = client.chat.completions.create(
        model=DRAFT_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Write concise, evidence-grounded chest X-ray draft text. "
                    "Never state a finding as confirmed. "
                    "Do not use 'demonstrates', 'shows', 'is present', or 'there is' "
                    "for model-selected findings. Use 'model suggests', 'possible', "
                    "or 'may represent'."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )

    return {**state, "draft_report": response.choices[0].message.content}


def make_critic_prompt(state: AgentState) -> str:
    return f"""
Check this draft against the retrieved evidence.

Return raw JSON only. Do not wrap it in markdown or code fences.

Use these keys:
- supported: true or false
- missing_evidence: list of claims not supported
- safety_note: short note

Important scope:
The critic checks whether the draft is supported by the evidence text.
It does not verify whether the finding is truly present in the image.
Wrong-finding protection comes from the vision-confidence gate.
If the draft mentions a finding that is not present in the evidence, mark it unsupported.

Draft:
{state.get('draft_report', '')}

Evidence:
{short_evidence_block(state.get('evidence', []), max_rows=8)}
""".strip()


def critic_node(state: AgentState) -> AgentState:
    load_env()
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set.")

    client = OpenAI()
    response = client.chat.completions.create(
        model=DRAFT_MODEL,
        messages=[
            {"role": "system", "content": "You are a strict evidence checker. Return JSON only."},
            {"role": "user", "content": make_critic_prompt(state)},
        ],
        temperature=0,
    )

    text = response.choices[0].message.content or "{}"
    try:
        critic_result = json.loads(text)
    except json.JSONDecodeError:
        critic_result = {
            "supported": False,
            "missing_evidence": ["critic returned non-JSON text"],
            "safety_note": text,
        }

    return {
        **state,
        "critic_result": critic_result,
        "final_report": with_disclaimer(state.get("draft_report", "")),
    }


def save_trace_node(state: AgentState) -> AgentState:
    TRACE_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    image_stem = Path(state["image_path"]).stem.replace(" ", "_")
    path = TRACE_DIR / f"{stamp}_{image_stem}.json"

    path.write_text(json.dumps(_json_ready(state), indent=2), encoding="utf-8")
    return {**state, "trace_path": str(path)}
