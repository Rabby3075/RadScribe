from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from src.agent.graph import build_graph


def run_agent(image_path: str | Path, question: str | None = None) -> dict[str, Any]:
    graph = build_graph()
    state = {
        "image_path": str(image_path),
        "question": question,
    }
    return graph.invoke(state)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the RadScribe Phase 4 agent.")
    parser.add_argument("image_path")
    parser.add_argument("--question", default=None)
    args = parser.parse_args()

    result = run_agent(args.image_path, question=args.question)
    print(result["final_report"])
    print()
    print("trace:", result.get("trace_path"))


if __name__ == "__main__":
    main()

