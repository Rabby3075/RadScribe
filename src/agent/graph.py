from __future__ import annotations

from langgraph.graph import END, StateGraph

from src.agent.nodes import (
    critic_node,
    draft_node,
    guardrail_node,
    input_refusal_node,
    no_draft_node,
    retrieve_node,
    route_after_guardrail,
    route_after_retrieval,
    route_after_vision,
    save_trace_node,
    vision_node,
)
from src.agent.state import AgentState


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("guardrail", guardrail_node)
    graph.add_node("input_refusal", input_refusal_node)
    graph.add_node("vision", vision_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("no_draft", no_draft_node)
    graph.add_node("draft", draft_node)
    graph.add_node("critic", critic_node)
    graph.add_node("save_trace", save_trace_node)

    graph.set_entry_point("guardrail")
    graph.add_conditional_edges(
        "guardrail",
        route_after_guardrail,
        {
            "vision": "vision",
            "input_refusal": "input_refusal",
        },
    )
    graph.add_conditional_edges(
        "vision",
        route_after_vision,
        {
            "retrieve": "retrieve",
            "no_draft": "no_draft",
        },
    )
    graph.add_conditional_edges(
        "retrieve",
        route_after_retrieval,
        {
            "draft": "draft",
            "no_draft": "no_draft",
        },
    )
    graph.add_edge("draft", "critic")
    graph.add_edge("critic", "save_trace")
    graph.add_edge("no_draft", "save_trace")
    graph.add_edge("input_refusal", "save_trace")
    graph.add_edge("save_trace", END)

    return graph.compile()
