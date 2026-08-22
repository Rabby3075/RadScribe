# RadScribe Phase 4: Agent

Phase 4 connects the tools from the earlier phases into one workflow:

```text
image -> input guardrail -> vision model -> confidence gate -> retrieval -> draft -> critic -> trace
```

The goal is not to make a radiologist replacement. The goal is to show a careful agent that knows when to write and when to stay quiet.

## Entry Point

The main function is:

```python
run_agent(image_path, question=None)
```

It lives in:

```text
src/agent/run.py
```

## Safety Disclaimer

Every final output ends with:

```text
Educational prototype. Not a medical device, not a diagnosis. For research use only; consult a qualified radiologist.
```

This line is added to both report and no-draft/refusal paths, and it is saved in the trace as part of `final_report`.

## Input Guardrail

The first node checks the input path before vision inference:

- file must exist
- path must be a file
- file must be readable as an image

If the input fails this basic check, the graph refuses the input, writes a trace, and does not run the vision model. Strong out-of-domain detection, such as detecting non-chest images, is future work.

## Gate Logic

The most important part of this phase is the confidence gate.

```text
probability >= 0.70 -> main finding
0.50 <= probability < 0.70 -> borderline finding
probability < 0.50 -> ignored
```

If no finding passes the threshold, the agent does not retrieve evidence and does not draft a disease-specific report.

Retrieval is only used after the vision model has selected a finding. This matters because retrieval can prove that the knowledge base has a passage about a finding, but it cannot prove that the finding is present in the image.

## Nodes

The notebook flow maps into LangGraph like this:

```text
vision_node
  -> guardrail/refusal check happens before this
  -> retrieve_node or no_draft_node
  -> draft_node
  -> critic_node
  -> save_trace_node
```

The conditional edge after `vision_node` handles the safety gate. If there are no findings above threshold, the graph goes straight to `no_draft_node`.

## Critic Scope

The critic checks whether the draft is supported by the retrieved evidence.

It does not verify whether the finding is truly present in the image. Wrong-finding protection comes from the vision-confidence gate, not from the critic.

This is important because the critic sees the same evidence used to write the draft. It can catch invented clinical detail, but it should not be described as diagnostic verification.

## Traces

Each run saves a JSON trace in:

```text
outputs/agent/traces/
```

The trace keeps the vision probabilities, selected findings, retrieval evidence, draft, critic result, and final report. This makes the agent easier to debug and easier to explain.

The signed-off traces include:

| Case | Trace |
| --- | --- |
| Clear positive | `outputs/agent/traces/20260822T045514Z_797_IM-2332-1001.dcm.json` |
| Normal | `outputs/agent/traces/20260822T045715Z_3528_IM-1725-2002.dcm.json` |
| Other / low-confidence | `outputs/agent/traces/20260822T045724Z_1131_IM-0088-0001-0002.dcm.json` |
| Bad path guardrail | `outputs/agent/traces/20260822T045637Z_not_real.json` |

## Showcase Cases

The scratch notebook tested three fixed cases:

- clear positive: reports with evidence
- normal: no disease-specific draft
- Other/low-confidence: no disease-specific draft

The important result is that normal and Other cases do not force a fake disease finding.

## Critic Stress Test

The clean demo notebook includes an induced hallucination test. It feeds the critic a draft that claims a large pneumothorax while the evidence comes from the clear-positive trace. This is different from a natural run; it is a direct test of whether the critic can catch unsupported clinical detail.

The expected result is:

```text
supported = false
missing_evidence includes the unsupported pneumothorax claim
```

This is useful to show what the critic is actually for. It catches unsupported text. It does not decide whether the image truly contains a finding.
