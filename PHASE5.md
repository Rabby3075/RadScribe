# Phase 5 - End-to-End Evaluation

Phase 5 evaluates the full RadScribe agent on the test split. The goal was not to show a few good examples. The goal was to measure how the whole system behaves when every test study is included.

The agent has three main decisions:

- speak when the vision model has a main finding above the confidence gate
- stay quiet when no disease finding clears the gate
- keep every generated output grounded, checked, logged, and marked as non-clinical

## Evaluation Setup

The evaluation used the full test split from the Phase 1 manifest:

- test studies: 550
- true disease studies: 107
- normal or Other studies: 443

The Phase 4 gate was kept the same:

- main finding threshold: 0.70
- borderline threshold: 0.50
- retrieval threshold: 0.38 cosine similarity

The full test pass first runs the vision model and gate only. This gives the main decision metrics without spending LLM calls on studies where the agent should stay quiet. Then the full agent runs on the firing subset only.

## Study-Level Gate Results

| Metric | Value |
| --- | ---: |
| Test studies | 550 |
| Would draft | 124 |
| True positives | 66 |
| False positives | 58 |
| True negatives | 385 |
| False negatives | 41 |
| Sensitivity | 0.617 |
| Specificity | 0.869 |
| False-report rate | 0.131 |
| True label in main findings when drafted | 0.452 |

The most important number is the last one. When the agent writes a report, only about 45% of drafted cases have a true label in the main findings. That means the safety wrapper is not the main bottleneck. The bottleneck is the vision model: if the classifier is confidently wrong, the agent can still draft the wrong disease.

## Per-Finding Results

| Finding | Positives | Predicted Positive | Precision | Recall |
| --- | ---: | ---: | ---: | ---: |
| Cardiomegaly | 44 | 93 | 0.312 | 0.659 |
| Atelectasis | 44 | 53 | 0.396 | 0.477 |
| Consolidation / Pneumonia | 28 | 46 | 0.196 | 0.321 |
| Pleural Effusion | 22 | 30 | 0.433 | 0.591 |

Edema and Pneumothorax were not used as headline per-finding metrics because they are too rare in the test set.

## Agent Firing Subset

The full agent ran on the 124 studies where the gate fired.

| Metric | Value |
| --- | ---: |
| Agent runs | 124 |
| Retrieval OK rate | 1.000 |
| Critic supported rate | 0.952 |
| Disclaimer rate | 1.000 |

This is the good part of the result. Once a finding passed the vision gate, retrieval usually found usable evidence, the critic usually judged the draft as supported, and every final output kept the safety disclaimer.

## Critic Stress Test

The stress test planted unsupported claims into a draft. The critic returned:

- supported: false
- missing evidence: pleural effusion and large pneumothorax

This shows the critic can catch invented clinical detail. It does not prove the image finding is correct. Wrong-finding protection still depends on the vision confidence gate.

## Caveats

Disease prevalence in the test split is low: 107 out of 550 studies, about 19%. That makes precision hard because there are many normal or Other studies where false positives can happen.

The labels are also not perfect. The Phase 1 manual audit suggested roughly a 90% label-quality ceiling. So the 0.452 drafted-label match rate is a real weakness, but it should be read with the dataset limits in mind.

## Conclusion

The end-to-end system is safety-aware, but not clinically reliable. The agent behaves well around the model: it uses retrieval, checks evidence, logs traces, refuses bad inputs, and keeps the disclaimer on every output. The main failure is upstream. The vision model still produces confident false positives, and the agent cannot fix a confidently wrong classifier by adding better wording around it.

The next improvement should target the vision model first. A stronger option would be to compare this gate against the Phase 2 domain-pretrained baseline, or to train with more data before trusting the agent's drafted findings.

## Output Files

- `outputs/agent_eval/full_test_gate_results.csv`
- `outputs/agent_eval/full_test_gate_summary.json`
- `outputs/agent_eval/full_test_per_finding_metrics.csv`
- `outputs/agent_eval/agent_firing_subset_results.csv`
- `outputs/agent_eval/showcase_cases.csv`
- `outputs/agent_eval/critic_stress_test.json`
