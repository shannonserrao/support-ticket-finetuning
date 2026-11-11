# Model card — Support-Ticket Triage

No model has been trained yet. Every value below must come from a tracked run.

| Field | Value | Run ID |
|---|---|---|
| Model version | | |
| Base model | | |
| Data version | | |
| Git tag | | |
| Training time and hardware | | |

## Results on the test split
| Metric | Baseline (zero-shot) | Small model | Cascade | Gate |
|---|---|---|---|---|
| Accuracy | | | | see SPEC §5 |
| Macro F1 | | | | |
| Calibration error | n/a | | n/a | ≤ 0.05 |
| Selective accuracy at 80% coverage | n/a | | n/a | ≥ 0.97 |
| Cost per 1,000 tickets | | | | ≤ 10% of baseline |
| p95 latency | | | | ≤ 100 ms |
| Share escalated | 100% | 0% | | ≤ 20% |

## Intended use
Routing English retail-banking support messages to one of the listed intents, with escalation of uncertain cases.

## Not intended for
Other domains or languages without retraining; decisions with legal or financial effect on a customer without human review.

## Known failure modes
_Fill from the confusion analysis and reviewer notes._
