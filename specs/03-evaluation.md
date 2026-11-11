# Spec 03 — Evaluation

## 1. Goal
One command produces every number in `SPEC.md` section 5, for the small model, the baseline and the cascade, from files on disk.

## 2. Inputs
- `artifacts/predictions.jsonl`: `{"id", "label", "confidence", "latency_ms"}` per line; optional `probs`, `route`, `model_version`. `latency_ms` may be `null` (spec 06 §2).
- `data/processed/test.jsonl`: gold labels.
- Optional `artifacts/baseline_predictions.jsonl` in the same format, plus `cost_usd` per line.

## 3. Metrics (implemented in `metrics.py`)
| Name | Function | Notes |
|---|---|---|
| Accuracy | `accuracy` | |
| Macro F1 | `macro_f1` | Classes with no gold and no predicted examples are skipped |
| Per-class precision, recall, F1 | `per_class` | Sorted worst first in the report |
| Expected calibration error | `expected_calibration_error` | 15 equal-width bins |
| Selective accuracy and coverage | `selective_accuracy` | At a threshold |
| Coverage curve | `coverage_curve` | Thresholds from 0.50 to 0.99 |
| Cascade | `cascade` | Small model above threshold, baseline answers below |
| Cost per 1,000 | `cost_per_1k` | From the `cost` config |
| Monthly projection | `business_projection` | From the `business` config |

## 4. Baseline protocol
- Prompt contains the label list with a one-line description per label and asks for exactly one label.
- The prompt is frozen and committed before the first test-set run.
- Responses are cached on disk keyed by model, prompt hash and message id, so reruns cost nothing.
- Unparseable answers count as wrong and are reported as a separate rate.

## 5. Statistical care
- Report a 95% bootstrap confidence interval for accuracy and macro F1 (1,000 resamples).
- For the final configuration report the mean and standard deviation over three seeds.
- Compare small model and baseline with a paired bootstrap on the same test items.

## 6. Report
`artifacts/report.json`: `{"metrics": {...}, "per_class": [...], "confusions": [...], "coverage_curve": [...], "cascade": {...}, "business": {...}, "meta": {...}}`.

## 7. Rules
- The threshold is read from config or from `artifacts/threshold.json`, which is fitted on validation. `evaluate` never fits anything on test.
- A prediction file whose ids do not exactly match the gold ids is an error.

## 8. Acceptance
`tests/test_metrics.py` and `tests/test_evaluate.py` pass; hand-computed examples match.
