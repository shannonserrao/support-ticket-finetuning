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
- Unparseable answers count as wrong and are reported as a separate rate. They are left out of the macro F1 mean as a class of their own, but still lower the recall of the gold class.
- The prompt template and the label descriptions live in `configs/baseline/`. The prompt hash is the SHA-256 of the rendered system prompt and user template.
- An answer is accepted when, after trimming quotes and punctuation, it equals one label (case-insensitive), or when exactly one label occurs in it as a whole token.
- Each answer record holds `id`, `label` (null when unparseable), `raw`, `unparseable`, `input_tokens`, `output_tokens`, `cost_usd`, `latency_ms`, `model`, `prompt_hash`.
- The baseline may run outside this repository (`platforms/databricks/baseline/`). `make baseline-bundle` writes the inputs: validation with labels, **test without labels**, the rendered prompt and a copy of `baseline.py`. `make baseline-import FROM=<folder>` accepts the results only if the prompt hash and data version match this checkout and every split is complete, then writes `artifacts/baseline_predictions.jsonl` (test), `artifacts/baseline_predictions_validation.jsonl` and `artifacts/baseline_summary.json`, and logs a run tagged `baseline=true`. Test accuracy of the baseline is computed only by `make eval`.
- The test split is run only in full and only after the prompt is committed.

## 5. Statistical care
- Report a 95% bootstrap confidence interval for accuracy and macro F1 (1,000 resamples).
- For the final configuration report the mean and standard deviation over three seeds.
- Compare small model and baseline with a paired bootstrap on the same test items.

## 6. Report
`artifacts/report.json`: `{"metrics": {...}, "per_class": [...], "confusions": [...], "coverage_curve": [...], "cascade": {...}, "business": {...}, "meta": {...}}`.

- `cascade` (present when baseline answers exist): accuracy and escalation rate at the threshold in use, `threshold_source` (`validation` or `fixed`) and `curve`, the cascade at every threshold of the sweep. The curve on test is for reporting only.
- `meta`: `n`, `threshold`, `threshold_source`, and `threshold_fit` (the validation fit without its curve) when the threshold came from validation.
- Baseline cost per ticket is `cost.baseline_usd_per_ticket`, or the mean `cost_usd` of the baseline answers when that is null.

## 7. Rules
- The threshold is read from config or from `artifacts/threshold.json`, which is fitted on validation. `evaluate` never fits anything on test.
- A prediction file whose ids do not exactly match the gold ids is an error.

## 8. Acceptance
`tests/test_metrics.py` and `tests/test_evaluate.py` pass; hand-computed examples match.
