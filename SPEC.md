# SPEC — Support-Ticket Triage

## 1. Problem
Support teams read each incoming message to decide which queue it belongs to. Sending every message to a large hosted LLM works but is slow, costly at volume, and sends customer text to a third party. Most tickets are easy. We want a small in-house model to handle those and to know when it should not.

## 2. Users
- **Support operations lead:** wants tickets in the right queue faster, with fewer misroutes.
- **ML engineer (you):** wants a reproducible fine-tuning and evaluation pipeline to learn from.
- **Reviewer:** corrects uncertain predictions in a spreadsheet.

## 3. Scope
In scope:
- Single-label intent classification of short English support messages.
- Confidence score per prediction and a threshold-based cascade to an expensive model.
- Detection of out-of-scope messages (second dataset, stretch).
- Batch inference; a minimal HTTP endpoint is a stretch task.

Out of scope: reply generation, multi-label tagging, languages other than English, production deployment.

## 4. Functional requirements
- **FR-1** Given a message, return `{label, confidence, model_version, route}` where `route` is `small` or `escalate`.
- **FR-2** Labels come from a fixed, versioned label set (`data/labels.json`).
- **FR-3** `confidence` is calibrated: temperature scaling fitted on the validation split.
- **FR-4** Messages with confidence below `routing.confidence_threshold` are marked `escalate`.
- **FR-5** The pipeline runs unchanged on local, Colab, Kaggle and Databricks.
- **FR-6** Every prediction is logged for human review (`specs/05-human-feedback.md`).
- **FR-7** A zero-shot baseline with the expensive model is evaluated on the identical test split.

## 5. Success metrics and release gates
Measured on the untouched test split. Baseline = zero-shot expensive model.

| Metric | Definition | Gate for release |
|---|---|---|
| Accuracy | Correct / total | At least baseline accuracy, and at least 0.92 on Banking77 |
| Macro F1 | Mean per-class F1 | At least baseline macro F1 |
| Expected calibration error | 15 equal-width bins | At most 0.05 after calibration |
| Selective accuracy at 80% coverage | Accuracy on the 80% most confident | At least 0.97 |
| Cost per 1,000 tickets | From `cost` config: GPU-hour price and measured throughput, or tokens times price | At most 10% of baseline |
| p95 latency | Single message, batch size 1 | At most 100 ms on the target GPU |
| Cascade accuracy | Small model plus escalation at the chosen threshold | At least baseline accuracy while escalating at most 20% |

The 0.92 figure is a target chosen from published fine-tuned results on this dataset, not a measured result of this project. Record the measured baseline in `docs/model_card.md` before training and adjust gates only through a `CHANGELOG.md` entry.

## 6. Business value model
Inputs, set in `configs/base.yaml` under `cost` and `business`:
- tickets per month, baseline cost per ticket, small-model cost per ticket, escalation rate;
- minutes of agent time lost per misrouted ticket, agent cost per hour, misroute rate before and after.

Outputs produced by `evaluate` in `report.json`:
- monthly inference cost for baseline, small-only and cascade;
- monthly cost of misroutes for each;
- net monthly saving of the cascade against the baseline.

These are projections from test-set accuracy and stated volumes. Label them as projections wherever they are shown.

## 7. Non-functional requirements
- **NFR-1 Reproducibility:** fixed seeds; a rerun of the same config, data version and commit gives accuracy within 0.5 points.
- **NFR-2 Compute:** full training in under 30 minutes on a single 16 GB GPU; smoke run in under 3 minutes on CPU.
- **NFR-3 Privacy:** no message text leaves the environment except in the explicit baseline and escalation calls.
- **NFR-4 Traceability:** every reported number has a run ID.

## 8. Risks
| Risk | Mitigation |
|---|---|
| Public data is cleaner than real tickets | Report this limit; add real tickets through the feedback loop |
| Baseline prompt is weak, making the comparison unfair | Fix the prompt before looking at test results; include label descriptions; record it in the run |
| Threshold tuned on test data | Tune on validation only (enforced in `evaluate.py`) |
| Label set changes in production | `NEW:<name>` proposals in review; a label-set change is a MAJOR version |

## 9. Open questions
- Which expensive model is the baseline, and at what price? Set in `configs/base.yaml: baseline`.
- Is out-of-scope detection (CLINC150) required for the first release? Default: stretch.
