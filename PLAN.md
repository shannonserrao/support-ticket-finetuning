# PLAN — Support-Ticket Triage

Diagrams are in `docs/architecture.md`.

## 1. Approach
Fine-tune a pretrained encoder with a classification head. Calibrate its probabilities. Use the top-class probability as confidence and escalate below a threshold.

## 2. Components
| Module | Responsibility | Spec |
|---|---|---|
| `data.py` | Download pinned dataset revision, make train/validation/test splits, write `data/manifest.json` and `labels.json`, merge corrections | 01 |
| `train.py` | Fine-tune encoder, save checkpoints, fit temperature on validation, log to tracker | 02 |
| `predict.py` | Batch inference, calibrated confidence, routing decision, write predictions and feedback log | 06 |
| `metrics.py` | Accuracy, macro F1, calibration error, selective accuracy, cost model | 03 |
| `evaluate.py` | Join predictions with gold, compute metrics, cascade curve, business projection | 03 |
| `tracking.py` | MLflow wrapper with JSONL fallback | 04 |
| `feedback.py` | Review queue export, correction ingest | 05 |
| `baseline.py` (to add) | Zero-shot classification with the expensive model, cached responses | 03 |

## 3. Model choices
| Role | Default | Notes |
|---|---|---|
| Primary | `answerdotai/ModernBERT-base` | Encoder, about 150M parameters; fits any 16 GB GPU. Confirm the ID resolves in T-02. |
| Small alternative | `distilbert-base-uncased` | For CPU smoke runs and as a size-versus-accuracy point |
| Stretch | LoRA on a 1–4B instruction model | Generative classification, to compare with the encoder |
| Baseline | Set in `baseline.model` | Zero-shot with label list and descriptions |

## 4. Data
- Primary: Banking77 (77 banking intents; about 10k train and 3k test messages). The published split has no validation set; carve 10% of train, stratified, with a fixed seed.
- Stretch: CLINC150 for out-of-scope detection.
- Dataset IDs and revisions are pinned in `configs/base.yaml` and verified in T-02.

## 5. Training
Cross-entropy, AdamW, linear warm-up and decay, early stopping on validation macro F1, mixed precision on GPU. Defaults in `configs/base.yaml`. Sweep learning rate and epochs only; three seeds for the final configuration.

## 6. Calibration and routing
Fit a single temperature on validation logits. Sweep thresholds on validation to build the coverage-versus-accuracy curve; choose the lowest-cost threshold whose cascade accuracy meets the gate. Apply that fixed threshold to test once.

## 7. Evaluation
`evaluate` reads `artifacts/predictions.jsonl` and `data/processed/test.jsonl`. It reports the metrics in `SPEC.md` section 5, a per-class table, the top confusions, the cascade curve and the business projection. Latency and throughput are measured by `predict` and stored with the predictions.

## 8. Versioning and release
See `CONTRIBUTING.md`. A release is a git tag, a data version and a registered model that passed the gates.

## 9. Order of work
Data, then baseline measurement, then training, calibration, evaluation, routing, feedback merge, release. Details in `TASKS.md`.
