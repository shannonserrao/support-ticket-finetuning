# Changelog

All notable changes to specs, data, code and models. Newest first.

## [Unreleased]
- New config key `train.micro_batch_size` (default off) and `configs/platform/local_4gb.yaml`: batches are processed in chunks with gradient accumulation so the base configuration fits a 4 GB GPU. Spec 02 §3 updated. Batch size, step count and schedule are unchanged.
- T-13: `data.merge_corrections` merges reviewed corrections into the training split only. Spec 05 HF-5 now spells out the cases (test items excluded and reported, validation items skipped, existing train text relabelled, new items appended as `feedback-<id>`, `NEW:` proposals skipped). Spec 01 §3 step 5 reads the path from `feedback.corrections`.
- T-06/T-07: implemented `train.py` (custom loop: AdamW, linear warm-up and decay, mixed precision on CUDA, best-macro-F1 checkpoint, early stopping, resumable `artifacts/checkpoints/last.pt`, temperature scaling on validation) and `modeling.py` (shared device choice, length-sorted batched inference, temperature fit). Spec 02 §2–3 updated: checkpoint location and reuse rule, `model_version` format, final evaluation when the last step was not evaluated. New config key `train.resume`.
- T-08: implemented `predict.py`. Spec 06 §2–3 and §6 and spec 03 §2 updated: `route` and `model_version` in `predictions.jsonl`, `latency_ms` null outside the latency sample, feedback log de-duplicated on (`id`, `model_version`), batch-size invariance stated as same labels and confidence within 1e-5. New config keys `predict.latency_warmup_calls`, `predict.latency_calls`.
- `configs/smoke.yaml` now also writes artifacts and the prediction log under `artifacts/smoke`, so a smoke run cannot overwrite the real model, predictions or feedback log.
- T-05 parked: no baseline model available. Request-volume estimate (about 5,000, provision 6,000) recorded in `TASKS.md`. T-06 to T-08 proceed without it.
- T-02: pinned dataset and model revisions in `configs/base.yaml`; added `data.oos_revision`, `data.oos_config`, `data.oos_label_column`, `model.revision`; added `scripts/verify_sources.py` (`make verify-sources`); licences and counts recorded in `docs/data_card.md`. Spec 01 §2 unchanged: counts match (10,003 / 3,080 / 77 labels).
- T-03/T-04: implemented `data.py` (load parquet via `huggingface_hub` + `pyarrow`, no `datasets` dependency; pure-Python stratified split ordered by a seeded hash of the id). Spec 01 §6 coverage check is skipped when `max_*_samples` is set, because a 64-row smoke subset cannot cover 77 labels. `data_version` and `created` are kept when no hash changes, so repeated runs give an identical manifest.
- `configs/smoke.yaml` now writes to `data/smoke` so smoke runs cannot overwrite the real dataset or manifest. `cli.py` enables INFO logging.
- Initial scaffold: specs, plan, tasks, architecture diagrams, config loader, metrics, evaluation, tracking and feedback modules with tests.
- Stubs awaiting implementation: `data.py`, `train.py`, `predict.py` (see `TASKS.md`).
