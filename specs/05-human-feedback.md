# Spec 05 — Human feedback loop

## Goal
People check the predictions the model is least sure of, plus a random sample, and their corrections become training data for the next model version.

## Flow
See diagram 3 in `docs/architecture.md`.
1. `predict` appends every prediction to `feedback/predictions_log.jsonl` with `id`, `input_ref`, `prediction`, `confidence`, `model_version`.
2. `make feedback-export` writes `feedback/review_queue.csv` containing:
   - every item with confidence below `feedback.review_threshold` (reason `low_confidence`);
   - a random `feedback.audit_rate` share of the rest (reason `random_audit`).
3. A reviewer opens the CSV in a spreadsheet and fills in `verdict`, `corrected`, `reviewer`, `note`. `input_ref` is the ticket text.
4. Save as `feedback/reviewed.csv` and run `make feedback-ingest`. Valid rows are appended to `feedback/corrections.jsonl`, which is committed.
5. The next `make data` merges corrections into the training split and bumps the data version (task T-13).

## Requirements
- **HF-1** `verdict` is one of `correct`, `incorrect`, `unsure`.
- **HF-2** `incorrect` requires `corrected`: the correct intent label (must be in `data/labels.json`), or `NEW:<name>` to propose a label.
- **HF-3** `reviewer` is mandatory. Rows with a blank verdict are skipped, not rejected.
- **HF-4** An item already in `corrections.jsonl` is never queued or ingested twice.
- **HF-5** Corrections are added only to the training split. Corrected items that came from the test split are excluded from training and reported separately.
  - The label is `corrected` for an `incorrect` verdict and the model's `prediction` for a `correct` verdict. `unsure` rows are skipped.
  - An item whose id or text is in the test split is never added; the count and ids are logged as a warning and returned by `build_dataset`.
  - An item whose id or text is in the validation split is skipped, so validation stays fixed and no text is shared between splits.
  - A text already in train is relabelled in place. Any other item is appended to train with id `feedback-<id>`.
  - `NEW:<name>` proposals and labels not in `data/labels.json` are skipped and counted; a label-set change is a separate, deliberate step.
  - `corrections_applied` in the manifest is rows added plus rows relabelled. The data version bumps only when a split hash changes.
- **HF-6** The random-audit sample is the only unbiased estimate of live accuracy. Low-confidence items must never be used to estimate accuracy.
- **HF-7** At least 10% of reviewed items are reviewed by a second person; inter-reviewer agreement is reported.

## Section 4 — Promotion gate
A `challenger` model becomes `champion` only when agreement on the random-audit sample is at least 95% over 200 or more reviewed tickets, and no release gate in `SPEC.md` section 5 has regressed.

## Reports
`make feedback-ingest` prints: items added, agreement overall, agreement on the audit sample. Log these to the tracker with each retrain.

## Implemented in
`src/triage/feedback.py`, tested in `tests/test_feedback.py`. Open work: T-13 (merge into data), T-14 (optional review UI).
