# TASKS — Support-Ticket Triage

Work top to bottom. Tick the box in the commit that completes the task. Each task lists the spec it implements and the check that proves it is done.

## Done in the scaffold
- [x] **T-00** Repository structure, specs, configs, CI for GitHub and GitLab.
- [x] **T-01** `config.py`, `metrics.py`, `evaluate.py`, `tracking.py`, `feedback.py`, `cli.py` with unit tests. Check: `make lint test`.

## Data
- [x] **T-02** Verify dataset and model IDs resolve; pin dataset revisions in `configs/base.yaml`. Spec 01 §2. Check: a script prints row counts per split and they match `docs/data_card.md`.
- [x] **T-03** Implement `data.build_dataset`: download, stratified validation split, write `data/processed/{train,validation,test}.jsonl`, `data/labels.json`, `data/manifest.json`. Spec 01 §3–5. Check: `make data` twice gives identical SHA-256 values; new test `tests/test_data.py` passes.
- [x] **T-04** Data checks: no text overlap between splits, label coverage in every split, length statistics logged. Spec 01 §6. Check: `make data` fails loudly on an injected duplicate.

## Baseline
- [ ] **T-05** Add `baseline.py`: zero-shot classification with the expensive model, on-disk response cache, token and cost accounting. Spec 03 §4. Check: baseline run on test logged with `baseline=true`; numbers copied to `docs/model_card.md`.
  - **PARKED (2026-11-11):** no baseline model is provisioned yet. T-06 to T-08 do not need it; T-09 (cascade, threshold), T-10 (baseline column), T-11 and the cost numbers in T-12 do. Resume when `baseline.model` and pricing are set in `configs/base.yaml`.
  - **Provisioning estimate** (one request per ticket, cached by model + prompt hash + id, so seeds and reruns add nothing):
    - Test, all 3,080 rows: baseline accuracy, paired bootstrap, cascade.
    - Validation, all 1,000 rows: threshold selection (spec 06 §4).
    - Prompt development on validation only, about 2 to 3 iterations of 300 rows: about 900.
    - **Total about 5,000 requests; provision for 6,000** (retries, unparseable answers).
    - Tokens (assumption, measure on 20 rows first): prompt with 77 labels and one-line descriptions about 1.5k to 2k input tokens, reply under 10 tokens. That is roughly 8M to 10M input and under 60k output tokens in total, much less input if the gateway supports prefix caching.
    - Smoke runs need only 32 rows.
    - Before resuming: write the 77 one-line label descriptions (needed regardless of model) and freeze the prompt.

## Training
- [ ] **T-06** Implement `train.train`: tokenise, fine-tune, early stopping, checkpoints, tracker logging. Spec 02 §2–4. Check: `make smoke` finishes on CPU in under 3 minutes and logs a run.
- [ ] **T-07** Temperature scaling on validation; store temperature with the model. Spec 02 §5. Check: validation calibration error after scaling is lower than before, logged as two metrics.

## Inference and evaluation
- [ ] **T-08** Implement `predict.predict`: batch inference, calibrated confidence, `route`, latency and throughput measurement, write `artifacts/predictions.jsonl` and the feedback log. Spec 06 §2–3. Check: `make predict eval` produces `artifacts/report.json`.
- [ ] **T-09** Threshold selection on validation and cascade evaluation on test using cached baseline answers for escalated tickets. Spec 06 §4. Check: report contains the cascade curve and the chosen threshold; the threshold was fitted without reading the test file (unit test).
- [ ] **T-10** Three-seed final run; mean and standard deviation reported. Spec 03 §5.
- [ ] **T-11** Results table and cost-versus-accuracy chart generated from tracked runs into `docs/results.md`. Spec 04. Check: every number has a run ID.

## Release and feedback
- [ ] **T-12** Register the model, tag the release, fill in `docs/model_card.md`. `CONTRIBUTING.md` release checklist.
- [x] **T-13** Merge `feedback/corrections.jsonl` into the training split in `data.build_dataset`; bump data version. Spec 05 HF-5. Check: a correction changes the manifest hash and never enters the test split.
  - Done out of order (2026-11-11) while T-05 is parked and the T-06 to T-08 checks wait on the GPU run.

## Stretch
- [ ] **T-14** Minimal review UI or Label Studio/Argilla export.
- [ ] **T-15** Out-of-scope detection on CLINC150.
- [ ] **T-16** LoRA fine-tune of a small generative model as a comparison point.
- [ ] **T-17** FastAPI endpoint `POST /triage` returning the FR-1 payload.
- [ ] **T-18** Quantised or ONNX export and CPU latency comparison.
