# Spec 06 — Inference, routing and serving

## 1. Goal
Run the model in batch with calibrated confidence and decide per ticket whether to escalate.

## 2. Batch inference (`predict.py`)
- Load `artifacts/model/`, including `calibration.json`.
- Sort by length and batch by `predict.batch_size` for throughput; restore the original order on output.
- Confidence = maximum of softmax(logits / temperature).
- Write `artifacts/predictions.jsonl` (format in spec 03, plus `route` and `model_version`) and append to `feedback/predictions_log.jsonl`. A prediction already in the log with the same `id` and `model_version` is not appended again.
- `latency_ms` is set only on the rows used for the single-message latency measurement (section 3) and is `null` elsewhere.

## 3. Performance measurement
- Throughput: tickets per second over the whole test set at `predict.batch_size`, tokenisation included.
- Latency: p50 and p95 over `predict.latency_calls` (200) single-message calls after `predict.latency_warmup_calls` (20) warm-up calls, tokenisation included.
- Record device name and precision. Log all of it to the tracker.

## 4. Routing
- `route = "small"` if confidence is at least the threshold, else `"escalate"`.
- Threshold selection (`routing.select: validation`): sweep thresholds on validation, compute cascade accuracy using cached baseline answers for validation items, pick the threshold with the lowest projected cost that meets `routing.target_accuracy`. Save to `artifacts/threshold.json`.
- `routing.select: fixed` uses `routing.confidence_threshold` as given.

## 5. Serving (stretch, T-17)
`POST /triage` with `{"text": "..."}` returns the FR-1 payload. One process, model loaded once, request batching optional. No authentication in the proof of concept; do not expose it publicly.

## 6. Acceptance
- Predictions are identical for batch sizes 1 and 64: same labels, confidence equal within 1e-5 (padding changes floating-point rounding).
- The routing decision is a pure function of confidence and threshold (unit test).
