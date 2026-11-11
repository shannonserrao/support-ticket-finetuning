"""Batch inference with calibrated confidence and routing. Returns the predictions path.

Implements specs/06-serving-and-routing.md sections 2-4 (T-08, T-09). The threshold fit itself is in routing.py.
"""
from __future__ import annotations

import json
import logging
import os
import time

from .evaluate import _threshold
from .feedback import read_jsonl
from .routing import fit_threshold

log = logging.getLogger("triage.predict")


def route(confidence: float, threshold: float) -> str:
    """FR-4: a pure function of confidence and threshold."""
    return "small" if confidence >= threshold else "escalate"


def _percentile(sorted_values: list[float], q: float) -> float:
    return sorted_values[min(len(sorted_values) - 1, int(q * len(sorted_values)))]


def predict(cfg: dict, tracker) -> str:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from . import modeling

    paths, pcfg, fcfg = cfg["paths"], cfg["predict"], cfg["feedback"]
    model_dir = os.path.join(paths["artifacts"], "model")
    if not os.path.exists(os.path.join(model_dir, "calibration.json")):
        raise FileNotFoundError("run `make train` before `make predict`")
    split = pcfg.get("split", "test")
    rows = read_jsonl(os.path.join(paths["data"], "processed", f"{split}.jsonl"))
    if not rows:
        raise FileNotFoundError("run `make data` before `make predict`")
    with open(os.path.join(model_dir, "labels.json"), encoding="utf-8") as handle:
        labels = json.load(handle)
    with open(os.path.join(model_dir, "calibration.json"), encoding="utf-8") as handle:
        temperature = float(json.load(handle)["temperature"])
    with open(os.path.join(model_dir, "training_meta.json"), encoding="utf-8") as handle:
        model_version = json.load(handle)["model_version"]

    device = modeling.pick_device(cfg["train"].get("device", "auto"))
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device).eval()
    max_length, texts = cfg["model"]["max_length"], [r["text"] for r in rows]

    # Routing threshold (spec 06 section 4). With routing.select = validation it is fitted on validation
    # predictions and cached baseline answers for validation; the split being predicted is not involved.
    if cfg["routing"].get("select") == "validation":
        val_rows = read_jsonl(os.path.join(paths["data"], "processed", "validation.jsonl"))
        val_logits = modeling.compute_logits(model, tokenizer, [r["text"] for r in val_rows], max_length,
                                             pcfg["batch_size"], device)
        val_pred, val_conf = modeling.confidences(val_logits, temperature)
        val_records = [{"id": r["id"], "label": labels[p], "confidence": round(c, 6), "model_version": model_version}
                       for r, p, c in zip(val_rows, val_pred, val_conf)]
        os.makedirs(paths["artifacts"], exist_ok=True)
        with open(os.path.join(paths["artifacts"], "predictions_validation.jsonl"), "w", encoding="utf-8",
                  newline="\n") as handle:
            for record in val_records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        fit = fit_threshold(cfg, val_records, model_version)
        if fit:
            tracker.log_metrics({"threshold": fit["threshold"], "threshold_target_met": int(fit["target_met"]),
                                 "val_cascade_accuracy": fit["validation"]["cascade_accuracy"],
                                 "val_escalation_rate": fit["validation"]["escalation_rate"]})
    threshold = _threshold(cfg)

    # Throughput over the whole split at predict.batch_size, tokenisation included.
    modeling.compute_logits(model, tokenizer, texts[:pcfg["batch_size"]], max_length, pcfg["batch_size"], device)
    started = time.perf_counter()
    logits = modeling.compute_logits(model, tokenizer, texts, max_length, pcfg["batch_size"], device)
    throughput = len(rows) / (time.perf_counter() - started)
    pred, conf = modeling.confidences(logits, temperature)

    # Latency: single-message calls after warm-up (spec 06 section 3).
    warmup, timed = pcfg.get("latency_warmup_calls", 20), pcfg.get("latency_calls", 200)
    latency_by_row: dict[int, float] = {}
    for call in range(warmup + timed):
        index = call % len(rows)
        t0 = time.perf_counter()
        modeling.confidences(modeling.compute_logits(model, tokenizer, [texts[index]], max_length, 1, device),
                             temperature)
        if call >= warmup:
            latency_by_row[index] = (time.perf_counter() - t0) * 1000
    latencies = sorted(latency_by_row.values())

    out_path = os.path.join(paths["artifacts"], "predictions.jsonl")
    os.makedirs(paths["artifacts"], exist_ok=True)
    records = []
    for i, row in enumerate(rows):
        confidence = round(conf[i], 6)
        records.append({"id": row["id"], "label": labels[pred[i]], "confidence": confidence,
                        "latency_ms": latency_by_row.get(i), "route": route(confidence, threshold),
                        "model_version": model_version})
    with open(out_path, "w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    # FR-6: every prediction goes to the feedback log, once per (id, model version).
    log_path = fcfg["predictions_log"]
    seen = {(r["id"], r["model_version"]) for r in read_jsonl(log_path)}
    new = [{"id": rec["id"], "input_ref": row["text"], "prediction": rec["label"], "confidence": rec["confidence"],
            "model_version": model_version}
           for rec, row in zip(records, rows) if (rec["id"], model_version) not in seen]
    if new:
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as handle:
            for item in new:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    measured = {
        "throughput_tickets_per_s": throughput,
        "latency_p50_ms": _percentile(latencies, 0.50),
        "latency_p95_ms": _percentile(latencies, 0.95),
        "escalation_rate": sum(r["route"] == "escalate" for r in records) / len(records),
    }
    if device.type == "cuda":
        measured["peak_gpu_memory_mb"] = torch.cuda.max_memory_allocated() / 2**20
    tracker.log_metrics(measured)
    tracker.log_params({"model_version": model_version, "device": modeling.device_name(device),
                        "precision": modeling.precision_name(device), "threshold": threshold,
                        "temperature": temperature, "predict_rows": len(rows)})
    tracker.log_artifact(out_path)
    log.info("%d predictions on %s (%s): %.0f tickets/s, latency p50 %.1f ms, p95 %.1f ms, escalated %.1f%%",
             len(rows), modeling.device_name(device), modeling.precision_name(device), throughput,
             measured["latency_p50_ms"], measured["latency_p95_ms"], 100 * measured["escalation_rate"])
    return out_path
