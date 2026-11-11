"""Evaluation from files on disk (specs/03-evaluation.md). Never fits anything on test data."""
from __future__ import annotations

import json
import os

from . import metrics as M
from .baseline import UNPARSEABLE
from .feedback import read_jsonl
from .routing import load_threshold_fit, per_ticket_costs


def _threshold(cfg: dict) -> float:
    path = os.path.join(cfg["paths"]["artifacts"], "threshold.json")
    if cfg["routing"].get("select") == "validation" and os.path.exists(path):
        with open(path, encoding="utf-8") as handle:
            return float(json.load(handle)["threshold"])
    return float(cfg["routing"]["confidence_threshold"])


def evaluate(gold_rows: list[dict], pred_rows: list[dict], cfg: dict,
             baseline_rows: list[dict] | None = None, threshold_fit: dict | None = None) -> dict:
    gold_by_id = {r["id"]: r["label"] for r in gold_rows}
    pred_by_id = {r["id"]: r for r in pred_rows}
    if set(gold_by_id) != set(pred_by_id):
        missing = len(set(gold_by_id) - set(pred_by_id))
        extra = len(set(pred_by_id) - set(gold_by_id))
        raise ValueError(f"prediction ids do not match gold ids ({missing} missing, {extra} extra)")
    ids = sorted(gold_by_id)
    gold = [gold_by_id[i] for i in ids]
    pred = [pred_by_id[i]["label"] for i in ids]
    conf = [float(pred_by_id[i]["confidence"]) for i in ids]
    correct = [g == p for g, p in zip(gold, pred)]
    threshold = _threshold(cfg)

    out = {
        "accuracy": M.accuracy(gold, pred),
        "macro_f1": M.macro_f1(gold, pred),
        "ece": M.expected_calibration_error(correct, conf),
        "selective_accuracy_at_80": M.selective_accuracy(
            correct, conf, M.threshold_for_coverage(conf, 0.80))["accuracy"],
        "escalation_rate_at_threshold": 1 - M.selective_accuracy(correct, conf, threshold)["coverage"],
    }
    latencies = sorted(r["latency_ms"] for r in pred_rows if r.get("latency_ms") is not None)
    if latencies:
        out["latency_p95_ms"] = latencies[min(len(latencies) - 1, int(0.95 * len(latencies)))]

    report = {
        "metrics": out,
        "per_class": sorted(M.per_class(gold, pred), key=lambda r: r["f1"]),
        "confusions": M.top_confusions(gold, pred),
        "coverage_curve": M.coverage_curve(correct, conf),
        "meta": {"n": len(ids), "threshold": threshold,
                 "threshold_source": "validation" if threshold_fit else "fixed"},
    }
    if threshold_fit:
        report["meta"]["threshold_fit"] = threshold_fit

    if baseline_rows:
        base_by_id = {r["id"]: r.get("label") for r in baseline_rows}
        if set(base_by_id) != set(gold_by_id):
            raise ValueError("baseline ids do not match gold ids")
        # Unparseable baseline answers count as wrong and are reported as a separate rate (spec 03 section 4).
        base_pred = [base_by_id[i] if base_by_id[i] is not None else UNPARSEABLE for i in ids]
        base_correct = [g == p for g, p in zip(gold, base_pred)]
        out["baseline_accuracy"] = M.accuracy(gold, base_pred)
        out["baseline_macro_f1"] = M.macro_f1(gold, base_pred, ignore={UNPARSEABLE})
        out["baseline_unparseable_rate"] = sum(p == UNPARSEABLE for p in base_pred) / len(ids)
        small_usd, big_usd = per_ticket_costs(cfg, baseline_rows)
        casc = M.cascade(correct, conf, base_correct, threshold)
        out["cascade_accuracy"] = casc["accuracy"]
        out["cascade_escalation_rate"] = casc["escalation_rate"]
        # The curve on test is for reporting only. The threshold itself comes from validation or from config.
        report["cascade"] = {**casc, "threshold_source": report["meta"]["threshold_source"],
                             "curve": M.cascade_curve(correct, conf, base_correct, None, small_usd, big_usd)}
        if big_usd is not None:
            out["baseline_usd_per_1k"] = big_usd * 1000
        if small_usd is not None and big_usd is not None:
            biz = cfg.get("business", {})
            out["small_usd_per_1k"] = small_usd * 1000
            out["cascade_usd_per_1k"] = (small_usd + casc["escalation_rate"] * big_usd) * 1000
            report["business"] = M.business_projection(
                biz["tickets_per_month"], small_usd, big_usd,
                casc["escalation_rate"], casc["accuracy"], out["baseline_accuracy"],
                biz["minutes_lost_per_misroute"], biz["agent_usd_per_hour"],
            )
    return report


def evaluate_files(cfg: dict) -> dict:
    paths = cfg["paths"]
    split = cfg.get("predict", {}).get("split", "test")
    gold = read_jsonl(os.path.join(paths["data"], "processed", f"{split}.jsonl"))
    pred = read_jsonl(os.path.join(paths["artifacts"], "predictions.jsonl"))
    if not gold or not pred:
        raise FileNotFoundError("run `make data` and `make predict` before `make eval`")
    baseline = read_jsonl(os.path.join(paths["artifacts"], "baseline_predictions.jsonl")) or None
    if baseline and split != "test":
        baseline = None      # the cached baseline file holds test answers only
    return evaluate(gold, pred, cfg, baseline, load_threshold_fit(cfg))
