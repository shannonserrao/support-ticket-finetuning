"""Threshold selection on the validation split (specs/06-serving-and-routing.md section 4, task T-09).

Reads only validation files: the gold labels, the small model's validation predictions and the cached baseline
answers for validation. It never opens a test file. Standard library only, so it is testable without PyTorch.
"""
from __future__ import annotations

import json
import logging
import os

from . import metrics as M
from .feedback import read_jsonl

log = logging.getLogger("triage.routing")

THRESHOLD_FILE = "threshold.json"
BASELINE_VALIDATION_FILE = "baseline_predictions_validation.jsonl"


def per_ticket_costs(cfg: dict, baseline_rows: list[dict]) -> tuple[float | None, float | None]:
    """(small model, baseline) cost per ticket in USD, or None where it is not known yet."""
    cost = cfg.get("cost", {})
    small = None
    if cost.get("small_tickets_per_second") and cost.get("gpu_usd_per_hour") is not None:
        small = M.cost_per_1k(cost["gpu_usd_per_hour"], cost["small_tickets_per_second"]) / 1000
    big = cost.get("baseline_usd_per_ticket")
    if big is None:
        measured = [r["cost_usd"] for r in baseline_rows if r.get("cost_usd") is not None]
        big = sum(measured) / len(measured) if measured else None
    return small, big


def fit_threshold(cfg: dict, validation_predictions: list[dict], model_version: str | None = None) -> dict | None:
    """Choose the routing threshold on validation and save it to `artifacts/threshold.json`.

    `validation_predictions` are the small model's `{"id", "label", "confidence"}` rows for the validation split.
    Returns None (and removes any stale threshold file) when there are no cached baseline answers for validation.
    """
    paths, rcfg = cfg["paths"], cfg["routing"]
    out_path = os.path.join(paths["artifacts"], THRESHOLD_FILE)
    baseline_rows = read_jsonl(os.path.join(paths["artifacts"], BASELINE_VALIDATION_FILE))
    if not baseline_rows:
        if os.path.exists(out_path):
            os.remove(out_path)      # a threshold fitted for an earlier model or baseline must not be reused silently
        log.warning("no baseline answers for validation (%s); using the fixed threshold %.2f",
                    BASELINE_VALIDATION_FILE, rcfg["confidence_threshold"])
        return None

    gold = {r["id"]: r["label"] for r in read_jsonl(os.path.join(paths["data"], "processed", "validation.jsonl"))}
    small = {r["id"]: r for r in validation_predictions}
    big = {r["id"]: r.get("label") for r in baseline_rows}
    if not (set(gold) == set(small) == set(big)):
        raise ValueError("validation ids differ between gold labels, small-model predictions and baseline answers")
    ids = sorted(gold)
    small_correct = [small[i]["label"] == gold[i] for i in ids]
    conf = [float(small[i]["confidence"]) for i in ids]
    big_correct = [big[i] == gold[i] for i in ids]

    baseline_accuracy = sum(big_correct) / len(ids)
    target = rcfg.get("target_accuracy")
    target = baseline_accuracy if target is None else float(target)
    small_usd, big_usd = per_ticket_costs(cfg, baseline_rows)
    curve = M.cascade_curve(small_correct, conf, big_correct, M.sweep_thresholds(rcfg.get("sweep_step", 0.01)),
                            small_usd, big_usd)
    chosen = M.select_threshold(curve, target)
    result = {
        "threshold": chosen["threshold"], "source": "validation",
        "target_accuracy": target, "target_met": chosen["target_met"],
        "validation": {"n": len(ids), "small_accuracy": sum(small_correct) / len(ids),
                       "baseline_accuracy": baseline_accuracy, "cascade_accuracy": chosen["accuracy"],
                       "escalation_rate": chosen["escalation_rate"], "usd_per_ticket": chosen.get("usd_per_ticket")},
        "model_version": model_version,
        "baseline_model": baseline_rows[0].get("model"), "baseline_prompt_hash": baseline_rows[0].get("prompt_hash"),
        "curve": curve,
    }
    os.makedirs(paths["artifacts"], exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    if not chosen["target_met"]:
        log.warning("no threshold reaches the target accuracy %.4f on validation; using the most accurate one", target)
    log.info("threshold %.2f chosen on validation: cascade accuracy %.4f, escalating %.1f%% (target %.4f)",
             chosen["threshold"], chosen["accuracy"], 100 * chosen["escalation_rate"], target)
    return result


def load_threshold_fit(cfg: dict) -> dict | None:
    """The saved fit without its curve, for reports. None when the threshold is fixed."""
    path = os.path.join(cfg["paths"]["artifacts"], THRESHOLD_FILE)
    if cfg["routing"].get("select") != "validation" or not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as handle:
        fit = json.load(handle)
    return {key: value for key, value in fit.items() if key != "curve"}
