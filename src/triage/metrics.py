"""Classification, calibration, routing and cost metrics (specs/03-evaluation.md)."""
from __future__ import annotations

from collections import Counter


def accuracy(gold: list[str], pred: list[str]) -> float:
    _check(gold, pred)
    return sum(g == p for g, p in zip(gold, pred)) / len(gold)


def per_class(gold: list[str], pred: list[str]) -> list[dict]:
    _check(gold, pred)
    tp, fp, fn = Counter(), Counter(), Counter()
    for g, p in zip(gold, pred):
        if g == p:
            tp[g] += 1
        else:
            fp[p] += 1
            fn[g] += 1
    rows = []
    for label in sorted(set(gold) | set(pred)):
        precision = tp[label] / (tp[label] + fp[label]) if tp[label] + fp[label] else 0.0
        recall = tp[label] / (tp[label] + fn[label]) if tp[label] + fn[label] else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append({"label": label, "precision": precision, "recall": recall, "f1": f1,
                     "support": tp[label] + fn[label]})
    return rows


def macro_f1(gold: list[str], pred: list[str], ignore: set[str] | None = None) -> float:
    """Mean per-class F1. Labels in `ignore` (for example an "unparseable" marker that only ever appears as a
    prediction) are left out of the mean; they still count as errors for the gold class."""
    rows = [r for r in per_class(gold, pred) if not ignore or r["label"] not in ignore]
    return sum(r["f1"] for r in rows) / len(rows)


def top_confusions(gold: list[str], pred: list[str], k: int = 10) -> list[dict]:
    pairs = Counter((g, p) for g, p in zip(gold, pred) if g != p)
    return [{"gold": g, "pred": p, "count": n} for (g, p), n in pairs.most_common(k)]


def expected_calibration_error(correct: list[bool], conf: list[float], bins: int = 15) -> float:
    if len(correct) != len(conf) or not conf:
        raise ValueError("correct and conf must be non-empty and the same length")
    total, n = 0.0, len(conf)
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, c in enumerate(conf) if (lo < c <= hi) or (b == 0 and c == 0.0)]
        if idx:
            acc = sum(correct[i] for i in idx) / len(idx)
            avg = sum(conf[i] for i in idx) / len(idx)
            total += len(idx) / n * abs(acc - avg)
    return total


def selective_accuracy(correct: list[bool], conf: list[float], threshold: float) -> dict:
    kept = [c for c, p in zip(correct, conf) if p >= threshold]
    return {
        "threshold": threshold,
        "coverage": len(kept) / len(conf),
        "accuracy": sum(kept) / len(kept) if kept else None,
    }


def coverage_curve(correct: list[bool], conf: list[float]) -> list[dict]:
    thresholds = [round(0.50 + 0.01 * i, 2) for i in range(50)]
    return [selective_accuracy(correct, conf, t) for t in thresholds]


def threshold_for_coverage(conf: list[float], coverage: float) -> float:
    """Confidence value such that about `coverage` of items are at or above it."""
    ordered = sorted(conf, reverse=True)
    k = max(1, min(len(ordered), round(coverage * len(ordered))))
    return ordered[k - 1]


def cascade(
    small_correct: list[bool], conf: list[float], big_correct: list[bool], threshold: float,
) -> dict:
    """Small model answers at or above the threshold; the big model answers the rest."""
    if not (len(small_correct) == len(conf) == len(big_correct)):
        raise ValueError("inputs must be the same length")
    n = len(conf)
    escalated = [p < threshold for p in conf]
    right = sum(b if e else s for s, b, e in zip(small_correct, big_correct, escalated))
    return {"threshold": threshold, "accuracy": right / n, "escalation_rate": sum(escalated) / n}


def sweep_thresholds(step: float = 0.01) -> list[float]:
    """0.00, 0.01, ..., 1.00 and one value above 1 (escalate everything)."""
    n = round(1 / step)
    return [round(i * step, 6) for i in range(n + 1)] + [1.01]


def cascade_curve(
    small_correct: list[bool], conf: list[float], big_correct: list[bool], thresholds: list[float] | None = None,
    small_usd_per_ticket: float | None = None, big_usd_per_ticket: float | None = None,
) -> list[dict]:
    """Cascade accuracy and escalation rate at each threshold, with projected cost per ticket when prices are known."""
    rows = []
    for threshold in thresholds or sweep_thresholds():
        row = cascade(small_correct, conf, big_correct, threshold)
        if small_usd_per_ticket is not None and big_usd_per_ticket is not None:
            row["usd_per_ticket"] = small_usd_per_ticket + row["escalation_rate"] * big_usd_per_ticket
        rows.append(row)
    return rows


def select_threshold(curve: list[dict], target_accuracy: float) -> dict:
    """Cheapest point of a cascade curve that reaches `target_accuracy` (spec 06 section 4).

    Cost is `usd_per_ticket` when present, otherwise the escalation rate (cost grows with escalation).
    If no point reaches the target, the most accurate point is returned with `target_met` false.
    """
    def cost(row: dict) -> float:
        return row.get("usd_per_ticket", row["escalation_rate"])

    feasible = [row for row in curve if row["accuracy"] >= target_accuracy - 1e-12]
    if feasible:
        best = min(feasible, key=lambda row: (cost(row), row["threshold"]))
    else:
        best = max(curve, key=lambda row: (row["accuracy"], -cost(row)))
    return {**best, "target_accuracy": target_accuracy, "target_met": bool(feasible)}


def cost_per_1k(gpu_usd_per_hour: float, items_per_second: float) -> float:
    if items_per_second <= 0:
        raise ValueError("items_per_second must be positive")
    return gpu_usd_per_hour / 3600.0 / items_per_second * 1000.0


def business_projection(
    tickets_per_month: int, small_usd_per_ticket: float, baseline_usd_per_ticket: float,
    escalation_rate: float, cascade_accuracy: float, baseline_accuracy: float,
    minutes_lost_per_misroute: float, agent_usd_per_hour: float,
) -> dict:
    """Projected monthly cost of the baseline and the cascade. A projection, not a measurement."""
    misroute_usd = minutes_lost_per_misroute / 60.0 * agent_usd_per_hour

    def monthly(inference_per_ticket, acc):
        inference = tickets_per_month * inference_per_ticket
        misroutes = tickets_per_month * (1 - acc) * misroute_usd
        return {"inference_usd": inference, "misroute_usd": misroutes, "total_usd": inference + misroutes}

    base = monthly(baseline_usd_per_ticket, baseline_accuracy)
    casc = monthly(small_usd_per_ticket + escalation_rate * baseline_usd_per_ticket, cascade_accuracy)
    return {"baseline": base, "cascade": casc,
            "net_monthly_saving_usd": base["total_usd"] - casc["total_usd"]}


def _check(gold, pred):
    if len(gold) != len(pred) or not gold:
        raise ValueError("gold and pred must be non-empty and the same length")
