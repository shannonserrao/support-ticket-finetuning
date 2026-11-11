"""Threshold selection on validation and cascade reporting (spec 06 §4, task T-09). No PyTorch needed."""
import json
import os

import pytest
from conftest import LABELS, make_rows, oracle_answer, write_jsonl

from triage import metrics as M
from triage import routing
from triage.baseline import parse_label
from triage.evaluate import evaluate_files


def baseline_rows(split, n, cost=0.002):
    return [{"id": r["id"], "label": parse_label(oracle_answer(r["text"]), LABELS), "cost_usd": cost,
             "model": "big-model", "prompt_hash": "abc"} for r in make_rows(split, n)]


def small_rows(split, n, wrong=(1, 2, 3)):
    """Small model: wrong with low confidence on a few tickets, right with high confidence elsewhere."""
    rows = []
    for i, r in enumerate(make_rows(split, n)):
        label = LABELS[(LABELS.index(r["label"]) + 1) % 3] if i in wrong else r["label"]
        rows.append({"id": r["id"], "label": label, "confidence": 0.30 if i in wrong else 0.90 + i / 1000,
                     "latency_ms": None})
    return rows


def with_baseline(project, splits=("validation",)):
    names = {"validation": routing.BASELINE_VALIDATION_FILE, "test": "baseline_predictions.jsonl"}
    for split in splits:
        n = 30 if split == "validation" else 21
        write_jsonl(os.path.join(project["paths"]["artifacts"], names[split]), baseline_rows(split, n))
    return project


def test_select_threshold_takes_the_cheapest_point_that_meets_the_target():
    curve = [{"threshold": 0.0, "accuracy": 0.90, "escalation_rate": 0.0},
             {"threshold": 0.5, "accuracy": 0.95, "escalation_rate": 0.1},
             {"threshold": 0.7, "accuracy": 0.95, "escalation_rate": 0.2},
             {"threshold": 0.9, "accuracy": 0.97, "escalation_rate": 0.5}]
    assert M.select_threshold(curve, 0.95)["threshold"] == 0.5
    assert M.select_threshold(curve, 0.80)["threshold"] == 0.0
    best = M.select_threshold(curve, 0.99)
    assert best["threshold"] == 0.9 and best["target_met"] is False


def test_sweep_covers_no_escalation_and_full_escalation():
    grid = M.sweep_thresholds(0.01)
    assert grid[0] == 0.0 and grid[100] == 1.0 and grid[-1] > 1 and len(grid) == 102
    curve = M.cascade_curve([True, False], [0.9, 0.2], [False, True], grid, 0.001, 0.01)
    assert curve[0]["escalation_rate"] == 0 and curve[-1]["escalation_rate"] == 1
    assert curve[-1]["usd_per_ticket"] == pytest.approx(0.011)


def test_default_target_is_baseline_accuracy_on_validation(project):
    fit = routing.fit_threshold(with_baseline(project), small_rows("validation", 30), "model-v1")
    assert fit["target_accuracy"] == pytest.approx(21 / 30)          # baseline accuracy on validation
    assert fit["threshold"] == 0.0 and fit["validation"]["escalation_rate"] == 0   # small model alone already beats it
    assert fit["validation"]["small_accuracy"] == pytest.approx(27 / 30)
    assert fit["model_version"] == "model-v1" and fit["baseline_model"] == "big-model"


def test_explicit_target_escalates_only_the_low_confidence_tickets(project):
    project["routing"]["target_accuracy"] = 1.0
    fit = routing.fit_threshold(with_baseline(project), small_rows("validation", 30))
    assert fit["threshold"] == pytest.approx(0.31) and fit["target_met"]
    assert fit["validation"]["cascade_accuracy"] == 1.0
    assert fit["validation"]["escalation_rate"] == pytest.approx(3 / 30)
    assert fit["validation"]["usd_per_ticket"] is None               # small-model throughput not set yet
    saved = json.load(open(os.path.join(project["paths"]["artifacts"], routing.THRESHOLD_FILE)))
    assert saved["threshold"] == fit["threshold"] and len(saved["curve"]) == 102


def test_threshold_is_fitted_without_reading_any_test_file(project, monkeypatch):
    """T-09 check: there is no test file to read, and every file that is read is a validation file."""
    with_baseline(project)
    os.remove(os.path.join(project["paths"]["data"], "processed", "test.jsonl"))
    opened = []
    real = routing.read_jsonl
    monkeypatch.setattr(routing, "read_jsonl", lambda path: opened.append(path) or real(path))
    fit = routing.fit_threshold(project, small_rows("validation", 30))
    assert fit is not None
    assert len(opened) == 2 and all("validation" in os.path.basename(p) for p in opened)
    assert not os.path.exists(os.path.join(project["paths"]["artifacts"], "baseline_predictions.jsonl"))


def test_without_baseline_answers_the_fixed_threshold_is_used_and_stale_fit_removed(project):
    stale = os.path.join(project["paths"]["artifacts"], routing.THRESHOLD_FILE)
    os.makedirs(os.path.dirname(stale))
    open(stale, "w").write(json.dumps({"threshold": 0.11}))
    assert routing.fit_threshold(project, small_rows("validation", 30)) is None
    assert not os.path.exists(stale)


def test_mismatched_validation_ids_are_an_error(project):
    with pytest.raises(ValueError, match="validation ids differ"):
        routing.fit_threshold(with_baseline(project), small_rows("validation", 29))


def test_report_contains_cascade_curve_and_chosen_threshold(project):
    """T-09 check: threshold fitted on validation, cascade evaluated on test with cached baseline answers."""
    project["routing"]["target_accuracy"] = 1.0
    project["cost"]["small_tickets_per_second"] = 500
    with_baseline(project, ("validation", "test"))
    fit = routing.fit_threshold(project, small_rows("validation", 30))
    write_jsonl(os.path.join(project["paths"]["artifacts"], "predictions.jsonl"), small_rows("test", 21))
    report = evaluate_files(project)
    assert report["meta"]["threshold"] == fit["threshold"] and report["meta"]["threshold_source"] == "validation"
    assert report["meta"]["threshold_fit"]["target_accuracy"] == 1.0 and "curve" not in report["meta"]["threshold_fit"]
    cascade = report["cascade"]
    assert cascade["threshold"] == fit["threshold"] and len(cascade["curve"]) == 102
    assert cascade["accuracy"] == 1.0 and cascade["escalation_rate"] == pytest.approx(3 / 21)
    metrics = report["metrics"]
    assert metrics["baseline_unparseable_rate"] == pytest.approx(2 / 21)      # tickets 7 and 17
    assert metrics["baseline_accuracy"] == pytest.approx(14 / 21)
    assert metrics["baseline_usd_per_1k"] == pytest.approx(2.0)               # measured from cost_usd per answer
    assert metrics["cascade_usd_per_1k"] < metrics["baseline_usd_per_1k"]
    assert report["business"]["net_monthly_saving_usd"] > 0


def test_fixed_threshold_is_reported_as_fixed(project):
    project["routing"]["select"] = "fixed"
    with_baseline(project, ("test",))
    write_jsonl(os.path.join(project["paths"]["artifacts"], "predictions.jsonl"), small_rows("test", 21))
    report = evaluate_files(project)
    assert report["meta"] == {"n": 21, "threshold": 0.8, "threshold_source": "fixed"}
    assert report["cascade"]["threshold_source"] == "fixed"
