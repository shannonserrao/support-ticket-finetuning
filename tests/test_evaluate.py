import pytest

from triage.config import load_config
from triage.evaluate import evaluate


def _cfg():
    cfg = load_config("configs/base.yaml", ["routing.select=fixed", "routing.confidence_threshold=0.7"])
    cfg["cost"].update(small_tickets_per_second=1000, baseline_usd_per_ticket=0.01)
    return cfg


def test_evaluate_with_baseline():
    gold = [{"id": str(i), "label": lab} for i, lab in enumerate("aabb")]
    pred = [
        {"id": "0", "label": "a", "confidence": 0.9, "latency_ms": 5},
        {"id": "1", "label": "a", "confidence": 0.8, "latency_ms": 6},
        {"id": "2", "label": "a", "confidence": 0.4, "latency_ms": 7},
        {"id": "3", "label": "b", "confidence": 0.95, "latency_ms": 8},
    ]
    base = [{"id": str(i), "label": lab} for i, lab in enumerate("aabb")]
    report = evaluate(gold, pred, _cfg(), base)
    m = report["metrics"]
    assert m["accuracy"] == 0.75
    assert m["baseline_accuracy"] == 1.0
    assert m["cascade_accuracy"] == 1.0          # the one error was low-confidence and escalated
    assert m["escalation_rate_at_threshold"] == 0.25
    assert report["business"]["net_monthly_saving_usd"] > 0


def test_id_mismatch_is_error():
    with pytest.raises(ValueError):
        evaluate([{"id": "1", "label": "a"}], [{"id": "2", "label": "a", "confidence": 1.0}], _cfg())
