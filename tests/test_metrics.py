import pytest

from triage import metrics as M

GOLD = ["a", "a", "b", "b", "c", "c"]
PRED = ["a", "b", "b", "b", "c", "a"]


def test_accuracy_and_macro_f1():
    assert M.accuracy(GOLD, PRED) == pytest.approx(4 / 6)
    # a: p=1/2 r=1/2 f1=.5 | b: p=2/3 r=1 f1=.8 | c: p=1 r=1/2 f1=2/3
    assert M.macro_f1(GOLD, PRED) == pytest.approx((0.5 + 0.8 + 2 / 3) / 3)


def test_confusions():
    top = M.top_confusions(GOLD, PRED)
    assert {"gold": "a", "pred": "b", "count": 1} in top and len(top) == 2


def test_ece_perfect_and_overconfident():
    assert M.expected_calibration_error([True, True], [1.0, 1.0]) == pytest.approx(0.0)
    assert M.expected_calibration_error([False, False], [0.9, 0.9]) == pytest.approx(0.9)


def test_selective_and_cascade():
    correct = [True, True, False, False]
    conf = [0.9, 0.8, 0.6, 0.3]
    sel = M.selective_accuracy(correct, conf, 0.7)
    assert sel["coverage"] == 0.5 and sel["accuracy"] == 1.0
    casc = M.cascade(correct, conf, [True, True, True, False], 0.7)
    assert casc["accuracy"] == 0.75 and casc["escalation_rate"] == 0.5
    assert M.threshold_for_coverage(conf, 0.5) == 0.8


def test_cost_and_projection():
    assert M.cost_per_1k(3.6, 1000) == pytest.approx(0.001)
    proj = M.business_projection(1000, 0.0, 0.01, 0.2, 0.9, 0.9, 6, 30)
    assert proj["baseline"]["inference_usd"] == pytest.approx(10.0)
    assert proj["cascade"]["inference_usd"] == pytest.approx(2.0)
    assert proj["net_monthly_saving_usd"] == pytest.approx(8.0)


def test_length_mismatch():
    with pytest.raises(ValueError):
        M.accuracy(["a"], [])
