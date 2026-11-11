"""Runs the Databricks baseline notebook offline against a stand-in endpoint, then imports its results.

Covers the whole hand-off: bundle -> notebook -> result files -> baseline-import -> threshold fit -> evaluate.
`dbutils`, `displayHTML` and `mlflow` are replaced by small fakes; nothing here proves the real endpoint call.
"""
import json
import os
import sys
import types

import pytest
from conftest import REPO, make_rows, oracle_answer, write_jsonl

from triage import baseline as B
from triage import routing
from triage.evaluate import evaluate_files
from triage.tracking import Tracker

NOTEBOOK = os.path.join(REPO, "platforms", "databricks", "baseline", "baseline_zero_shot.py")


class FakeWidgets:
    def __init__(self, values):
        self.values, self.defaults = values, {}

    def text(self, name, default, label=None):
        self.defaults[name] = default

    def dropdown(self, name, default, choices, label=None):
        self.defaults[name] = default

    def get(self, name):
        return self.values.get(name, self.defaults[name])


@pytest.fixture()
def endpoint_calls(monkeypatch):
    """Install a fake `mlflow` whose deployment client answers like a chat endpoint. Returns the list of calls."""
    calls = []

    class Client:
        def predict(self, endpoint, inputs):
            calls.append(endpoint)
            assert inputs["max_tokens"] == 20 and inputs["temperature"] == 0.0
            text = inputs["messages"][1]["content"].split("Customer message: ")[1].split("\n")[0]
            return {"choices": [{"message": {"content": oracle_answer(text)}}],
                    "usage": {"prompt_tokens": 1000, "completion_tokens": 5}}

    class Run:
        info = types.SimpleNamespace(run_id="dbx-run-1")

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    mlflow = types.ModuleType("mlflow")
    deployments = types.ModuleType("mlflow.deployments")
    deployments.get_deploy_client = lambda target: Client()
    mlflow.deployments = deployments
    mlflow.start_run = lambda run_name=None: Run()
    mlflow.set_tags = mlflow.log_params = mlflow.log_metrics = mlflow.log_artifact = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "mlflow", mlflow)
    monkeypatch.setitem(sys.modules, "mlflow.deployments", deployments)
    monkeypatch.setattr(sys, "path", list(sys.path))          # the notebook prepends the bundle's src folder
    return calls


def run_notebook(project, **widgets):
    """Execute the notebook source the way Databricks would, cell after cell. Returns (outputs dir, html shown)."""
    base = os.path.dirname(project["baseline"]["bundle_dir"])
    values = {"endpoint": "big-model", "usd_per_1m_input_tokens": "2", "usd_per_1m_output_tokens": "10",
              "limit": "0", "base_dir": base, **widgets}
    shown = []
    scope = {"dbutils": types.SimpleNamespace(widgets=FakeWidgets(values)), "displayHTML": shown.append,
             "__name__": "__notebook__"}
    with open(NOTEBOOK, encoding="utf-8") as handle:
        source = handle.read()
    assert source.startswith("# Databricks notebook source")
    exec(compile(source, NOTEBOOK, "exec"), scope)
    return os.path.join(base, "outputs"), "\n".join(shown)


def test_notebook_to_report_end_to_end(project, endpoint_calls):
    B.write_bundle(project)

    out, shown = run_notebook(project, splits="validation")
    assert len(endpoint_calls) == 1 + 30                       # one probe call, then every validation ticket
    assert "0.7000" in shown                                   # validation accuracy 21/30 is displayed
    assert json.load(open(os.path.join(out, B.SUMMARY_FILE)))["mlflow_run_id"] == "dbx-run-1"

    out, shown = run_notebook(project, splits="validation,test", prompt_frozen="yes")
    assert len(endpoint_calls) == 31 + 1 + 21                  # validation answers came from the cache
    assert "computed by `make eval`" in shown                  # no test accuracy in the notebook
    assert not any("label" in json.loads(line) for line in open(os.path.join(os.path.dirname(out), "inputs",
                                                                              "test.jsonl")))

    with Tracker(project, "baseline", use_mlflow=False) as tracker:
        result = B.import_results(project, out, tracker)
    assert result["model"] == "big-model" and result["remote_run_id"] == "dbx-run-1"
    assert result["splits"]["test"]["usd_per_ticket"] == pytest.approx(0.00205)

    # The small model's predictions: right and confident, so the cascade keeps everything on the small model.
    val = [{"id": r["id"], "label": r["label"], "confidence": 0.9} for r in make_rows("validation", 30)]
    fit = routing.fit_threshold(project, val, "model-v1")
    assert fit["baseline_model"] == "big-model" and fit["threshold"] == 0.0
    write_jsonl(os.path.join(project["paths"]["artifacts"], "predictions.jsonl"),
                [{"id": r["id"], "label": r["label"], "confidence": 0.9, "latency_ms": None}
                 for r in make_rows("test", 21)])
    report = evaluate_files(project)
    assert report["metrics"]["baseline_accuracy"] == pytest.approx(14 / 21)
    assert report["metrics"]["cascade_accuracy"] == 1.0 and len(report["cascade"]["curve"]) == 102


def test_partial_run_is_marked_and_cannot_be_imported(project, endpoint_calls):
    B.write_bundle(project)
    out, shown = run_notebook(project, splits="validation", limit="10")
    assert len(endpoint_calls) == 11 and "PARTIAL RUN" in shown
    assert os.path.exists(os.path.join(out, B.predictions_file("validation") + ".partial"))
    assert not os.path.exists(os.path.join(out, B.predictions_file("validation")))
    with pytest.raises(ValueError, match="partial run"):
        B.import_results(project, out)


@pytest.mark.parametrize(("widgets", "message"), [
    ({"splits": "validation,test"}, "prompt_frozen = yes"),
    ({"splits": "validation,test", "prompt_frozen": "yes", "limit": "5"}, "only in full"),
    ({"endpoint": " "}, "endpoint"),
])
def test_notebook_refuses_unsafe_settings_before_calling_the_model(project, endpoint_calls, widgets, message):
    B.write_bundle(project)
    with pytest.raises(ValueError, match=message):
        run_notebook(project, **widgets)
    assert endpoint_calls == []


def test_notebook_explains_a_missing_bundle(project, endpoint_calls):
    with pytest.raises(FileNotFoundError, match="baseline-bundle"):
        run_notebook(project)
