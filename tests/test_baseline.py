"""Baseline protocol (spec 03 §4): prompt, answer parsing, cache, cost, bundle and import. No network."""
import json
import os

import pytest
import yaml
from conftest import LABELS, REPO, make_rows, oracle_answer

from triage import baseline as B
from triage.tracking import Tracker


def fake_model(calls):
    def call(messages):
        calls.append(messages)
        text = messages[1]["content"].split("Customer message: ")[1].split("\n")[0]
        return {"text": oracle_answer(text), "input_tokens": 1000, "output_tokens": 5}
    return call


def rendered(project):
    return B.render_prompt(B.load_prompt(project), LABELS)


def test_committed_label_descriptions_are_complete_one_liners():
    with open(os.path.join(REPO, "configs", "baseline", "label_descriptions.yaml"), encoding="utf-8") as handle:
        descriptions = yaml.safe_load(handle)
    assert len(descriptions) == 77
    assert all(isinstance(v, str) and v.strip() and "\n" not in v.strip() for v in descriptions.values())
    labels_path = os.path.join(REPO, "data", "labels.json")
    if os.path.exists(labels_path):                      # present after `make data`; not committed
        assert set(descriptions) == set(json.load(open(labels_path)))


def test_prompt_lists_every_label_and_hash_tracks_content(project):
    prompt = rendered(project)
    for label in LABELS:
        assert f"- {label}: tickets about {label}" in prompt["system"]
    messages = B.build_messages(prompt, "my card is lost")
    assert messages[0]["role"] == "system" and "my card is lost" in messages[1]["content"]
    changed = B.load_prompt(project)
    changed["descriptions"]["alpha"] = "something else"
    assert B.render_prompt(changed, LABELS)["hash"] != prompt["hash"]


def test_descriptions_must_match_labels(project):
    with pytest.raises(ValueError, match="do not match"):
        B.render_prompt(B.load_prompt(project), LABELS + ["delta"])


@pytest.mark.parametrize(("answer", "expected"), [
    ("card_arrival", "card_arrival"), (" Card_Arrival.\n", "card_arrival"), ("`card_arrival`", "card_arrival"),
    ("Label: card_arrival", "card_arrival"), ("The label is card_arrival because it is late.", "card_arrival"),
    ("reverted_card_payment?", "reverted_card_payment?"),
    ("card_arrival or card_linking", None), ("card", None), ("", None), (None, None), ("no idea", None),
])
def test_parse_label(answer, expected):
    assert B.parse_label(answer, ["card_arrival", "card_linking", "reverted_card_payment?"]) == expected


def test_cache_makes_reruns_free_and_costs_are_counted(project, tmp_path):
    rows, calls = make_rows("validation", 30), []
    cache_path = str(tmp_path / "cache" / "responses.jsonl")
    first = B.run_baseline(rows, LABELS, rendered(project), "big-model", fake_model(calls),
                           B.ResponseCache(cache_path), 2.0, 10.0)
    assert len(calls) == 30 and [r["id"] for r in first] == [r["id"] for r in rows]
    assert first[1]["cost_usd"] == pytest.approx((1000 * 2.0 + 5 * 10.0) / 1e6)
    second = B.run_baseline(rows, LABELS, rendered(project), "big-model", fake_model(calls),
                            B.ResponseCache(cache_path), 2.0, 10.0)                 # new process: cache from disk
    assert len(calls) == 30 and all(r["cached"] for r in second)
    assert [r["label"] for r in second] == [r["label"] for r in first]
    assert B.summarise(second)["spent_usd_this_run"] == 0
    B.run_baseline(rows[:3], LABELS, rendered(project), "other-model", fake_model(calls),
                   B.ResponseCache(cache_path))                                     # another model: new key
    assert len(calls) == 33


def test_unparseable_answers_count_as_wrong_and_are_reported(project, tmp_path):
    rows = make_rows("validation", 30)
    records = B.run_baseline(rows, LABELS, rendered(project), "m", fake_model([]),
                             B.ResponseCache(str(tmp_path / "c.jsonl")), 2.0, 10.0)
    stats = B.summarise(records, rows)
    assert stats["unparseable"] == 3 and stats["unparseable_rate"] == pytest.approx(0.1)
    assert stats["accuracy"] == pytest.approx(21 / 30)       # 3 unparseable + 6 wrong labels
    assert 0 < stats["macro_f1"] < 1
    assert stats["input_tokens"] == 30000 and stats["usd_per_ticket"] == pytest.approx(0.00205)


def test_failed_calls_are_retried_then_reported_and_not_cached(project, tmp_path):
    attempts = {"n": 0}

    def flaky(messages):
        attempts["n"] += 1
        if attempts["n"] <= 2:
            raise TimeoutError("busy")
        return {"text": "alpha", "input_tokens": 10, "output_tokens": 1}

    def broken(messages):
        raise RuntimeError("endpoint down")

    row = make_rows("validation", 1)
    cache = B.ResponseCache(str(tmp_path / "c.jsonl"))
    ok = B.run_baseline(row, LABELS, rendered(project), "m", flaky, cache, backoff_seconds=0)
    assert ok[0]["label"] == "alpha" and attempts["n"] == 3
    bad = B.run_baseline(row, LABELS, rendered(project), "m2", broken, cache, max_retries=1, backoff_seconds=0)
    assert bad[0]["unparseable"] and "endpoint down" in bad[0]["error"] and len(cache) == 1
    assert B.summarise(bad)["failed_calls"] == 1


def test_bundle_has_no_test_labels_and_the_repository_code(project):
    dest = B.write_bundle(project)
    test_rows = [json.loads(line) for line in open(os.path.join(dest, "test.jsonl"))]
    assert len(test_rows) == 21 and all(set(r) == {"id", "text"} for r in test_rows)
    assert "label" in json.loads(open(os.path.join(dest, "validation.jsonl")).readline())
    manifest = json.load(open(os.path.join(dest, "bundle_manifest.json")))
    assert manifest["prompt_hash"] == rendered(project)["hash"] and manifest["data_version"] == "data-v1"
    for name in ("baseline.py", "metrics.py", "__init__.py"):
        assert os.path.exists(os.path.join(dest, "src", "triage", name))
    B.write_bundle(project)                                   # rebuilding replaces the old bundle


def test_bundle_refuses_to_overwrite_an_unrelated_folder(project):
    os.makedirs(project["baseline"]["bundle_dir"])
    open(os.path.join(project["baseline"]["bundle_dir"], "notes.txt"), "w").close()
    with pytest.raises(FileExistsError):
        B.write_bundle(project)


def _remote_outputs(project, tmp_path, splits=("validation", "test"), **summary_overrides):
    """Result files as the notebook writes them."""
    out, prompt = tmp_path / "outputs", rendered(project)
    out.mkdir()
    cache = B.ResponseCache(str(out / "cache" / "responses.jsonl"))
    for split in splits:
        records = B.run_baseline(make_rows(split, 30 if split == "validation" else 21), LABELS, prompt, "big-model",
                                 fake_model([]), cache, 2.0, 10.0)
        with open(out / B.predictions_file(split), "w", encoding="utf-8") as handle:
            handle.writelines(json.dumps(r) + "\n" for r in records)
    summary = {"model": "big-model", "prompt_hash": prompt["hash"], "data_version": "data-v1", "complete": True,
               "usd_per_1m_input_tokens": 2.0, "usd_per_1m_output_tokens": 10.0, "mlflow_run_id": "dbx123",
               **summary_overrides}
    (out / B.SUMMARY_FILE).write_text(json.dumps(summary))
    return str(out)


def test_import_places_files_and_logs_a_baseline_run(project, tmp_path):
    source = _remote_outputs(project, tmp_path)
    with Tracker(project, "baseline", use_mlflow=False) as tracker:
        result = B.import_results(project, source, tracker)
    artifacts = project["paths"]["artifacts"]
    assert len(open(os.path.join(artifacts, "baseline_predictions.jsonl")).readlines()) == 21
    assert len(open(os.path.join(artifacts, "baseline_predictions_validation.jsonl")).readlines()) == 30
    assert result["splits"]["validation"]["accuracy"] == pytest.approx(21 / 30)
    assert "accuracy" not in result["splits"]["test"]         # test accuracy is left to `make eval`
    assert result["remote_run_id"] == "dbx123" and result["local_run_id"] == tracker.run_id
    events = [json.loads(line) for line in open(tracker._fallback_path)]
    assert {"baseline": "true"} in [e.get("tags") for e in events]


@pytest.mark.parametrize(("overrides", "message"), [
    ({"prompt_hash": "other"}, "different prompt"), ({"data_version": "data-v9"}, "data-v9"),
    ({"complete": False, "limit": 50}, "partial run"),
])
def test_import_rejects_results_that_do_not_match_this_checkout(project, tmp_path, overrides, message):
    with pytest.raises(ValueError, match=message):
        B.import_results(project, _remote_outputs(project, tmp_path, **overrides))


def test_import_rejects_incomplete_split(project, tmp_path):
    source = _remote_outputs(project, tmp_path)
    path = os.path.join(source, B.predictions_file("test"))
    lines = open(path).readlines()
    open(path, "w").writelines(lines[:-1])
    with pytest.raises(ValueError, match="exactly the ids"):
        B.import_results(project, source)
