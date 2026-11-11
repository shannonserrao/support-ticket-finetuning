import copy
import json
import os

import pytest

from triage import data
from triage.data import DataCheckError

LABELS = ["alpha", "beta", "gamma"]


def make_source():
    train = []
    for label in LABELS:
        for i in range(20):
            train.append({"text": f"{label} question number {i}", "label": label})
    train.append({"text": "  alpha question number 0  ", "label": "alpha"})   # duplicate after strip
    train.append({"text": "   ", "label": "beta"})                            # empty after strip
    train.append({"text": "leaked sentence", "label": "gamma"})               # also in test
    test = [{"text": f"{label} test sentence {i}", "label": label} for label in LABELS for i in range(5)]
    test.append({"text": "leaked sentence", "label": "gamma"})
    return {"revision": "abc123", "labels": LABELS, "splits": {"train": train, "test": test}}


def make_cfg(tmp_path, **overrides):
    cfg = {"paths": {"data": str(tmp_path / "data")},
           "data": {"dataset_id": "x/y", "validation_fraction": 0.1, "seed": 13,
                    "max_train_samples": None, "max_eval_samples": None, "allow_label_change": False}}
    cfg["data"].update(overrides)
    return cfg


def read_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def test_normalise_dedupe_and_leak_removal(tmp_path):
    manifest = data.build_from_source(make_cfg(tmp_path), make_source())
    root = tmp_path / "data" / "processed"
    train, val, test = (read_jsonl(root / f"{n}.jsonl") for n in ("train", "validation", "test"))
    texts = [r["text"] for r in train + val]
    assert len(texts) == len(set(texts)) == 60              # 60 unique; duplicate, empty and leaked rows gone
    assert "leaked sentence" not in texts
    assert "leaked sentence" in {r["text"] for r in test}   # test set is not altered
    assert all(r["text"] == r["text"].strip() for r in train + val + test)
    assert manifest["splits"]["train"]["rows"] == len(train)


def test_split_is_stratified_and_covers_every_label(tmp_path):
    data.build_from_source(make_cfg(tmp_path), make_source())
    val = read_jsonl(tmp_path / "data" / "processed" / "validation.jsonl")
    counts = {lab: sum(r["label"] == lab for r in val) for lab in LABELS}
    assert counts == {"alpha": 2, "beta": 2, "gamma": 2}    # 10% of 20 per label


def test_build_is_deterministic_and_version_stable(tmp_path):
    first = data.build_from_source(make_cfg(tmp_path), make_source())
    second = data.build_from_source(make_cfg(tmp_path), make_source())
    assert first["splits"] == second["splits"]
    assert first["labels_sha256"] == second["labels_sha256"]
    assert second["data_version"] == "data-v1"
    assert second["created"] == first["created"]            # unchanged data keeps its manifest


def test_version_bumps_when_a_hash_changes(tmp_path):
    data.build_from_source(make_cfg(tmp_path), make_source())
    changed = make_source()
    changed["splits"]["train"].append({"text": "alpha brand new text", "label": "alpha"})
    manifest = data.build_from_source(make_cfg(tmp_path), changed)
    assert manifest["data_version"] == "data-v2"


def test_different_seed_changes_the_split(tmp_path):
    a = data.build_from_source(make_cfg(tmp_path, seed=1), make_source())
    b = data.build_from_source(make_cfg(tmp_path, seed=2), make_source())
    assert a["splits"]["validation"]["sha256"] != b["splits"]["validation"]["sha256"]


def test_leakage_check_fails_on_injected_duplicate():
    splits = {"train": [{"id": "train-1", "text": "same", "label": "alpha"}],
              "validation": [{"id": "validation-1", "text": "same", "label": "alpha"}],
              "test": [{"id": "test-1", "text": "other", "label": "alpha"}]}
    with pytest.raises(DataCheckError, match="texts shared between train and validation"):
        data.check_splits(splits, ["alpha"])


def test_leakage_check_fails_on_shared_id():
    splits = {"train": [{"id": "a", "text": "one", "label": "alpha"}],
              "test": [{"id": "a", "text": "two", "label": "alpha"}]}
    with pytest.raises(DataCheckError, match="ids shared"):
        data.check_splits(splits, ["alpha"])


def test_coverage_check_fails_when_a_label_is_missing():
    splits = {"train": [{"id": "1", "text": "a", "label": "alpha"}],
              "test": [{"id": "2", "text": "b", "label": "alpha"}]}
    with pytest.raises(DataCheckError, match="missing 1 labels"):
        data.check_splits(splits, ["alpha", "beta"])
    data.check_splits(splits, ["alpha", "beta"], require_coverage=False)   # smoke subsets skip coverage


def test_label_change_needs_the_flag(tmp_path):
    data.build_from_source(make_cfg(tmp_path), make_source())
    renamed = copy.deepcopy(make_source())
    renamed["labels"] = ["alpha", "beta", "delta"]
    for split in renamed["splits"].values():
        for row in split:
            row["label"] = "delta" if row["label"] == "gamma" else row["label"]
    with pytest.raises(DataCheckError, match="allow_label_change"):
        data.build_from_source(make_cfg(tmp_path), renamed)
    data.build_from_source(make_cfg(tmp_path, allow_label_change=True), renamed)


def test_build_fails_loudly_when_the_published_data_leaks(tmp_path, monkeypatch):
    """If the cleaning step were skipped, the check still stops the command."""
    monkeypatch.setattr(data, "remove_overlap", lambda rows, reference: rows)
    with pytest.raises(DataCheckError, match="shared between"):
        data.build_from_source(make_cfg(tmp_path), make_source())


def test_smoke_subsample_limits_rows_and_skips_coverage(tmp_path):
    cfg = make_cfg(tmp_path, max_train_samples=5, max_eval_samples=3)
    manifest = data.build_from_source(cfg, make_source())
    assert manifest["splits"]["train"]["rows"] == 5
    assert manifest["splits"]["validation"]["rows"] <= 3
    assert manifest["splits"]["test"]["rows"] == 3


def test_length_stats_and_manifest_files(tmp_path):
    manifest = data.build_from_source(make_cfg(tmp_path), make_source())
    assert manifest["stats"]["train"]["words_max"] >= manifest["stats"]["train"]["words_mean"] > 0
    assert os.path.exists(tmp_path / "data" / "labels.json")
    assert json.loads((tmp_path / "data" / "labels.json").read_text()) == LABELS
    saved = json.loads((tmp_path / "data" / "manifest.json").read_text())
    assert saved["source"] == {"dataset_id": "x/y", "revision": "abc123"}
    assert saved["corrections_applied"] == 0
    assert "stats" not in saved


# --------------------------------------------------------------------------- T-13 corrections merge (spec 05 HF-5)

def correction(item_id, text, verdict, prediction="alpha", corrected=""):
    return {"id": item_id, "input_ref": text, "prediction": prediction, "verdict": verdict,
            "corrected": corrected, "reviewer": "r1"}


def cfg_with_corrections(tmp_path, items):
    path = tmp_path / "feedback" / "corrections.jsonl"
    path.parent.mkdir(exist_ok=True)
    path.write_text("".join(json.dumps(i) + "\n" for i in items), encoding="utf-8")
    cfg = make_cfg(tmp_path)
    cfg["feedback"] = {"corrections": str(path)}
    return cfg


def test_correction_changes_manifest_hash_and_bumps_version(tmp_path):
    before = data.build_from_source(make_cfg(tmp_path), make_source())
    cfg = cfg_with_corrections(tmp_path, [correction("live-1", "a brand new live ticket", "incorrect",
                                                     prediction="alpha", corrected="beta")])
    after = data.build_from_source(cfg, make_source())
    assert after["splits"]["train"]["sha256"] != before["splits"]["train"]["sha256"]
    assert after["splits"]["train"]["rows"] == before["splits"]["train"]["rows"] + 1
    assert after["splits"]["test"] == before["splits"]["test"]              # test split untouched
    assert after["splits"]["validation"] == before["splits"]["validation"]
    assert after["data_version"] == "data-v2" and after["corrections_applied"] == 1
    train = read_jsonl(tmp_path / "data" / "processed" / "train.jsonl")
    assert {"id": "feedback-live-1", "text": "a brand new live ticket", "label": "beta"} in train


def test_corrected_test_items_never_enter_training(tmp_path):
    before = data.build_from_source(make_cfg(tmp_path), make_source())
    items = [correction("test-00000", "alpha test sentence 0", "incorrect", corrected="beta"),   # by id
             correction("live-9", "gamma test sentence 2", "correct", prediction="gamma")]       # by text
    after = data.build_from_source(cfg_with_corrections(tmp_path, items), make_source())
    assert after["splits"] == before["splits"]                               # nothing changed anywhere
    assert after["data_version"] == "data-v1" and after["corrections_applied"] == 0
    assert after["corrections"]["excluded_test"] == 2
    assert after["corrections"]["excluded_test_ids"] == ["test-00000", "live-9"]
    train_texts = {r["text"] for r in read_jsonl(tmp_path / "data" / "processed" / "train.jsonl")}
    assert not train_texts & {"alpha test sentence 0", "gamma test sentence 2"}


def test_merge_corrections_cases():
    train = [{"id": "train-1", "text": "one", "label": "alpha"}, {"id": "train-2", "text": "two", "label": "beta"}]
    val = [{"id": "validation-1", "text": "three", "label": "alpha"}]
    test = [{"id": "test-1", "text": "four", "label": "alpha"}]
    items = [
        correction("train-1", "one", "incorrect", corrected="gamma"),        # relabel existing train row
        correction("train-2", "two", "correct", prediction="beta"),          # confirmed, already right
        correction("validation-1", "three", "incorrect", corrected="beta"),  # validation is left alone
        correction("live-1", "five", "correct", prediction="alpha"),         # confirmed live ticket is added
        correction("live-2", "  six ", "incorrect", corrected="NEW:loans"),  # label proposal, not merged
        correction("live-3", "seven", "unsure"),
        correction("live-4", "five", "incorrect", corrected="beta"),         # same text again: relabels, no duplicate
    ]
    merged, report = data.merge_corrections(train, val, test, items, LABELS)
    assert merged == [{"id": "train-1", "text": "one", "label": "gamma"},
                      {"id": "train-2", "text": "two", "label": "beta"},
                      {"id": "feedback-live-1", "text": "five", "label": "beta"}]
    assert train[0]["label"] == "alpha"                                      # input rows are not mutated
    assert {k: report[k] for k in ("added", "relabelled", "unchanged", "skipped_validation",
                                   "skipped_unknown_label", "skipped_unsure", "excluded_test", "applied")} == {
        "added": 1, "relabelled": 2, "unchanged": 1, "skipped_validation": 1,
        "skipped_unknown_label": 1, "skipped_unsure": 1, "excluded_test": 0, "applied": 3}


def test_merged_dataset_still_passes_leakage_check(tmp_path):
    items = [correction("live-1", "alpha question number 3", "incorrect", corrected="beta")]   # existing train/val text
    manifest = data.build_from_source(cfg_with_corrections(tmp_path, items), make_source())
    root = tmp_path / "data" / "processed"
    texts = [r["text"] for n in ("train", "validation") for r in read_jsonl(root / f"{n}.jsonl")]
    assert len(texts) == len(set(texts))
    assert manifest["corrections"]["relabelled"] + manifest["corrections"]["skipped_validation"] == 1
