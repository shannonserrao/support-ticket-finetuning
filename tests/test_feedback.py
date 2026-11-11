import csv

import pytest

from triage import feedback


def _log(tmp_path):
    log = tmp_path / "log.jsonl"
    for i, conf in enumerate([0.2, 0.95, 0.99, 0.4]):
        feedback.log_prediction(str(log), {
            "id": f"x{i}", "input_ref": f"ref{i}", "prediction": "A",
            "confidence": conf, "model_version": "model-0.1.0",
        })
    return log


def test_log_requires_keys(tmp_path):
    with pytest.raises(ValueError):
        feedback.log_prediction(str(tmp_path / "l.jsonl"), {"id": "1"})


def test_export_and_ingest_round_trip(tmp_path):
    log = _log(tmp_path)
    queue, corrections = tmp_path / "queue.csv", tmp_path / "corrections.jsonl"
    n = feedback.export_review_queue(str(log), str(queue), str(corrections), 0.5, 0.0)
    assert n == 2  # only the two low-confidence items

    with open(queue, encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows[0].update(verdict="correct", reviewer="ana")
    rows[1].update(verdict="incorrect", corrected="B", reviewer="ana")
    reviewed = tmp_path / "reviewed.csv"
    with open(reviewed, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=feedback.QUEUE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    counts = feedback.ingest_reviewed(str(reviewed), str(corrections))
    assert counts["added"] == 2
    assert feedback.ingest_reviewed(str(reviewed), str(corrections))["skipped_duplicate"] == 2
    assert feedback.review_summary(str(corrections))["agreement_all"] == 0.5
    # reviewed items are not queued again
    assert feedback.export_review_queue(str(log), str(queue), str(corrections), 0.5, 0.0) == 0


def test_incorrect_needs_correction(tmp_path):
    reviewed = tmp_path / "r.csv"
    with open(reviewed, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=feedback.QUEUE_FIELDS)
        writer.writeheader()
        writer.writerow({"id": "1", "input_ref": "r", "prediction": "A", "verdict": "incorrect",
                         "reviewer": "ana"})
    with pytest.raises(ValueError):
        feedback.ingest_reviewed(str(reviewed), str(tmp_path / "c.jsonl"))
