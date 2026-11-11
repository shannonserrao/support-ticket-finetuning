"""Shared fixture: a tiny project directory with processed data, labels and a baseline prompt. No network."""
import json
import os
import shutil

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LABELS = ["alpha", "beta", "gamma"]


def make_rows(split: str, n: int) -> list[dict]:
    return [{"id": f"{split}-{i:05d}", "text": f"{LABELS[i % 3]} ticket number {i} in {split}", "label": LABELS[i % 3]}
            for i in range(n)]


def write_jsonl(path, rows) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def oracle_answer(text: str) -> str:
    """Stand-in for the expensive model: mostly right, sometimes wrong, sometimes not a label at all."""
    label, number = text.split()[0], int(text.split()[3])
    if number % 10 == 7:
        return "I am not sure which label applies here."
    if number % 5 == 0:
        return LABELS[(LABELS.index(label) + 1) % 3]
    return f" {label}.\n"


@pytest.fixture()
def project(tmp_path):
    data = tmp_path / "data"
    write_jsonl(data / "processed" / "validation.jsonl", make_rows("validation", 30))
    write_jsonl(data / "processed" / "test.jsonl", make_rows("test", 21))
    (data / "labels.json").write_text(json.dumps(LABELS))
    (data / "manifest.json").write_text(json.dumps({
        "data_version": "data-v1",
        "splits": {"validation": {"rows": 30, "sha256": "v"}, "test": {"rows": 21, "sha256": "t"}}}))
    prompt_dir = tmp_path / "configs" / "baseline"
    prompt_dir.mkdir(parents=True)
    shutil.copyfile(os.path.join(REPO, "configs", "baseline", "prompt.yaml"), prompt_dir / "prompt.yaml")
    (prompt_dir / "label_descriptions.yaml").write_text("".join(f"{lab}: tickets about {lab}\n" for lab in LABELS))
    return {
        "paths": {"data": str(data), "artifacts": str(tmp_path / "artifacts")},
        "baseline": {"model": None, "prompt_file": str(prompt_dir / "prompt.yaml"),
                     "label_descriptions_file": str(prompt_dir / "label_descriptions.yaml"),
                     "max_output_tokens": 20, "bundle_dir": str(tmp_path / "notebook" / "inputs")},
        "routing": {"select": "validation", "confidence_threshold": 0.8, "target_accuracy": None, "sweep_step": 0.01},
        "cost": {"gpu_usd_per_hour": 0.6, "small_tickets_per_second": None, "baseline_usd_per_ticket": None},
        "business": {"tickets_per_month": 100000, "minutes_lost_per_misroute": 6, "agent_usd_per_hour": 25},
        "predict": {"split": "test"},
    }
