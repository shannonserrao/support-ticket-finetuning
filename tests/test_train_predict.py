"""Train and predict on a tiny random model, on CPU, without network access (spec 02 §7, spec 06 §6)."""
import json
import os

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from triage import modeling  # noqa: E402
from triage.evaluate import evaluate_files  # noqa: E402
from triage.feedback import read_jsonl  # noqa: E402
from triage.predict import predict, route  # noqa: E402
from triage.tracking import Tracker  # noqa: E402
from triage.train import train  # noqa: E402

LABELS = ["alpha", "beta", "gamma"]
WORDS = ["card", "transfer", "refund", "please", "help", "my", "is", "late", "broken", "now"]


def _rows(split, n):
    rows = []
    for i in range(n):
        label = LABELS[i % 3]
        extra = " ".join(WORDS[(i + k) % len(WORDS)] for k in range(1 + i % 5))
        rows.append({"id": f"{split}-{i:05d}", "text": f"{label} {extra} {split}{i}", "label": label})
    return rows


def _write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


@pytest.fixture()
def cfg(tmp_path):
    from tokenizers import Tokenizer, models, pre_tokenizers

    base = tmp_path / "tiny-base"
    vocab = {tok: i for i, tok in enumerate(["[PAD]", "[UNK]", "[CLS]", "[SEP]", *LABELS, *WORDS])}
    tok = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    transformers.PreTrainedTokenizerFast(tokenizer_object=tok, pad_token="[PAD]", unk_token="[UNK]",
                                         cls_token="[CLS]", sep_token="[SEP]").save_pretrained(base)
    transformers.BertModel(transformers.BertConfig(
        vocab_size=len(vocab), hidden_size=32, num_hidden_layers=1, num_attention_heads=2,
        intermediate_size=64, max_position_embeddings=64)).save_pretrained(base)

    data = tmp_path / "data"
    _write_jsonl(data / "processed" / "train.jsonl", _rows("train", 48))
    _write_jsonl(data / "processed" / "validation.jsonl", _rows("validation", 12))
    _write_jsonl(data / "processed" / "test.jsonl", _rows("test", 15))
    (data / "labels.json").write_text(json.dumps(LABELS))
    (data / "manifest.json").write_text(json.dumps({"data_version": "data-v1"}))
    return {
        "paths": {"data": str(data), "artifacts": str(tmp_path / "artifacts")},
        "model": {"base_id": str(base), "revision": None, "max_length": 16},
        "train": {"seed": 13, "device": "cpu", "epochs": 2, "batch_size": 8, "lr": 1e-3, "weight_decay": 0.01,
                  "warmup_ratio": 0.1, "eval_every_steps": 4, "log_every_steps": 2, "save_every_steps": 4,
                  "early_stopping_patience": 5, "resume": True},
        "predict": {"batch_size": 64, "split": "test", "latency_warmup_calls": 2, "latency_calls": 5},
        "routing": {"select": "fixed", "confidence_threshold": 0.5},
        "feedback": {"predictions_log": str(tmp_path / "feedback" / "predictions_log.jsonl")},
    }


def _metrics(tracker):
    """All metrics a fallback (JSONL) tracker run logged, merged."""
    merged = {}
    for event in read_jsonl(tracker._fallback_path):
        if event["event"] == "metrics":
            merged.update(event["metrics"])
    return merged


def _train(cfg):
    with Tracker(cfg, "test-train", use_mlflow=False) as tracker:
        model_dir = train(cfg, tracker)
    return model_dir, tracker


def test_train_writes_model_calibration_and_meta(cfg):
    model_dir, tracker = _train(cfg)
    for name in ("config.json", "model.safetensors", "tokenizer.json", "labels.json", "calibration.json",
                 "training_meta.json"):
        assert os.path.exists(os.path.join(model_dir, name)), name
    calibration = json.load(open(os.path.join(model_dir, "calibration.json")))
    assert calibration["temperature"] > 0
    meta = json.load(open(os.path.join(model_dir, "training_meta.json")))
    assert meta["data_version"] == "data-v1" and meta["steps"] == 12 and meta["resumed_from_step"] is None
    assert meta["model_version"].startswith("tiny-base-data-v1-")
    logged = _metrics(tracker)
    for key in ("train_loss", "lr", "val_accuracy", "val_macro_f1", "train_seconds",
                "val_ece_before_scaling", "val_ece_after_scaling"):       # T-07: both calibration errors logged
        assert key in logged, key
    assert logged["val_nll_after_scaling"] <= logged["val_nll_before_scaling"] + 1e-6
    assert not os.path.exists(os.path.join(cfg["paths"]["artifacts"], "checkpoints", "last.pt"))


def test_interrupted_training_resumes_from_checkpoint(cfg):
    class Interrupt(Tracker):
        def log_metrics(self, metrics, step=None):
            super().log_metrics(metrics, step)
            if step == 6:
                raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        with Interrupt(cfg, "interrupted", use_mlflow=False) as tracker:
            train(cfg, tracker)
    assert os.path.exists(os.path.join(cfg["paths"]["artifacts"], "checkpoints", "last.pt"))
    model_dir, _ = _train(cfg)
    meta = json.load(open(os.path.join(model_dir, "training_meta.json")))
    assert meta["resumed_from_step"] == 4 and meta["steps"] == 12


def test_checkpoint_from_other_config_is_ignored(cfg):
    class Interrupt(Tracker):
        def log_metrics(self, metrics, step=None):
            super().log_metrics(metrics, step)
            if step == 6:
                raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        with Interrupt(cfg, "interrupted", use_mlflow=False) as tracker:
            train(cfg, tracker)
    cfg["train"]["lr"] = 5e-4
    model_dir, _ = _train(cfg)
    assert json.load(open(os.path.join(model_dir, "training_meta.json")))["resumed_from_step"] is None


def test_predict_writes_predictions_log_and_report(cfg):
    _train(cfg)
    with Tracker(cfg, "test-predict", use_mlflow=False) as tracker:
        out = predict(cfg, tracker)
    preds = read_jsonl(out)
    gold = read_jsonl(os.path.join(cfg["paths"]["data"], "processed", "test.jsonl"))
    assert [p["id"] for p in preds] == [g["id"] for g in gold]            # original order restored
    for p in preds:
        assert p["label"] in LABELS and 0 < p["confidence"] <= 1
        assert p["route"] == route(p["confidence"], 0.5)
        assert p["model_version"].startswith("tiny-base-data-v1-")
    assert sum(p["latency_ms"] is not None for p in preds) == 5
    logged = _metrics(tracker)
    assert logged["throughput_tickets_per_s"] > 0 and logged["latency_p95_ms"] >= logged["latency_p50_ms"] > 0

    log_path = cfg["feedback"]["predictions_log"]
    assert len(read_jsonl(log_path)) == 15
    assert read_jsonl(log_path)[0]["input_ref"] == gold[0]["text"]
    with Tracker(cfg, "test-predict-again", use_mlflow=False) as tracker:
        predict(cfg, tracker)
    assert len(read_jsonl(log_path)) == 15                                # rerun does not duplicate the log

    report = evaluate_files(cfg)
    assert report["meta"]["n"] == 15 and "latency_p95_ms" in report["metrics"]


def test_predictions_identical_for_batch_sizes_1_and_64(cfg):
    _train(cfg)
    outputs = []
    for batch_size in (1, 64):
        cfg["predict"]["batch_size"] = batch_size
        with Tracker(cfg, f"bs{batch_size}", use_mlflow=False) as tracker:
            outputs.append(read_jsonl(predict(cfg, tracker)))
    assert [p["label"] for p in outputs[0]] == [p["label"] for p in outputs[1]]
    assert [p["confidence"] for p in outputs[0]] == pytest.approx([p["confidence"] for p in outputs[1]], abs=1e-5)


def test_route_is_a_pure_function_of_confidence_and_threshold():
    assert route(0.80, 0.80) == "small"
    assert route(0.7999, 0.80) == "escalate"
    assert route(1.0, 0.5) == "small" and route(0.0, 0.5) == "escalate"


def test_temperature_scaling_fixes_overconfident_logits():
    generator = torch.Generator().manual_seed(0)
    base = torch.randn(2000, 5, generator=generator) * 2
    targets = torch.distributions.Categorical(logits=base).sample()      # `base` is perfectly calibrated
    temperature = modeling.fit_temperature(base * 4, targets)             # 4x overconfident
    assert temperature == pytest.approx(4.0, rel=0.15)


def test_length_sorted_batches_cover_every_index_once():
    batches = modeling.length_sorted_batches([5, 1, 9, 3, 7], 2)
    assert batches == [[1, 3], [0, 4], [2]]


def test_micro_batches_keep_the_step_count(cfg):
    cfg["train"]["micro_batch_size"] = 3                                  # 8 = 3 + 3 + 2 per optimiser step
    model_dir, tracker = _train(cfg)
    meta = json.load(open(os.path.join(model_dir, "training_meta.json")))
    assert meta["steps"] == 12
    assert _metrics(tracker)["train_loss"] > 0
