"""Fine-tune the encoder, calibrate, save to artifacts/model/. Returns the model directory.

Implements specs/02-training.md (T-06 training loop, T-07 temperature scaling).
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import random
import time

from . import metrics as M
from .feedback import read_jsonl
from .tracking import git_info

log = logging.getLogger("triage.train")


def _set_seeds(seed: int) -> None:
    import torch

    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover - numpy always ships with torch
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _fingerprint(cfg: dict, data_version: str, n_train: int) -> str:
    """Identifies a run for resuming: a checkpoint is reused only for the same data, model and train config."""
    payload = json.dumps([cfg["model"], cfg["train"], data_version, n_train], sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def _save_json(path: str, obj) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, default=str)


def train(cfg: dict, tracker) -> str:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    from . import modeling

    paths, mcfg, tcfg = cfg["paths"], cfg["model"], cfg["train"]
    data_root, artifacts = paths["data"], paths["artifacts"]
    model_dir = os.path.join(artifacts, "model")
    ckpt_path = os.path.join(artifacts, "checkpoints", "last.pt")

    train_rows = read_jsonl(os.path.join(data_root, "processed", "train.jsonl"))
    val_rows = read_jsonl(os.path.join(data_root, "processed", "validation.jsonl"))
    if not train_rows or not val_rows:
        raise FileNotFoundError("run `make data` before `make train`")
    with open(os.path.join(data_root, "labels.json"), encoding="utf-8") as handle:
        labels = json.load(handle)
    with open(os.path.join(data_root, "manifest.json"), encoding="utf-8") as handle:
        data_version = json.load(handle)["data_version"]
    label2id = {name: i for i, name in enumerate(labels)}

    seed = tcfg["seed"]
    _set_seeds(seed)
    device = modeling.pick_device(tcfg.get("device", "auto"))
    amp = device.type == "cuda"
    log.info("training on %s (%s)", modeling.device_name(device), modeling.precision_name(device))

    tokenizer = AutoTokenizer.from_pretrained(mcfg["base_id"], revision=mcfg.get("revision"))
    model = AutoModelForSequenceClassification.from_pretrained(
        mcfg["base_id"], revision=mcfg.get("revision"), num_labels=len(labels),
        id2label=dict(enumerate(labels)), label2id=label2id,
    ).to(device)

    encoded = dict(tokenizer([r["text"] for r in train_rows], truncation=True, max_length=mcfg["max_length"]))
    targets = torch.tensor([label2id[r["label"]] for r in train_rows])
    val_texts, val_gold = [r["text"] for r in val_rows], [r["label"] for r in val_rows]

    n, batch_size = len(train_rows), tcfg["batch_size"]
    micro = tcfg.get("micro_batch_size") or batch_size
    steps_per_epoch = math.ceil(n / batch_size)
    total_steps = steps_per_epoch * tcfg["epochs"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])
    scheduler = get_linear_schedule_with_warmup(optimizer, int(tcfg["warmup_ratio"] * total_steps), total_steps)
    scaler = torch.amp.GradScaler(device.type, enabled=amp)

    step, best_f1, best_step, bad_evals, resumed_from = 0, -1.0, 0, 0, None
    fingerprint = _fingerprint(cfg, data_version, n)
    if tcfg.get("resume", True) and os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        if state["fingerprint"] == fingerprint:
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            scaler.load_state_dict(state["scaler"])
            step, best_f1, best_step, bad_evals = state["step"], state["best_f1"], state["best_step"], state["bad"]
            resumed_from = step
            log.info("resumed from checkpoint at step %d (best macro F1 so far %.4f)", step, best_f1)
        else:
            log.info("ignoring checkpoint from a different configuration")

    def save_checkpoint() -> None:
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(), "step": step,
                    "best_f1": best_f1, "best_step": best_step, "bad": bad_evals, "fingerprint": fingerprint},
                   ckpt_path + ".tmp")
        os.replace(ckpt_path + ".tmp", ckpt_path)

    def validate() -> dict:
        logits = modeling.compute_logits(model, tokenizer, val_texts, mcfg["max_length"],
                                         cfg["predict"]["batch_size"], device)
        pred = [labels[i] for i in logits.argmax(dim=-1).tolist()]
        model.train()
        if device.type == "cuda":
            torch.cuda.empty_cache()  # release evaluation-sized blocks so training batches do not fragment memory
        return {"val_accuracy": M.accuracy(val_gold, pred), "val_macro_f1": M.macro_f1(val_gold, pred)}

    def evaluate_and_keep_best() -> None:
        nonlocal best_f1, best_step, bad_evals
        scores = validate()
        tracker.log_metrics(scores, step=step)
        log.info("step %d/%d  val_accuracy %.4f  val_macro_f1 %.4f", step, total_steps,
                 scores["val_accuracy"], scores["val_macro_f1"])
        if scores["val_macro_f1"] > best_f1:
            best_f1, best_step, bad_evals = scores["val_macro_f1"], step, 0
            model.save_pretrained(model_dir)
        else:
            bad_evals += 1

    os.makedirs(model_dir, exist_ok=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    started = time.time()
    model.train()
    running, running_n, last_eval, stopped_early = 0.0, 0, step, False
    while step < total_steps and not stopped_early:
        epoch, offset = divmod(step, steps_per_epoch)
        # The order of each epoch depends only on the seed and the epoch, so a resumed run sees the same batches.
        order = torch.randperm(n, generator=torch.Generator().manual_seed(seed * 1000 + epoch)).tolist()
        for b in range(offset, steps_per_epoch):
            indices = order[b * batch_size:(b + 1) * batch_size]
            # One optimiser step per batch. With train.micro_batch_size the batch is processed in chunks and the
            # gradients are accumulated, which gives the same update with less GPU memory.
            optimizer.zero_grad(set_to_none=True)
            loss_value = 0.0
            for start in range(0, len(indices), micro):
                chunk = indices[start:start + micro]
                batch = modeling.pad_batch(tokenizer, encoded, chunk, device)
                with modeling.autocast(device):
                    loss = model(**batch, labels=targets[chunk].to(device)).loss * (len(chunk) / len(indices))
                scaler.scale(loss).backward()
                loss_value += loss.item()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            step += 1
            running, running_n = running + loss_value, running_n + 1
            if step % tcfg["log_every_steps"] == 0:
                tracker.log_metrics({"train_loss": running / running_n, "lr": scheduler.get_last_lr()[0]}, step=step)
                running, running_n = 0.0, 0
            if step % tcfg["eval_every_steps"] == 0:
                evaluate_and_keep_best()
                last_eval = step
                if bad_evals >= tcfg["early_stopping_patience"]:
                    log.info("early stop at step %d: no improvement in %d evaluations", step, bad_evals)
                    stopped_early = True
            if step % tcfg["save_every_steps"] == 0:
                save_checkpoint()
            if stopped_early:
                break
    if last_eval != step or best_f1 < 0:
        evaluate_and_keep_best()  # short runs (smoke) may never reach eval_every_steps
    seconds = time.time() - started
    final = {"train_seconds": seconds, "best_val_macro_f1": best_f1, "best_step": best_step, "steps": step}
    if device.type == "cuda":
        final["peak_gpu_memory_mb"] = torch.cuda.max_memory_allocated() / 2**20
    tracker.log_metrics(final)

    # Calibration on validation logits from the best checkpoint (T-07). Test data is never read here.
    best = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.load_state_dict(best.state_dict())
    del best
    val_logits = modeling.compute_logits(model, tokenizer, val_texts, mcfg["max_length"],
                                         cfg["predict"]["batch_size"], device)
    val_targets = torch.tensor([label2id[g] for g in val_gold])
    temperature = modeling.fit_temperature(val_logits, val_targets)
    calibration = {"temperature": temperature}
    for name, temp in (("before", 1.0), ("after", temperature)):
        pred, conf = modeling.confidences(val_logits, temp)
        correct = [p == t for p, t in zip(pred, val_targets.tolist())]
        calibration[f"val_ece_{name}_scaling"] = M.expected_calibration_error(correct, conf)
        nll = torch.nn.functional.cross_entropy(val_logits / temp, val_targets)
        calibration[f"val_nll_{name}_scaling"] = nll.item()
    tracker.log_metrics(calibration)
    log.info("temperature %.4f  validation ECE %.4f -> %.4f", temperature,
             calibration["val_ece_before_scaling"], calibration["val_ece_after_scaling"])

    tokenizer.save_pretrained(model_dir)
    _save_json(os.path.join(model_dir, "labels.json"), labels)
    _save_json(os.path.join(model_dir, "calibration.json"), calibration)
    git = git_info()
    model_version = f"{os.path.basename(str(mcfg['base_id']).rstrip('/'))}-{data_version}-{tracker.run_id[:8]}"
    meta = {"model_version": model_version, "run_id": tracker.run_id, "git_sha": git["git.sha"],
            "git_dirty": git["git.dirty"], "data_version": data_version, "device": modeling.device_name(device),
            "precision": modeling.precision_name(device), "resumed_from_step": resumed_from,
            "stopped_early": stopped_early, **final, "config": cfg}
    _save_json(os.path.join(model_dir, "training_meta.json"), meta)
    tracker.log_params({"data_version": data_version, "model_version": model_version,
                        "device": modeling.device_name(device), "precision": modeling.precision_name(device)})
    for name in ("calibration.json", "training_meta.json"):
        tracker.log_artifact(os.path.join(model_dir, name))
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)  # finished: a later run with the same config must start fresh
    return model_dir
