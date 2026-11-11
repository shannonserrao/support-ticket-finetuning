"""Shared model helpers: device choice, length-sorted batched inference, temperature scaling.

Used by train.py (validation, calibration) and predict.py, so both compute logits the same way.
"""
from __future__ import annotations

import math

import torch


def pick_device(name: str = "auto") -> torch.device:
    """`auto` selects CUDA, then MPS, then CPU (spec 02 section 3.7)."""
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def device_name(device: torch.device) -> str:
    return torch.cuda.get_device_name(device) if device.type == "cuda" else device.type


def precision_name(device: torch.device) -> str:
    return "fp16-autocast" if device.type == "cuda" else "fp32"


def autocast(device: torch.device):
    """Mixed precision on CUDA only."""
    return torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda")


def length_sorted_batches(lengths: list[int], batch_size: int) -> list[list[int]]:
    """Indices grouped into batches of similar length, to minimise padding."""
    order = sorted(range(len(lengths)), key=lambda i: (lengths[i], i))
    return [order[i:i + batch_size] for i in range(0, len(order), batch_size)]


def pad_batch(tokenizer, encoded: dict, indices: list[int], device: torch.device) -> dict:
    batch = tokenizer.pad({key: [values[i] for i in indices] for key, values in encoded.items()}, return_tensors="pt")
    return {key: value.to(device) for key, value in batch.items()}


@torch.no_grad()
def compute_logits(model, tokenizer, texts: list[str], max_length: int, batch_size: int,
                   device: torch.device) -> torch.Tensor:
    """Logits for `texts` in the original order, as float32 on CPU. The model is left in eval mode."""
    model.eval()
    encoded = dict(tokenizer(texts, truncation=True, max_length=max_length))
    out: torch.Tensor | None = None
    for indices in length_sorted_batches([len(ids) for ids in encoded["input_ids"]], batch_size):
        with autocast(device):
            logits = model(**pad_batch(tokenizer, encoded, indices, device)).logits
        logits = logits.float().cpu()
        if out is None:
            out = torch.empty(len(texts), logits.shape[-1])
        out[torch.tensor(indices)] = logits
    if device.type == "cuda":
        torch.cuda.synchronize()
    return out if out is not None else torch.empty(0, 0)


def fit_temperature(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """One temperature minimising negative log-likelihood (spec 02 section 5)."""
    log_t = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(logits / log_t.exp(), targets)
        loss.backward()
        return loss

    optimizer.step(closure)
    temperature = log_t.detach().exp().item()
    return temperature if math.isfinite(temperature) and temperature > 0 else 1.0


def confidences(logits: torch.Tensor, temperature: float = 1.0) -> tuple[list[int], list[float]]:
    """Predicted class index and max softmax(logits / temperature) per row."""
    probs = torch.softmax(logits / temperature, dim=-1)
    conf, pred = probs.max(dim=-1)
    return pred.tolist(), conf.tolist()
