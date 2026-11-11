"""Zero-shot baseline with the expensive model (specs/03-evaluation.md section 4, task T-05).

Standard library only (PyYAML is imported lazily for the config files), because this file is also copied
into the Databricks bundle and imported by the notebook there. The model call itself is injected as
`call_model(messages) -> {"text", "input_tokens", "output_tokens"}`, so nothing here depends on a provider.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from . import metrics as M

UNPARSEABLE = "__unparseable__"
BUNDLE_MANIFEST = "bundle_manifest.json"
SUMMARY_FILE = "baseline_summary.json"


def predictions_file(split: str) -> str:
    return f"baseline_predictions_{split}.jsonl"


# --------------------------------------------------------------------------- prompt

def load_prompt(cfg: dict) -> dict:
    """Read the prompt template and the label descriptions named in the `baseline` config section."""
    import yaml

    bcfg = cfg["baseline"]
    with open(bcfg["prompt_file"], encoding="utf-8") as handle:
        prompt = yaml.safe_load(handle)
    with open(bcfg["label_descriptions_file"], encoding="utf-8") as handle:
        prompt["descriptions"] = yaml.safe_load(handle)
    return prompt


def render_prompt(prompt: dict, labels: list[str]) -> dict:
    """Fill the label list into the system prompt. The hash identifies the frozen prompt."""
    descriptions = prompt["descriptions"]
    missing, extra = set(labels) - set(descriptions), set(descriptions) - set(labels)
    if missing or extra:
        raise ValueError(f"label descriptions do not match labels.json: missing {sorted(missing)[:5]}, "
                         f"unknown {sorted(extra)[:5]}")
    block = "\n".join(f"- {label}: {' '.join(str(descriptions[label]).split())}" for label in labels)
    system = prompt["system"].strip().replace("{label_block}", block)
    user_template = prompt["user"].strip()
    digest = hashlib.sha256(json.dumps([system, user_template], ensure_ascii=False).encode("utf-8")).hexdigest()
    return {"system": system, "user_template": user_template, "hash": digest}


def build_messages(rendered: dict, text: str) -> list[dict]:
    return [{"role": "system", "content": rendered["system"]},
            {"role": "user", "content": rendered["user_template"].replace("{text}", text)}]


def parse_label(answer: str | None, labels: list[str]) -> str | None:
    """Map a model answer to a label, or None if it does not name exactly one label."""
    if not answer:
        return None
    cleaned = answer.strip().strip("`'\"*.,:; \n\t")
    cleaned = re.sub(r"^label\s*[:=]\s*", "", cleaned, flags=re.I).strip("`'\"*.,:; ")
    by_lower = {label.lower(): label for label in labels}
    if cleaned.lower() in by_lower:
        return by_lower[cleaned.lower()]
    # Otherwise accept the answer only if exactly one label occurs in it as a whole token.
    found = {label for label in labels
             if re.search(r"(?<![A-Za-z0-9_])" + re.escape(label.lower()) + r"(?![A-Za-z0-9_])", answer.lower())}
    return found.pop() if len(found) == 1 else None


# --------------------------------------------------------------------------- cache and cost

class ResponseCache:
    """Append-only JSONL cache keyed by model, prompt hash and message id, so reruns cost nothing."""

    def __init__(self, path: str):
        self.path = path
        self._lock = threading.Lock()
        self._items: dict[str, dict] = {}
        if os.path.exists(path):
            with open(path, encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        item = json.loads(line)
                        self._items[item["key"]] = item

    @staticmethod
    def key(model: str, prompt_hash: str, item_id: str) -> str:
        return f"{model}|{prompt_hash}|{item_id}"

    def get(self, key: str) -> dict | None:
        return self._items.get(key)

    def put(self, key: str, value: dict) -> None:
        item = {"key": key, **value}
        with self._lock:
            self._items[key] = item
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    def __len__(self) -> int:
        return len(self._items)


def call_cost(input_tokens: int, output_tokens: int, usd_per_1m_input: float | None,
              usd_per_1m_output: float | None) -> float | None:
    if usd_per_1m_input is None or usd_per_1m_output is None:
        return None
    return (input_tokens * usd_per_1m_input + output_tokens * usd_per_1m_output) / 1_000_000


# --------------------------------------------------------------------------- run

def run_baseline(rows: list[dict], labels: list[str], rendered: dict, model: str, call_model, cache: ResponseCache,
                 usd_per_1m_input: float | None = None, usd_per_1m_output: float | None = None,
                 max_workers: int = 4, max_retries: int = 5, backoff_seconds: float = 1.0,
                 on_progress=None) -> list[dict]:
    """Classify `rows` (`id`, `text`) with the baseline model. Output is in input order, one record per row.

    A failed call (after retries) is recorded as unparseable with an `error` and is not cached, so a rerun retries it.
    """
    done = [0]
    lock = threading.Lock()

    def one(row: dict) -> dict:
        key = cache.key(model, rendered["hash"], row["id"])
        hit = cache.get(key)
        response, error = hit, None
        if hit is None:
            for attempt in range(max_retries + 1):
                try:
                    started = time.perf_counter()
                    reply = call_model(build_messages(rendered, row["text"]))
                    response = {"raw": reply.get("text"), "input_tokens": int(reply.get("input_tokens") or 0),
                                "output_tokens": int(reply.get("output_tokens") or 0),
                                "latency_ms": (time.perf_counter() - started) * 1000}
                    cache.put(key, response)
                    break
                except Exception as exc:  # noqa: BLE001 - any provider error is retried, then reported
                    error = f"{type(exc).__name__}: {exc}"[:300]
                    if attempt < max_retries:
                        time.sleep(backoff_seconds * 2 ** attempt)
        with lock:
            done[0] += 1
            if on_progress:
                on_progress(done[0], len(rows))
        if response is None:
            return {"id": row["id"], "label": None, "raw": None, "unparseable": True, "error": error,
                    "input_tokens": 0, "output_tokens": 0, "cost_usd": None, "latency_ms": None,
                    "cached": False, "model": model, "prompt_hash": rendered["hash"]}
        label = parse_label(response["raw"], labels)
        return {"id": row["id"], "label": label, "raw": response["raw"], "unparseable": label is None,
                "input_tokens": response["input_tokens"], "output_tokens": response["output_tokens"],
                "cost_usd": call_cost(response["input_tokens"], response["output_tokens"],
                                      usd_per_1m_input, usd_per_1m_output),
                "latency_ms": response["latency_ms"], "cached": hit is not None,
                "model": model, "prompt_hash": rendered["hash"]}

    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as pool:
        return list(pool.map(one, rows))


def summarise(records: list[dict], gold_rows: list[dict] | None = None) -> dict:
    """Counts, tokens and cost; plus accuracy and macro F1 when gold labels are given.

    Unparseable answers count as wrong and are reported as a separate rate (spec 03 section 4).
    """
    n = len(records)
    costs = [r["cost_usd"] for r in records if r.get("cost_usd") is not None]
    out = {
        "n": n,
        "unparseable": sum(bool(r.get("unparseable")) for r in records),
        "unparseable_rate": sum(bool(r.get("unparseable")) for r in records) / n if n else None,
        "failed_calls": sum(bool(r.get("error")) for r in records),
        "cached": sum(bool(r.get("cached")) for r in records),
        "input_tokens": sum(r.get("input_tokens") or 0 for r in records),
        "output_tokens": sum(r.get("output_tokens") or 0 for r in records),
        "cost_usd_total": sum(costs) if costs else None,
        "usd_per_ticket": sum(costs) / len(costs) if costs else None,
        "spent_usd_this_run": sum(r["cost_usd"] for r in records
                                  if r.get("cost_usd") is not None and not r.get("cached")) if costs else None,
    }
    if gold_rows is not None:
        gold_by_id = {r["id"]: r["label"] for r in gold_rows}
        gold = [gold_by_id[r["id"]] for r in records]
        pred = [r["label"] if r.get("label") is not None else UNPARSEABLE for r in records]
        out["accuracy"] = M.accuracy(gold, pred)
        out["macro_f1"] = M.macro_f1(gold, pred, ignore={UNPARSEABLE})
    return out


# --------------------------------------------------------------------------- bundle for a remote run

def _read_jsonl(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: str, rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_json(path: str):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def write_bundle(cfg: dict) -> str:
    """Write everything a remote notebook needs into `baseline.bundle_dir`. Returns that directory.

    Validation rows keep their labels (for prompt development). Test rows are written WITHOUT labels, so test
    accuracy can only be computed by `make eval` in this repository.
    """
    data_root, dest = cfg["paths"]["data"], cfg["baseline"]["bundle_dir"]
    labels = _load_json(os.path.join(data_root, "labels.json"))
    manifest = _load_json(os.path.join(data_root, "manifest.json"))
    rendered = render_prompt(load_prompt(cfg), labels)
    validation = _read_jsonl(os.path.join(data_root, "processed", "validation.jsonl"))
    test = _read_jsonl(os.path.join(data_root, "processed", "test.jsonl"))
    if not validation or not test:
        raise FileNotFoundError("run `make data` before building the baseline bundle")

    if os.path.isdir(dest):
        if os.listdir(dest) and not os.path.exists(os.path.join(dest, BUNDLE_MANIFEST)):
            raise FileExistsError(f"{dest} exists and is not a baseline bundle; refusing to overwrite it")
        shutil.rmtree(dest)
    os.makedirs(os.path.join(dest, "src", "triage"))
    _write_jsonl(os.path.join(dest, "validation.jsonl"), validation)
    _write_jsonl(os.path.join(dest, "test.jsonl"), [{"id": r["id"], "text": r["text"]} for r in test])
    for name, obj in (("labels.json", labels), ("prompt.json", rendered)):
        with open(os.path.join(dest, name), "w", encoding="utf-8") as handle:
            json.dump(obj, handle, indent=2, ensure_ascii=False)
    here = os.path.dirname(os.path.abspath(__file__))
    for name in ("__init__.py", "baseline.py", "metrics.py"):
        shutil.copyfile(os.path.join(here, name), os.path.join(dest, "src", "triage", name))
    bundle = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_version": manifest["data_version"],
        "splits": {name: manifest["splits"][name] for name in ("validation", "test")},
        "labels": len(labels),
        "prompt_hash": rendered["hash"],
        "max_output_tokens": cfg["baseline"].get("max_output_tokens", 20),
    }
    with open(os.path.join(dest, BUNDLE_MANIFEST), "w", encoding="utf-8") as handle:
        json.dump(bundle, handle, indent=2)
    return dest


# --------------------------------------------------------------------------- import of remote results

def import_results(cfg: dict, source_dir: str, tracker=None) -> dict:
    """Validate result files from a remote baseline run and place them under `paths.artifacts`.

    Checks: the prompt hash equals the prompt committed here, the data version matches, and each predictions
    file has exactly the ids of its split. Test accuracy is NOT computed here; `make eval` does that.
    """
    data_root, artifacts = cfg["paths"]["data"], cfg["paths"]["artifacts"]
    summary_path = os.path.join(source_dir, SUMMARY_FILE)
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"{summary_path} not found; download the notebook's `outputs` folder first")
    summary = _load_json(summary_path)
    if summary.get("complete") is False:
        raise ValueError(f"this was a partial run (limit = {summary.get('limit')}); rerun the notebook with limit = 0")
    labels = _load_json(os.path.join(data_root, "labels.json"))
    manifest = _load_json(os.path.join(data_root, "manifest.json"))
    rendered = render_prompt(load_prompt(cfg), labels)
    if summary.get("prompt_hash") != rendered["hash"]:
        raise ValueError("the results were produced with a different prompt than configs/baseline/ in this "
                         "checkout; rebuild the bundle and rerun, or check out the commit that was used")
    if summary.get("data_version") != manifest["data_version"]:
        raise ValueError(f"results are for {summary.get('data_version')}, this checkout has {manifest['data_version']}")

    os.makedirs(artifacts, exist_ok=True)
    targets = {"validation": "baseline_predictions_validation.jsonl", "test": "baseline_predictions.jsonl"}
    imported: dict[str, dict] = {}
    for split, target in targets.items():
        records = _read_jsonl(os.path.join(source_dir, predictions_file(split)))
        if not records:
            continue
        gold = _read_jsonl(os.path.join(data_root, "processed", f"{split}.jsonl"))
        if {r["id"] for r in records} != {r["id"] for r in gold} or len(records) != len(gold):
            raise ValueError(f"{predictions_file(split)} does not contain exactly the ids of the {split} split "
                             f"({len(records)} records, {len(gold)} expected); partial runs cannot be imported")
        if any(r.get("prompt_hash") != rendered["hash"] for r in records):
            raise ValueError(f"{predictions_file(split)} mixes answers from different prompts")
        _write_jsonl(os.path.join(artifacts, target), records)
        imported[split] = summarise(records, gold if split == "validation" else None)
    if "validation" not in imported:
        raise FileNotFoundError(f"{predictions_file('validation')} is missing or empty in {source_dir}")

    result = {
        "model": summary.get("model"), "prompt_hash": rendered["hash"], "data_version": manifest["data_version"],
        "usd_per_1m_input_tokens": summary.get("usd_per_1m_input_tokens"),
        "usd_per_1m_output_tokens": summary.get("usd_per_1m_output_tokens"),
        "remote_run_id": summary.get("mlflow_run_id"), "remote_created": summary.get("created"),
        "imported": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "local_run_id": getattr(tracker, "run_id", None), "splits": imported,
    }
    out_path = os.path.join(artifacts, SUMMARY_FILE)
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    if tracker is not None:
        tracker.set_tags({"baseline": "true"})
        tracker.log_params({"baseline.model_used": result["model"], "baseline.prompt_hash": rendered["hash"],
                            "baseline.remote_run_id": result["remote_run_id"],
                            "data_version": manifest["data_version"]})
        for split, stats in imported.items():
            tracker.log_metrics({f"baseline_{split}_{key}": value for key, value in stats.items()})
        tracker.log_artifact(out_path)
    return result
