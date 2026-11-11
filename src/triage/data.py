"""Download, split, check and write the dataset and manifest. Implements specs/01-data.md.

Everything except `fetch_source` is pure and works on in-memory rows, so it is testable without network access.
A row is `{"id": str, "text": str, "label": str}`.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import statistics
from datetime import datetime, timezone

log = logging.getLogger("triage.data")

SPLITS = ("train", "validation", "test")


class DataCheckError(Exception):
    """Raised when a spec 01 section 6 check fails."""


# --------------------------------------------------------------------------- source (network)

def _parquet_label_names(path: str, column: str = "label") -> list[str] | None:
    """Read ClassLabel names from the Hugging Face metadata embedded in a parquet file."""
    import pyarrow.parquet as pq

    meta = pq.read_schema(path).metadata or {}
    raw = meta.get(b"huggingface")
    if not raw:
        return None
    features = json.loads(raw).get("info", {}).get("features", {})
    names = features.get(column, {}).get("names")
    if isinstance(names, dict):  # {"0": "a", "1": "b"}
        names = [names[k] for k in sorted(names, key=int)]
    return list(names) if names else None


def resolve_revision(repo_id: str, revision: str | None, repo_type: str = "dataset") -> str:
    """Return the commit hash for `repo_id` at `revision` (or the current head when None)."""
    from huggingface_hub import HfApi

    api = HfApi()
    info = api.dataset_info(repo_id, revision=revision) if repo_type == "dataset" else api.model_info(
        repo_id, revision=revision)
    return info.sha


def fetch_source(dataset_id: str, revision: str | None, subdir: str = "data", label_column: str = "label") -> dict:
    """Download the parquet files of a Hub dataset at a pinned revision.

    Returns `{"revision": sha, "labels": [...], "splits": {name: [{"text", "label"}, ...]}}`.
    Rows are in published order; labels are names, not indices.
    """
    import pyarrow.parquet as pq
    from huggingface_hub import snapshot_download

    sha = resolve_revision(dataset_id, revision)
    root = snapshot_download(dataset_id, repo_type="dataset", revision=sha, allow_patterns=[f"{subdir}/*.parquet"])
    folder = os.path.join(root, subdir)
    splits: dict[str, list[dict]] = {}
    labels: list[str] | None = None
    for name in sorted(os.listdir(folder)):
        if not name.endswith(".parquet"):
            continue
        split = name.split("-")[0]
        path = os.path.join(folder, name)
        names = _parquet_label_names(path, label_column)
        if names and labels is None:
            labels = names
        table = pq.read_table(path).to_pydict()
        splits[split] = [{"text": t, "label": lab} for t, lab in zip(table["text"], table[label_column])]
    if labels is None:
        raise DataCheckError(f"{dataset_id}@{sha}: no ClassLabel names found in parquet metadata")
    for rows in splits.values():
        for row in rows:
            if isinstance(row["label"], int):
                row["label"] = labels[row["label"]]
    return {"revision": sha, "labels": labels, "splits": splits}


# --------------------------------------------------------------------------- processing (pure)

def to_rows(split: str, raw: list[dict]) -> list[dict]:
    """Normalise (strip whitespace, drop empty, keep casing) and assign stable ids from published position."""
    rows = []
    for index, item in enumerate(raw):
        text = str(item["text"]).strip()
        if text:
            rows.append({"id": f"{split}-{index:05d}", "text": text, "label": item["label"]})
    return rows


def dedupe(rows: list[dict]) -> list[dict]:
    """Remove exact-duplicate texts, keeping the first occurrence."""
    seen: set[str] = set()
    out = []
    for row in rows:
        if row["text"] not in seen:
            seen.add(row["text"])
            out.append(row)
    return out


def remove_overlap(rows: list[dict], reference: list[dict]) -> list[dict]:
    """Drop rows whose text also appears in `reference` (the test split)."""
    blocked = {r["text"] for r in reference}
    return [r for r in rows if r["text"] not in blocked]


def _order_key(seed: int, row_id: str) -> str:
    return hashlib.sha256(f"{seed}|{row_id}".encode()).hexdigest()


def stratified_split(rows: list[dict], fraction: float, seed: int) -> tuple[list[dict], list[dict]]:
    """Split rows into (train, validation), stratified by label.

    Deterministic and independent of Python version: rows are ordered within each label by a seeded hash of the id.
    Every label with at least two rows keeps at least one row in each side. Output keeps the input order.
    """
    by_label: dict[str, list[dict]] = {}
    for row in rows:
        by_label.setdefault(row["label"], []).append(row)
    val_ids: set[str] = set()
    for label in sorted(by_label):
        group = sorted(by_label[label], key=lambda r: _order_key(seed, r["id"]))
        n_val = round(len(group) * fraction)
        if len(group) >= 2:
            n_val = min(max(n_val, 1), len(group) - 1)
        val_ids.update(r["id"] for r in group[:n_val])
    train = [r for r in rows if r["id"] not in val_ids]
    val = [r for r in rows if r["id"] in val_ids]
    return train, val


def subsample(rows: list[dict], limit: int | None, seed: int) -> list[dict]:
    """Deterministic subset of at most `limit` rows (smoke runs only), keeping input order."""
    if limit is None or len(rows) <= limit:
        return rows
    keep = {r["id"] for r in sorted(rows, key=lambda r: _order_key(seed, r["id"]))[:limit]}
    return [r for r in rows if r["id"] in keep]


# --------------------------------------------------------------------------- corrections (T-13, spec 05 HF-5)

def merge_corrections(train: list[dict], validation: list[dict], test: list[dict], corrections: list[dict],
                      labels: list[str]) -> tuple[list[dict], dict]:
    """Merge reviewed corrections into the training split only. Returns (new train rows, report).

    The label is `corrected` for an `incorrect` verdict and the model's `prediction` for a `correct` one.
    An item whose id or text is in test or validation is never added to train; it is counted in the report.
    A text already in train is relabelled in place; any other item is appended with id `feedback-<id>`.
    """
    report = {"added": 0, "relabelled": 0, "unchanged": 0, "excluded_test": 0, "skipped_validation": 0,
              "skipped_unsure": 0, "skipped_unknown_label": 0, "excluded_test_ids": []}
    known = set(labels)
    test_ids, test_texts = {r["id"] for r in test}, {r["text"] for r in test}
    val_ids, val_texts = {r["id"] for r in validation}, {r["text"] for r in validation}
    out = [dict(r) for r in train]
    by_text = {r["text"]: r for r in out}
    for item in corrections:
        text = str(item.get("input_ref", "")).strip()
        verdict = item.get("verdict")
        if verdict not in ("correct", "incorrect") or not text:
            report["skipped_unsure"] += 1
            continue
        if item["id"] in test_ids or text in test_texts:      # checked first: test items never reach training
            report["excluded_test"] += 1
            report["excluded_test_ids"].append(item["id"])
            continue
        label = item.get("corrected") if verdict == "incorrect" else item.get("prediction")
        if label not in known:                                 # includes NEW:<name> proposals
            report["skipped_unknown_label"] += 1
        elif item["id"] in val_ids or text in val_texts:
            report["skipped_validation"] += 1
        elif text in by_text:
            if by_text[text]["label"] == label:
                report["unchanged"] += 1
            else:
                by_text[text]["label"] = label
                report["relabelled"] += 1
        else:
            row = {"id": f"feedback-{item['id']}", "text": text, "label": label}
            out.append(row)
            by_text[text] = row
            report["added"] += 1
    report["applied"] = report["added"] + report["relabelled"]
    return out, report


# --------------------------------------------------------------------------- checks (T-04)

def length_stats(rows: list[dict]) -> dict:
    """Character and word length summary for one split."""
    if not rows:
        return {"rows": 0}
    chars = sorted(len(r["text"]) for r in rows)
    words = sorted(len(r["text"].split()) for r in rows)
    p95 = lambda xs: xs[min(len(xs) - 1, int(0.95 * len(xs)))]  # noqa: E731
    return {
        "rows": len(rows),
        "chars_mean": round(statistics.fmean(chars), 1), "chars_p95": p95(chars), "chars_max": chars[-1],
        "words_mean": round(statistics.fmean(words), 1), "words_p95": p95(words), "words_max": words[-1],
    }


def check_splits(splits: dict[str, list[dict]], labels: list[str], previous_labels: list[str] | None = None,
                 allow_label_change: bool = False, require_coverage: bool = True) -> None:
    """Spec 01 section 6. Raises DataCheckError listing every failure. `require_coverage` is off for smoke subsets."""
    problems: list[str] = []
    names = list(splits)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            shared_ids = {r["id"] for r in splits[a]} & {r["id"] for r in splits[b]}
            shared_text = {r["text"] for r in splits[a]} & {r["text"] for r in splits[b]}
            if shared_ids:
                problems.append(f"{len(shared_ids)} ids shared between {a} and {b}, e.g. {sorted(shared_ids)[:3]}")
            if shared_text:
                problems.append(f"{len(shared_text)} texts shared between {a} and {b}, e.g. "
                                f"{[t[:60] for t in sorted(shared_text)[:3]]}")
    known = set(labels)
    for name, rows in splits.items():
        unknown = {r["label"] for r in rows} - known
        if unknown:
            problems.append(f"{name} has labels not in labels.json: {sorted(unknown)[:5]}")
        if require_coverage:
            missing = known - {r["label"] for r in rows}
            if missing:
                problems.append(f"{name} is missing {len(missing)} labels, e.g. {sorted(missing)[:5]}")
    if previous_labels is not None and previous_labels != labels and not allow_label_change:
        problems.append("label list differs from the previous data/labels.json and data.allow_label_change is false")
    if problems:
        raise DataCheckError("data checks failed:\n  - " + "\n  - ".join(problems))


# --------------------------------------------------------------------------- output

def _dump_jsonl(path: str, rows: list[dict]) -> str:
    """Write rows as canonical JSONL (LF, sorted keys, UTF-8) and return the SHA-256 of the file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = "".join(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n" for r in rows).encode("utf-8")
    with open(path, "wb") as handle:
        handle.write(payload)
    return hashlib.sha256(payload).hexdigest()


def _read_jsonl(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _read_json(path: str):
    if os.path.exists(path):
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    return None


def _next_version(previous: str | None) -> str:
    if not previous:
        return "data-v1"
    return f"data-v{int(previous.rsplit('v', 1)[1]) + 1}"


def build_from_source(cfg: dict, source: dict) -> dict:
    """Run spec 01 sections 3-6 on an already-loaded `source` and write the output files. Returns the manifest."""
    dcfg = cfg["data"]
    root = cfg.get("paths", {}).get("data", "data")
    seed = dcfg["seed"]
    labels = list(source["labels"])

    test = to_rows("test", source["splits"]["test"])  # normalised only: the published test set stays the benchmark
    published = to_rows("train", source["splits"]["train"])
    train_all = dedupe(published)
    n_dupes = len(published) - len(train_all)
    train_all = remove_overlap(train_all, test)
    n_leaks = len(published) - n_dupes - len(train_all)
    train, validation = stratified_split(train_all, dcfg["validation_fraction"], seed)

    corrections_path = cfg.get("feedback", {}).get("corrections")
    corrections = _read_jsonl(corrections_path) if corrections_path else []
    train, merge_report = merge_corrections(train, validation, test, corrections, labels)
    if corrections:
        log.info("corrections: %s", {k: v for k, v in merge_report.items() if k != "excluded_test_ids"})
        if merge_report["excluded_test"]:
            log.warning("%d corrected items came from the test split and were NOT added to training: %s",
                        merge_report["excluded_test"], merge_report["excluded_test_ids"][:10])

    subsampled = dcfg.get("max_train_samples") is not None or dcfg.get("max_eval_samples") is not None
    train = subsample(train, dcfg.get("max_train_samples"), seed)
    validation = subsample(validation, dcfg.get("max_eval_samples"), seed)
    test = subsample(test, dcfg.get("max_eval_samples"), seed)
    splits = {"train": train, "validation": validation, "test": test}

    labels_path = os.path.join(root, "labels.json")
    manifest_path = os.path.join(root, "manifest.json")
    previous_labels = _read_json(labels_path)
    check_splits(splits, labels, previous_labels, dcfg.get("allow_label_change", False),
                 require_coverage=not subsampled)

    stats = {name: length_stats(rows) for name, rows in splits.items()}
    log.info("removed %d exact-duplicate train texts and %d train texts that also occur in test", n_dupes, n_leaks)
    for name, s in stats.items():
        log.info("%s: %s", name, s)

    hashes = {name: {"rows": len(rows), "sha256": _dump_jsonl(os.path.join(root, "processed", f"{name}.jsonl"), rows)}
              for name, rows in splits.items()}
    labels_payload = (json.dumps(labels, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    os.makedirs(root, exist_ok=True)
    with open(labels_path, "wb") as handle:
        handle.write(labels_payload)

    manifest = {
        "data_version": "data-v1",
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": {"dataset_id": dcfg["dataset_id"], "revision": source["revision"]},
        "seed": seed,
        "splits": hashes,
        "labels_sha256": hashlib.sha256(labels_payload).hexdigest(),
        "corrections_applied": merge_report["applied"],
    }
    previous = _read_json(manifest_path)
    fingerprint = lambda m: (m["splits"], m["labels_sha256"], m["source"])  # noqa: E731
    if previous and fingerprint(previous) == fingerprint(manifest):
        manifest["data_version"], manifest["created"] = previous["data_version"], previous["created"]
    elif previous:
        manifest["data_version"] = _next_version(previous.get("data_version"))
    with open(manifest_path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    manifest["stats"] = stats
    manifest["corrections"] = merge_report
    return manifest


def build_dataset(cfg: dict) -> dict:
    """Download the pinned source, build all splits, merge corrections, run the checks and write the files."""
    dcfg = cfg["data"]
    source = fetch_source(dcfg["dataset_id"], dcfg.get("revision"))
    if not dcfg.get("revision"):
        log.warning("data.revision is not pinned; using %s. Pin it in configs/base.yaml.", source["revision"])
    return build_from_source(cfg, source)
