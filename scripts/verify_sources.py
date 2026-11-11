"""T-02: check that dataset and model IDs resolve, print row counts per split, compare with docs/data_card.md.

Usage: python scripts/verify_sources.py [--config configs/base.yaml] [--card docs/data_card.md]
Exit code 1 if an ID does not resolve or the counts differ from the data card.
"""
from __future__ import annotations

import argparse
import re
import sys

from triage.config import load_config
from triage.data import fetch_source, resolve_revision


def card_counts(path: str) -> dict[str, int]:
    """Read `Train 10,003; test 3,080` style counts from the 'Published splits' row of the data card."""
    text = open(path, encoding="utf-8").read()
    row = next((ln for ln in text.splitlines() if "Published splits" in ln), "")
    return {m.group(1).lower(): int(m.group(2).replace(",", ""))
            for m in re.finditer(r"(train|test)\s+([\d,]+)", row, flags=re.I)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--card", default="docs/data_card.md")
    args = parser.parse_args()
    cfg = load_config(args.config)
    dcfg, mcfg = cfg["data"], cfg["model"]
    failed = False

    source = fetch_source(dcfg["dataset_id"], dcfg.get("revision"))
    print(f"dataset {dcfg['dataset_id']} @ {source['revision']} (pinned in config: {dcfg.get('revision') or 'NO'})")
    counts = {name: len(rows) for name, rows in source["splits"].items()}
    for name, n in counts.items():
        print(f"  {name}: {n} rows")
    print(f"  labels: {len(source['labels'])}")

    expected = card_counts(args.card)
    for name, n in expected.items():
        if counts.get(name) != n:
            print(f"  MISMATCH {name}: data card says {n}, source has {counts.get(name)}")
            failed = True
    if not expected:
        print("  data card has no parseable 'Published splits' counts")
        failed = True

    oos = dcfg.get("oos_dataset_id")
    if oos:
        oos_source = fetch_source(oos, dcfg.get("oos_revision"), subdir=dcfg.get("oos_config", "plus"),
                                  label_column=dcfg.get("oos_label_column", "intent"))
        oos_labels = [x for x in oos_source["labels"] if "oos" in x.lower()]
        print(f"dataset {oos} @ {oos_source['revision']} (pinned in config: {dcfg.get('oos_revision') or 'NO'})")
        print(f"  splits: { {k: len(v) for k, v in oos_source['splits'].items()} }; "
              f"labels: {len(oos_source['labels'])}; out-of-scope label: {oos_labels}")

    print(f"model {mcfg['base_id']} @ {resolve_revision(mcfg['base_id'], mcfg.get('revision'), 'model')} "
          f"(pinned in config: {mcfg.get('revision') or 'NO'})")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
