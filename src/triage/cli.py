"""Command-line entry point. One subcommand per pipeline stage."""
from __future__ import annotations

import argparse
import json
import logging
import os

from . import feedback
from .config import load_config
from .tracking import Tracker


def _paths(cfg: dict) -> dict:
    return cfg.get("paths", {})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="triage")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("data", "train", "predict", "evaluate", "feedback-export", "feedback-ingest"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--config", default="configs/base.yaml")
        cmd.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
        if name == "feedback-ingest":
            cmd.add_argument("--reviewed", required=True)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = load_config(args.config, args.set)
    paths = _paths(cfg)
    fb = cfg.get("feedback", {})

    if args.command == "data":
        from .data import build_dataset

        build_dataset(cfg)
    elif args.command == "train":
        from .train import train

        with Tracker(cfg, run_name="train") as tracker:
            train(cfg, tracker)
    elif args.command == "predict":
        from .predict import predict

        with Tracker(cfg, run_name="predict") as tracker:
            predict(cfg, tracker)
    elif args.command == "evaluate":
        from .evaluate import evaluate_files

        with Tracker(cfg, run_name="evaluate") as tracker:
            report = evaluate_files(cfg)
            out = os.path.join(paths.get("artifacts", "artifacts"), "report.json")
            os.makedirs(os.path.dirname(out), exist_ok=True)
            with open(out, "w", encoding="utf-8") as handle:
                json.dump(report, handle, indent=2)
            tracker.log_metrics(report["metrics"])
            tracker.log_artifact(out)
            print(json.dumps(report["metrics"], indent=2))
    elif args.command == "feedback-export":
        count = feedback.export_review_queue(
            fb["predictions_log"], fb["review_queue"], fb["corrections"],
            threshold=fb["review_threshold"], audit_rate=fb["audit_rate"],
        )
        print(f"wrote {count} items to {fb['review_queue']}")
    elif args.command == "feedback-ingest":
        counts = feedback.ingest_reviewed(args.reviewed, fb["corrections"])
        print(json.dumps({**counts, **feedback.review_summary(fb["corrections"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
