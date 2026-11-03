import argparse
import json
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd

from .data import FEATURES, synthetic
from .drift import drift_report
from .train import train


def main():
    parser = argparse.ArgumentParser(description="Sentinel ML workflows")
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("generate")
    demo.add_argument("--out", type=Path, default=Path("data/synthetic.csv"))
    demo.add_argument("--rows", type=int, default=20000)
    demo.add_argument("--seed", type=int, default=42)
    fit = sub.add_parser("train")
    fit.add_argument("--csv", type=Path, required=True)
    fit.add_argument("--out", type=Path, required=True)
    fit.add_argument("--source", choices=["synthetic", "public"], required=True)
    fit.add_argument("--seed", type=int, default=42)
    fit.add_argument("--fp-cost", type=float, default=5.0)
    fit.add_argument("--fn-cost", type=float, default=250.0)
    fit.add_argument("--cap", type=float, default=0.02)
    for cmd in ["monitor", "benchmark"]:
        p = sub.add_parser(cmd)
        p.add_argument("--model", type=Path, required=True)
        p.add_argument("--csv", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "generate":
        synthetic(args.out, args.rows, args.seed)
        print(f"Synthetic demo written to {args.out}")
    elif args.command == "train":
        report = train(
            args.csv, args.out, args.source, args.seed, args.fp_cost, args.fn_cost, args.cap
        )
        print(
            json.dumps(
                {"winner": report["winner"], "source": report["source"], "test": report["test"]},
                indent=2,
            )
        )
    else:
        bundle = joblib.load(args.model)  # Trusted artifacts only.
        frame = pd.read_csv(args.csv)[FEATURES]
        if args.command == "monitor":
            print(json.dumps(drift_report(bundle["reference"], frame), indent=2))
        else:
            if not len(frame):
                raise ValueError("Benchmark needs at least one row")
            batch = frame.iloc[: min(100, len(frame))]
            bundle["model"].predict_proba(batch)
            elapsed = []
            for _ in range(100):
                start = perf_counter()
                bundle["model"].predict_proba(batch)
                elapsed.append((perf_counter() - start) * 1000)
            print(
                json.dumps(
                    {
                        "batch_size": len(batch),
                        "iterations": 100,
                        "p50_ms": float(np.percentile(elapsed, 50)),
                        "p95_ms": float(np.percentile(elapsed, 95)),
                        "scope": "In-process warm inference; excludes HTTP and concurrency",
                    },
                    indent=2,
                )
            )
