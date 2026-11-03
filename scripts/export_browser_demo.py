"""Export the trusted synthetic HGB artifact for browser inference and verify in Node."""

import hashlib
import json
import subprocess
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sentinel.data import split_temporal, validate

ROOT = Path(__file__).resolve().parents[1]
bundle = joblib.load(ROOT / "artifacts/demo/model.joblib")
model = bundle["model"]
if bundle["source"] != "synthetic" or type(model).__name__ != "HistGradientBoostingClassifier":
    raise ValueError("This exporter supports only the numeric synthetic HGB demo")
if model.is_categorical_ is not None and model.is_categorical_.any():
    raise ValueError("Categorical splits are unsupported")
output = ROOT / "demo"
output.mkdir(exist_ok=True)
trees = []
for predictors in model._predictors:
    nodes = predictors[0].nodes
    trees.append(
        [
            [
                int(n["feature_idx"]),
                float(n["num_threshold"]),
                int(n["left"]),
                int(n["right"]),
                float(n["value"]),
                int(n["is_leaf"]),
            ]
            for n in nodes
        ]
    )
dataset = ROOT / "data/synthetic.csv"
report = json.loads((ROOT / "artifacts/demo/report.json").read_text())
if hashlib.sha256(dataset.read_bytes()).hexdigest() != report["dataset_sha256"]:
    raise ValueError("Demo data differs from the model training input")
frame = split_temporal(validate(pd.read_csv(dataset)))[2]
scores = model.predict_proba(frame[bundle["features"]])[:, 1]
samples = []
for name, idx in [
    ("Lower-score example", int(np.argmin(scores))),
    ("Higher-score example", int(np.argmax(scores))),
]:
    row = frame.iloc[idx]
    samples.append(
        {
            "name": name,
            "transaction": {
                "transaction_id": f"demo-{idx}",
                "amount": float(row.Amount),
                "v": [float(row[f"V{i}"]) for i in range(1, 29)],
            },
        }
    )
export = {
    "kind": "numeric-hgb",
    "source": "synthetic",
    "features": bundle["features"],
    "threshold": bundle["threshold"],
    "baseline": float(model._baseline_prediction[0, 0]),
    "trees": trees,
    "samples": samples,
    "note": "Examples selected by minimum/maximum test scores to illustrate both decisions.",
}
(output / "model.json").write_text(json.dumps(export, separators=(",", ":"), allow_nan=False))
verification = {
    "features": frame[bundle["features"]].to_numpy().tolist(),
    "expected": scores.tolist(),
}
indices = list(range(0, len(verification["features"]), 125))
(output / "test-vectors.json").write_text(
    json.dumps({k: [v[i] for i in indices] for k, v in verification.items()}, separators=(",", ":"))
)
verification_path = ROOT.parent / "browser-parity-input.json"
verification_path.write_text(json.dumps(verification))
result = subprocess.run(
    ["node", str(output / "verify.mjs"), str(verification_path)],
    check=True,
    capture_output=True,
    text=True,
)
print(result.stdout)
(output / "parity.json").write_text(result.stdout)
