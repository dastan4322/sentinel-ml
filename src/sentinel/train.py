import hashlib
import json
import platform
from pathlib import Path
from time import perf_counter

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import FEATURES, split_temporal, validate
from .drift import reference_profile
from .policy import choose_threshold, evaluate


def train(csv: Path, out: Path, source: str, seed=42, fp_cost=5.0, fn_cost=250.0, cap=0.02):
    if source not in {"synthetic", "public"}:
        raise ValueError("source must be synthetic or public")
    if out.exists() and any(out.iterdir()):
        raise ValueError("Output directory must be empty; preserve previous runs")
    started = perf_counter()
    frame = validate(pd.read_csv(csv))
    train_df, val, test = split_temporal(frame)
    models = {
        "logistic": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000, random_state=seed)
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=150,
            max_leaf_nodes=15,
            learning_rate=0.08,
            l2_regularization=2.0,
            early_stopping=False,
            random_state=seed,
        ),
    }
    candidates = {}
    for name, model in models.items():
        # No test rows or validation rows participate in preprocessing or fitting.
        model.fit(train_df[FEATURES], train_df.Class)
        scores = model.predict_proba(val[FEATURES])[:, 1]
        threshold, _cost = choose_threshold(val.Class, scores, fp_cost, fn_cost, cap)
        candidates[name] = {
            "threshold": threshold,
            "validation": evaluate(val.Class, scores, threshold, fp_cost, fn_cost),
        }
    winner = min(
        candidates,
        key=lambda name: (
            candidates[name]["validation"]["cost"],
            -candidates[name]["validation"]["average_precision"],
        ),
    )
    model = models[winner]
    threshold = candidates[winner]["threshold"]
    scores = model.predict_proba(test[FEATURES])[:, 1]
    partitions = {
        name: {
            "rows": len(part),
            "fraud": int(part.Class.sum()),
            "start": float(part.Time.min()),
            "end": float(part.Time.max()),
        }
        for name, part in zip(["train", "validation", "test"], [train_df, val, test])
    }
    report = {
        "source": source,
        "dataset_sha256": hashlib.sha256(csv.read_bytes()).hexdigest(),
        "seed": seed,
        "features": FEATURES,
        "partitions": partitions,
        "policy": {"fp_cost": fp_cost, "fn_cost": fn_cost, "validation_alert_cap": cap},
        "candidates": candidates,
        "winner": winner,
        "threshold": threshold,
        "test": evaluate(test.Class, scores, threshold, fp_cost, fn_cost),
        "no_alert_test_cost": float(fn_cost * test.Class.sum()),
        "test_prevalence": float(test.Class.mean()),
        "training_seconds": perf_counter() - started,
        "environment": {"python": platform.python_version(), "sklearn": sklearn.__version__},
        "limitations": [
            "Synthetic scores are engineering checks only"
            if source == "synthetic"
            else "Two-day anonymized data cannot establish deployment readiness",
            "No delayed-label or customer-identity information",
            "Capacity enforced on validation only; future volume can exceed it",
            "Scores have not been separately calibrated",
        ],
    }
    out.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": model,
        "threshold": threshold,
        "features": FEATURES,
        "source": source,
        "reference": reference_profile(train_df[FEATURES]),
        "sklearn": sklearn.__version__,
    }
    joblib.dump(bundle, out / "model.joblib")
    report["artifact_sha256"] = hashlib.sha256((out / "model.joblib").read_bytes()).hexdigest()
    (out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    pd.DataFrame({"label": test.Class.to_numpy(), "score": scores}).to_csv(
        out / "test_predictions.csv", index=False
    )
    PrecisionRecallDisplay.from_predictions(test.Class, scores, name=winner)
    plt.title(f"{source.upper()} — held-out chronological test")
    plt.tight_layout()
    plt.savefig(out / "precision_recall.png", dpi=140)
    plt.close()
    # API sample is explicitly a test transaction, never a fabricated fraud claim.
    sample = test.iloc[0]
    request = {
        "transactions": [
            {
                "transaction_id": "example-001",
                "amount": float(sample.Amount),
                "v": [float(sample[f"V{i}"]) for i in range(1, 29)],
            }
        ]
    }
    (out / "request.json").write_text(json.dumps(request, indent=2))
    (out / "model-card.md").write_text(
        f"# Model card\n\nSource: **{source}**. Selected model: {winner}.\n\n"
        f"Threshold: {threshold:.6g}. Test AP: {report['test']['average_precision']:.4f}.\n\n"
        "Intended use: portfolio research and local evaluation. No payment decisions.\n\n"
        "See report.json for data hash, partitions, costs, environment and limitations.\n"
    )
    return report
