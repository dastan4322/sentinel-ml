import itertools
import json

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from sentinel.api import create_app
from sentinel.data import FEATURES, split_temporal, synthetic, validate
from sentinel.drift import drift_report, reference_profile
from sentinel.policy import choose_threshold, evaluate
from sentinel.train import train


@pytest.fixture(scope="session")
def run(tmp_path_factory):
    directory = tmp_path_factory.mktemp("run")
    csv = directory / "demo.csv"
    synthetic(csv, 6000)
    out = directory / "model"
    report = train(csv, out, "synthetic")
    return csv, out, report


def test_temporal_boundaries_and_shared_timestamps():
    df = pd.DataFrame({"Time": np.repeat(np.arange(100), 2), "Class": [0, 1] * 100})
    train_df, val, test = split_temporal(df)
    assert train_df.Time.max() < val.Time.min() < test.Time.min()
    assert val.Time.max() < test.Time.min()
    assert len(train_df) + len(val) + len(test) == len(df)


def test_single_class_partition_rejected():
    df = pd.DataFrame({"Time": range(100), "Class": [0] * 100})
    with pytest.raises(ValueError, match="both classes"):
        split_temporal(df)


@pytest.mark.parametrize("problem", ["nan", "infinite", "negative", "label", "extra", "duplicate"])
def test_data_contract(tmp_path, problem):
    path = tmp_path / "data.csv"
    synthetic(path, 1000)
    df = pd.read_csv(path)
    if problem == "nan":
        df.loc[0, "V1"] = np.nan
    elif problem == "infinite":
        df.loc[0, "V1"] = np.inf
    elif problem == "negative":
        df.loc[0, "Amount"] = -1
    elif problem == "label":
        df["Class"] = df["Class"].astype(float)
        df.loc[0, "Class"] = 0.5
    elif problem == "extra":
        df["leak"] = df.Class
    else:
        df = pd.concat([df, df.iloc[:1]])
    with pytest.raises(ValueError):
        validate(df)


def test_threshold_matches_brute_force_with_ties():
    scores = np.array([0.8, 0.8, 0.6, 0.2, 0.1])
    for y in itertools.product([0, 1], repeat=5):
        for cap in [0, 0.2, 0.6, 1]:
            t, cost = choose_threshold(y, scores, 5, 20, cap)
            options = []
            for candidate in [1.0000001, *np.unique(scores)]:
                pred = scores >= candidate
                if pred.mean() <= cap:
                    options.append(
                        5 * np.sum(pred & (np.array(y) == 0))
                        + 20 * np.sum(~pred & (np.array(y) == 1))
                    )
            assert cost == min(options)
            assert (scores >= t).mean() <= cap


def test_saved_artifact_agrees_with_api(run):
    _csv, out, report = run
    import joblib

    bundle = joblib.load(out / "model.joblib")
    payload = json.loads((out / "request.json").read_text())
    row = payload["transactions"][0]
    frame = pd.DataFrame([[*row["v"], row["amount"]]], columns=FEATURES)
    expected = bundle["model"].predict_proba(frame)[0, 1]
    with TestClient(create_app(out / "model.joblib")) as client:
        result = client.post("/predict", json=payload)
        assert result.status_code == 200
        body = result.json()
        assert body["predictions"][0]["risk_score"] == pytest.approx(expected)
        assert body["predictions"][0]["review"] == (expected >= report["threshold"])
        assert client.get("/health").json()["source"] == "synthetic"
        assert "sentinel_predictions_total 1.0" in client.get("/metrics").text


@pytest.mark.parametrize(
    "problem", ["short", "negative", "extra", "duplicate", "empty", "oversize", "extreme"]
)
def test_api_contract(run, problem):
    _, out, _ = run
    payload = json.loads((out / "request.json").read_text())
    if problem == "short":
        payload["transactions"][0]["v"] = [0.0] * 27
    elif problem == "negative":
        payload["transactions"][0]["amount"] = -1
    elif problem == "extra":
        payload["transactions"][0]["Class"] = 1
    elif problem == "duplicate":
        payload["transactions"] *= 2
    elif problem == "extreme":
        payload["transactions"][0]["v"][0] = 1e300
    elif problem == "empty":
        payload["transactions"] = []
    else:
        payload["transactions"] *= 1001
    with TestClient(create_app(out / "model.joblib")) as client:
        assert client.post("/predict", json=payload).status_code == 422


def test_test_metrics_reproducible_from_predictions(run):
    _, out, report = run
    saved = pd.read_csv(out / "test_predictions.csv")
    actual = evaluate(saved.label, saved.score, report["threshold"])
    assert actual["cost"] == report["test"]["cost"]
    assert actual["average_precision"] == pytest.approx(report["test"]["average_precision"])
    assert report["partitions"]["train"]["end"] < report["partitions"]["validation"]["start"]


def test_monitor_identical_and_shifted(run):
    csv, _, _ = run
    frame = pd.read_csv(csv)[FEATURES]
    ref = reference_profile(frame)
    assert max(drift_report(ref, frame)["psi"].values()) == pytest.approx(0.0)
    frame["V1"] += 10
    result = drift_report(ref, frame)
    assert "V1" in result["investigate"]
    assert np.isfinite(result["psi"]["V1"])


def test_run_cannot_be_overwritten(run):
    csv, out, _ = run
    with pytest.raises(ValueError, match="empty"):
        train(csv, out, "synthetic")


@pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity"])
def test_nonfinite_request_has_safe_validation_response(run, literal):
    _, out, _ = run
    payload = json.loads((out / "request.json").read_text())
    payload["transactions"][0]["amount"] = "SPECIAL_NUMBER"
    raw = json.dumps(payload).replace('"SPECIAL_NUMBER"', literal)
    with TestClient(create_app(out / "model.joblib")) as client:
        response = client.post(
            "/predict", content=raw, headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422
        assert "input" not in response.json()["detail"][0]


def test_test_features_cannot_change_selection(run, tmp_path):
    csv, _, original = run
    # Preserve training/validation numeric text exactly; avoid CSV round-trip noise.
    lines = csv.read_text().splitlines()
    for index in range(1, len(lines)):
        values = lines[index].split(",")
        if float(values[0]) >= original["partitions"]["test"]["start"]:
            values[1] = str(float(values[1]) + 100)
            lines[index] = ",".join(values)
    modified = tmp_path / "modified.csv"
    modified.write_text("\n".join(lines) + "\n")
    result = train(modified, tmp_path / "new-model", "synthetic")
    assert result["winner"] == original["winner"]
    assert result["threshold"] == original["threshold"]
    assert result["candidates"] == original["candidates"]


@pytest.mark.parametrize(
    "kwargs",
    [{"fp_cost": float("nan")}, {"fn_cost": float("inf")}, {"max_alert_rate": -1}, {"fp_cost": -1}],
)
def test_invalid_policy_rejected(kwargs):
    with pytest.raises(ValueError):
        choose_threshold([0, 1], [0.1, 0.9], **kwargs)
