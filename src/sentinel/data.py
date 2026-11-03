from pathlib import Path

import numpy as np
import pandas as pd

FEATURES = [f"V{i}" for i in range(1, 29)] + ["Amount"]
COLUMNS = ["Time", *FEATURES, "Class"]


def validate(df: pd.DataFrame) -> pd.DataFrame:
    if set(df.columns) != set(COLUMNS):
        raise ValueError(f"Expected exactly {COLUMNS}; received {list(df.columns)}")
    if len(df) < 100:
        raise ValueError("At least 100 transactions are required")
    df = df[COLUMNS].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(df.to_numpy()).all():
        raise ValueError("Missing and non-finite values are forbidden")
    if (df[["Time", "Amount"]] < 0).any().any():
        raise ValueError("Time and Amount must be nonnegative")
    if not df.Class.isin([0, 1]).all():
        raise ValueError("Class must contain only 0 and 1")
    if df.duplicated().any():
        raise ValueError("Exact duplicate transactions found; investigate before splitting")
    return df.sort_values("Time", kind="stable").reset_index(drop=True)


def split_temporal(df: pd.DataFrame):
    # Split unique timestamps, so simultaneous transactions never cross boundaries.
    times = np.sort(df.Time.unique())
    if len(times) < 10:
        raise ValueError("At least 10 distinct timestamps are required")
    t1, t2 = times[int(len(times) * 0.6)], times[int(len(times) * 0.8)]
    parts = (df[df.Time < t1], df[(df.Time >= t1) & (df.Time < t2)], df[df.Time >= t2])
    if any(p.Class.nunique() != 2 for p in parts):
        raise ValueError("Every temporal partition must contain both classes")
    return parts


def synthetic(path: Path, n: int = 20000, seed: int = 42):
    if n < 1000:
        raise ValueError("Generate at least 1000 demo rows")
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 28))
    # Intentionally learnable, stationary toy process. Never benchmark claims on it.
    risk = 1.7 * x[:, 0] - 1.4 * x[:, 3] + 0.8 * x[:, 1] * x[:, 2] - 6.4
    y = rng.binomial(1, 1 / (1 + np.exp(-risk)))
    df = pd.DataFrame(x, columns=FEATURES[:-1])
    df["Amount"] = rng.lognormal(3.2, 1.2, n)
    df["Time"] = np.arange(n, dtype=float)
    df["Class"] = y
    path.parent.mkdir(parents=True, exist_ok=True)
    df[COLUMNS].to_csv(path, index=False)
