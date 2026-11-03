import numpy as np

from .data import FEATURES


def reference_profile(frame):
    result = {}
    for name in FEATURES:
        internal = np.unique(np.quantile(frame[name], np.linspace(0.1, 0.9, 9)))
        edges = np.r_[-np.inf, internal, np.inf]
        counts = np.histogram(frame[name], bins=edges)[0]
        result[name] = {"cuts": internal.tolist(), "counts": counts.tolist()}
    return result


def drift_report(reference, current):
    """PSI with fixed train quantiles and additive smoothing; not a performance metric."""
    if not len(current) or not np.isfinite(current[FEATURES].to_numpy()).all():
        raise ValueError("Drift window must contain finite observations")
    result = {}
    for name, profile in reference.items():
        counts = np.histogram(current[name], bins=[-np.inf, *profile["cuts"], np.inf])[0]
        expected = np.asarray(profile["counts"], dtype=float) + 0.5
        observed = counts.astype(float) + 0.5
        expected /= expected.sum()
        observed /= observed.sum()
        result[name] = float(np.sum((observed - expected) * np.log(observed / expected)))
    return {
        "rows": len(current),
        "psi": result,
        "investigate": [k for k, v in result.items() if v >= 0.2],
        "note": "0.2 is an illustrative heuristic. Drift alone does not justify retraining.",
    }
