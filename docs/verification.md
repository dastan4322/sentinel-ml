# Verification evidence

Verified in a clean Python virtual environment on Linux x86_64,
Python 3.12.14. These are cloud-environment measurements, not Mac M1 benchmarks.

- Automated suite: **28 passed**. Covers temporal boundaries, schema validation,
  exact threshold optimization against brute force, API contracts, non-finite input
  handling, artifact/API agreement, saved metric reconstruction, drift detection,
  preservation of previous runs, and isolation of test features from model selection.
- Ruff checks passed; dependency consistency check passed.
- A real Uvicorn process passed HTTP health, prediction, invalid-request and metrics checks.
- GitHub Actions has been configured for Linux/macOS, Python 3.11/3.12. Remote CI
  has not been executed here. Native M1 and Docker execution have not been verified.
- The API test client emits one upstream Starlette/httpx deprecation warning;
  request validation and serving checks pass. The real HTTP test also passes.

## Synthetic run

20,000 generated rows, seed 42; partitions are recorded in the report.
Selected model: `hist_gradient_boosting`.

| Held-out metric | Measured value |
| --- | ---: |
| Fraud prevalence | 0.01625 |
| Average precision | 0.24044 |
| Precision at selected threshold | 0.28986 |
| Recall at selected threshold | 0.30769 |
| Alert rate | 0.01725 |
| Assumed cost units | 11495 |
| No-alert cost units | 16250 |

These validate an intentionally learnable toy process. Do not present these figures
as public-dataset accuracy, real customer performance, or money saved.

## Warm inference measurement

Batch size 100, 100 iterations, p50 4.35 ms,
p95 30.83 ms. Shared cloud CPU, two OpenMP/BLAS threads; resource
contention may affect results. Excludes HTTP, model loading, and concurrent-client load.
Run the benchmark on your M1 before claiming laptop performance.

`demo-results/drift.json` uses the entire generated dataset as a smoke test, not a
production monitoring window. `demo-results/http-smoke.json` captures actual responses.
`demo-results/environment.txt` records the installed packages; install the project
separately with `pip install -e .` when reproducing that environment.
