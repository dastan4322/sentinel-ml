# Engineering decisions

## Data flow

CSV → strict contract → chronological train/validation/test → train-only preprocessing
→ validation model/policy selection → frozen artifact → held-out report → FastAPI.
A separate batch workflow compares recent features to the frozen train reference.

## Why these choices

- **Logistic regression plus histogram boosting:** a transparent linear baseline and
  a nonlinear tabular model with modest CPU and memory requirements. No GPU dependency.
- **No class weighting by default:** weights change the fitted score distribution.
  Start with empirical probabilities and tune the review policy explicitly; weighting
  remains a controlled experiment, not an automatic improvement.
- **No randomized holdout:** ordering approximates evaluation on future transactions.
  This cannot correct unknown leakage in the original publisher's PCA transformation.
- **No internal random early stopping:** boosting uses a fixed iteration count.
- **Exact threshold sweep:** cumulative confusion counts at distinct scores keep ties
  intact and find the best feasible validation threshold in O(n log n).
- **No final refit:** refitting changes scores and can invalidate the selected threshold.
- **File artifacts instead of a registry service:** adequate for one developer and an
  8 GB laptop. Hashes and preserved run directories provide traceability.
- **Batch prediction:** reduces per-transaction overhead without adding a message broker.

## Costs and capacity

FP cost represents unnecessary review/friction; FN cost represents missed fraud.
Both are fixed, illustrative units; transaction-specific loss, review cost for true
positives, recoveries, and opportunity cost are absent. Change `--fp-cost`, `--fn-cost`,
and `--cap` for sensitivity analysis using validation data. Do not choose assumptions
by searching the test outcomes. Threshold above one represents the no-alert policy.

## Deployment boundary

This is a local portfolio service. Production work would require authentication,
transport security, request body limits at a proxy, durable queues and deduplication,
label feedback, feature contracts with upstream systems, artifact signing, rollout
controls, and operational ownership. None is claimed to be implemented here.

A single process loads one trusted model at startup. Concurrent request processing
uses FastAPI's synchronous endpoint thread pool; CPU work is limited by OpenMP and
BLAS settings. Run one worker on an 8 GB machine initially. Measurements should state
hardware, batch size, warmup, concurrent load and whether HTTP is included.

## Useful next experiments

Use rolling time windows; report dispersion across periods. Add bootstrap intervals
only with a justified temporal/block sampling method. Compare calibration using a
separate chronological calibration partition and reliability plots. Evaluate cost
sensitivity and precision/recall at several review budgets. Keep test data untouched
until the experimental protocol is frozen.
