# Dataset and model limitations

**Status:** working research implementation, verified on generated data. Real public-data
performance has not been measured in the included report.

**Task:** rank anonymized transactions and propose a review decision under an assumed
cost matrix. Intended users are reviewers of an ML engineering portfolio.

**Data:** optional ULB/Kaggle credit-card dataset: European cardholder transactions,
September 2013, approximately two days; V1–V28 are anonymized PCA components. Time is
elapsed seconds; Amount is transaction value. Labels are historical binary fraud labels.
Follow source licensing and attribution requirements before redistributing data.

**Training:** logistic regression with train-fit scaling and histogram gradient boosting;
60/20/20 split by distinct timestamps; validation cost selects model and threshold.

**Leakage limits:** the publisher's PCA provenance is outside this repository's control.
No cardholder identifiers exist to prevent entity overlap or study customer history.
Features cannot support customer velocity, device history, or delayed-label features.
The short observation period limits seasonality and time-generalization conclusions.

**Fairness and privacy:** no protected-attribute audit is possible with this dataset.
Absence of such attributes does not establish fairness. Anonymized components are not
an assurance that any real transaction dataset is safe to publish.

**Scores:** bounded classifier outputs; no separate probability calibration is performed.
Brier score is reported as a diagnostic, not proof of calibration. Label delay and drift
can invalidate score and threshold behavior after deployment.

**Decisions:** a review recommendation, not a claim that a transaction is fraudulent.
The fixed validation threshold does not impose a hard future queue capacity.

**Evidence:** see demo-results/report.json and verification.md. Synthetic data has an
intentionally learnable stationary relationship and does not reproduce real fraud.
