# Make the project your own

## Before publishing

1. Reproduce the synthetic run and explain every module without relying on generated text.
2. Obtain public data from its original source, document duplicates, freeze a protocol,
   and run one public-data experiment in a new output directory.
3. Write a short findings section: selected model, AP versus prevalence, recall/precision,
   alert rate, assumed cost versus no-alert policy, and limitations. Include actual counts.
4. Add your Mac's measured inference latency, hardware, Python version and batch size.
5. Run the tests locally and GitHub Actions. Publish small reports and figures only;
   keep raw transactions, secrets, binary models, and large prediction CSVs out of Git.

## Suggested GitHub description

End-to-end fraud detection with chronological evaluation, cost-aware alert policies,
FastAPI inference, drift checks, Docker, and CI. CPU-first Python implementation.

## Five-minute demo

Show the business objective and the no-alert baseline. Explain train/validation/test
ordering. Show candidate validation metrics and the frozen threshold. Send a real API
request, then an invalid request. Show the artifact fingerprint and Prometheus counters.
Explain why synthetic AP and a two-day public benchmark cannot prove deployment value.

## Interview questions to prepare

- Why optimize average precision and policy cost instead of accuracy?
- How can the publisher's PCA introduce leakage you cannot fix here?
- Why can a threshold chosen for a 2% validation alert rate exceed 2% tomorrow?
- What changes if fraud labels arrive a week late?
- Why is PSI insufficient to trigger automatic retraining?
- What happens if two equal scores straddle review capacity?
- Why preserve the chosen model instead of refitting after threshold selection?
- When would a feature store or Kafka be justified?

## CV wording after verification

“Built a reproducible fraud scoring pipeline with chronological model evaluation,
validation-selected review thresholds, a versioned FastAPI service, and automated tests.”

Add numeric improvements only after measuring them on the public dataset and stating
the baseline and assumptions. Do not describe simulated cost reductions as money saved
in production. Credit any tools or assistance according to your institution/employer rules.

## Publish from your machine

Create an empty GitHub repository named `sentinel-ml`, then run these commands from
this project directory, replacing YOUR_USERNAME:

```bash
git init
git add .
git commit -m "Build temporal fraud detection and serving pipeline"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/sentinel-ml.git
git push -u origin main
```

The repository includes a code license. Check that its authorship/notice fits your
publication, add a profile link, and use your own measured findings in the README.
