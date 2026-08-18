# dimer-dataset-validator-mitra-regressor

DIMER dataset validator for the Mitra regressor pipeline. It checks that an uploaded
tabular-regression dataset zip meets the CSV contract before fine-tuning runs.

- Runs as a CPU Kubernetes Job.
- DIMER builds the root `Dockerfile` into an ECR image and runs `validator.py`.
- Pairs with `dimer-finetuner-mitra-regressor`.

The complete pipeline documentation, dataset specification, and the fine-tuner are in the
[mitra-regressor-pipeline](https://github.com/kurtvalcorza/mitra-regressor-pipeline) project.

## Contract summary

The dataset zip must contain a `train.csv` with a numeric `target` column; every other
non-dropped column is a feature. The validator reports pass/fail per check in `result.json`.
See the project's dataset specification for the full rules.
