# Reproducibility protocol

## Required inputs

The included data and their expected variables and units are documented in `DATA_ACCESS.md` and `schema/VARIABLE_DICTIONARY.md`. Each processed row represents one emission date `D`, one outlet and one forecast lead. The centred target at `t=D+h` uses daily streamflow from `t-2` to `t+2`; future observations are used only to construct historical targets and verify forecasts.

## Run order

```text
python scripts/validate_repository.py
python scripts/run_all_models.py --data-root data --output-root outputs
python scripts/summarize_v2.py outputs/19094 19094
python scripts/summarize_v2.py outputs/19097 19097
```

The training script evaluates Elastic Net, support-vector regression, Extremely Randomised Trees and XGBoost for the continuous anomaly. Logistic regression, Extremely Randomised Trees and XGBoost are evaluated for the direct P80 signal. Candidate hybrid weights combine the selected machine-learning prediction with persistence. Bias correction, P80 thresholding and split-conformal PI90 calibration are estimated without using the 2023–2024 comparison period.

## Determinism

The fixed random seed is 20261117. Library versions are pinned in `requirements.txt`. Exact selected settings are stored by outlet and lead in `configs/19094/` and `configs/19097/`.

## Generated outputs

Generated model binaries, row-level predictions and diagnostics are written below `outputs/`, which is excluded from version control to avoid mixing reproducible inputs with generated artefacts. Aggregate tables and figures can be regenerated from the included processed matrices.

