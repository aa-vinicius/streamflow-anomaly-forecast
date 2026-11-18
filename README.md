# Operational multi-horizon forecasting of pentadal streamflow anomalies

Reproducibility package for the manuscript submitted to the *Journal of Hydroinformatics*. The workflow forecasts centred pentadal streamflow anomalies at Santa Cecília (19094) and Ilha dos Pombos (19097), for D+1 to D+5, using WRF-derived predictors, machine learning, persistence correction, seasonal P80 signals and conformal PI90 intervals.

## Scope of this public repository

This repository contains the source code, frozen model settings, variable definitions, environment specifications, manuscript figures, reproducibility instructions, original input files and the ten processed modelling matrices used for the two outlets and five leads. The data are organised under `data/raw/` and `data/processed/`. Redistribution must be confirmed with the respective providers before the repository is made public.

## Reproducibility

1. Review provenance and redistribution conditions in `DATA_ACCESS.md`.
2. Create the environment from `environment.yml` or `requirements.txt`.
3. Run `python scripts/run_all_models.py --data-root data --output-root outputs`.
5. Use the reporting scripts to regenerate diagnostics and figures.

The temporal protocol uses 2018–2019 for initial fitting, 2020–2022 for expanding-window validation and 2023–2024 for frozen development comparison. Records relating to 2025 may occur in original source files but are not used by the released frozen evaluation workflow.

## Data availability statement for the manuscript

Use the wording in `DATA_AVAILABILITY_TEXT.md`. Replace the GitHub placeholder only after the repository is published.

## Citation and contact

See `CITATION.cff`. Data-access requests should be directed to the corresponding author indicated in the manuscript.

