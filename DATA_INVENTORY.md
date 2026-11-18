# Data inventory

## Raw inputs

| Directory | Contents | Role in the study |
|---|---|---|
| `data/raw/00_ANEXOS_ORIGINAIS` | Original delivered daily attribute tables and delivery archive | Provenance copy; not read directly by the final training script |
| `data/raw/01_WRF_ATRIBUTOS_SELECIONADOS` | Selected WRF attributes for outlets 19094 and 19097, D+1–D+5 | Meteorological predictors |
| `data/raw/02_VAZAO_NATURAL` | Joint naturalised-streamflow workbook and original outlet packages | Hydrological predictors and target construction |
| `data/raw/03_ALVOS_PENTADAIS` | Pentadal anomalies and seasonal classes | Target verification and P80 classification |
| `data/raw/04_INSUMOS_AUXILIARES_NAO_USADOS` | Auxiliary or incompatible historical deliveries | Provenance only; explicitly excluded from modelling |

Warnings embedded in filenames and accompanying `AVISO_NAO_USAR` files are normative. A file marked pending, incompatible, superseded or unused must not enter training.

## Processed modelling matrices

`data/processed/19094` and `data/processed/19097` contain the complete matrices for D+1–D+5 in CSV and Parquet. CSV and Parquet versions represent alternate serialisations of the same modelling tables; the training workflow reads Parquet.

## Integrity

`checksums/SHA256SUMS.txt` records the SHA-256 hash of every included raw and processed data file. Recalculate the hashes after an authorised data update and document the change.

## Publication requirement

The technical package is complete, but public release is conditional on confirmation that both WRF-derived predictors and naturalised-streamflow records may be redistributed. Git LFS is required because some original files exceed GitHub's regular per-file limit.

