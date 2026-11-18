# Data provenance and access conditions

The prepared repository includes the original input collection under `data/raw/` and ten modelling matrices under `data/processed/`, one for each outlet and lead:

```text
data/processed/19094/matriz_completa_19094_Dmais1.parquet
...
data/processed/19094/matriz_completa_19094_Dmais5.parquet
data/processed/19097/matriz_completa_19097_Dmais1.parquet
...
data/processed/19097/matriz_completa_19097_Dmais5.parquet
```

The raw collection contains WRF-derived attributes, the joint naturalised-streamflow workbook, pentadal target files, original delivery archives and clearly labelled auxiliary or superseded files. Only matrices in `data/processed/19094` and `data/processed/19097` are inputs to the released training workflow. Files explicitly labelled `NAO_USAR`, `PENDENTE`, `INCOMPATIVEL` or `NAO_USADOS` are retained only for provenance and must not be used for model fitting.

Before public release, the authors must obtain written confirmation that the operational WRF-derived attributes and naturalised streamflow records may be redistributed. If authorisation is incomplete, remove the affected files from the public repository and retain the conditional-access wording in `DATA_AVAILABILITY_TEXT.md`.

Because two original CSV files exceed GitHub's ordinary 100 MB file limit, all tabular and archive formats are configured for Git Large File Storage (Git LFS) in `.gitattributes`.

