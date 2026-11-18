# Variable dictionary

| Symbol / field | Definition | Unit |
|---|---|---|
| `outlet` | Outlet code: 19094 (Santa Cecília) or 19097 (Ilha dos Pombos) | — |
| `emissao_D` | Operational forecast issue date and time | date/time |
| `h` | Lead of the centre of the forecast window, 1–5 | day |
| `centro_t` | Centre date, `t=D+h` | date |
| `Q_daily` | Observed naturalised daily streamflow | m³ s⁻¹ |
| `Qclim_daily` | Daily seasonal reference estimated only in 1998–2017 | m³ s⁻¹ |
| `Q5` | Mean streamflow over `t−2,…,t+2` | m³ s⁻¹ |
| `C5` | Mean of the five daily seasonal references | m³ s⁻¹ |
| `alvo_A5_m3s` | Pentadal anomaly, `A5=Q5−C5` | m³ s⁻¹ |
| `P20_A5_m3s` | Seasonal 20th percentile of pentadal anomaly | m³ s⁻¹ |
| `P80_A5_m3s` | Seasonal 80th percentile of pentadal anomaly | m³ s⁻¹ |
| `Q_D`, `Q_Dm1`, ... | Naturalised streamflow available at issue time and antecedent lags | m³ s⁻¹ |
| WRF predictors | Reduced set of basin-aggregated forecast attributes; exact selected names are frozen by outlet and lead | variable-specific |

No values or row-level records are included in this dictionary.

