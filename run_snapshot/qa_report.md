# Data QA
- rows: 15144
- columns: ['da_price_eur_mwh', 'load_mw', 'wind_onshore_mw', 'wind_offshore_mw', 'solar_mw', 'wind_onshore_fc_mw', 'solar_fc_mw', 'timestamp', 'wind_total_mw']
- duplicate timestamps: 0

## Missing (numeric)

| col | frac |
|---|--:|
| da_price_eur_mwh | 0.0111 |
| wind_onshore_fc_mw | 0.0111 |
| solar_fc_mw | 0.0111 |
| load_mw | 0.0019 |
| wind_offshore_mw | 0.0019 |
| wind_onshore_mw | 0.0019 |
| solar_mw | 0.0019 |
| wind_total_mw | 0.0000 |

## Outliers (IQR)

| col | frac |
|---|--:|
| da_price_eur_mwh | 0.0126 |
| load_mw | 0.0000 |
| wind_onshore_mw | 0.0000 |
| wind_offshore_mw | 0.0000 |
| solar_mw | 0.0062 |
| wind_onshore_fc_mw | 0.0000 |
| solar_fc_mw | 0.0000 |
| wind_total_mw | 0.0000 |

## Spacing

- usual step: `0 days 01:00:00`
- not 1h: 0

## Range

- 2024-07-14 22:00:00+00:00 .. 2026-04-06 21:00:00+00:00 (UTC)

## LLM extra checks

- FAIL {'type': 'non_null', 'column': 'da_price_eur_mwh'} — nulls=168
- OK {'type': 'unique_index', 'column': 'timestamp'} — dups=0
- OK {'type': 'range', 'column': 'load_mw', 'min': 0, 'max': 100000} — out_of_range=0
- FAIL {'type': 'range', 'column': 'wind_onshore_mw', 'min': 0, 'max': 10000} — out_of_range=7692
- OK {'type': 'range', 'column': 'wind_offshore_mw', 'min': 0, 'max': 10000} — out_of_range=0
- FAIL {'type': 'range', 'column': 'solar_mw', 'min': 0, 'max': 10000} — out_of_range=4048
- FAIL {'type': 'non_null', 'column': 'wind_onshore_fc_mw'} — nulls=168
- FAIL {'type': 'non_null', 'column': 'solar_fc_mw'} — nulls=168
- FAIL {'type': 'unique_index', 'column': 'load_mw'} — dups=124
- OK {'type': 'range', 'column': 'wind_total_mw', 'min': 0, 'max': 100000} — out_of_range=0
- OK {'type': 'max_missing_frac', 'column': 'da_price_eur_mwh', 'max_frac': 0.05} — missing=0.0111
