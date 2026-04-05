# Walk-forward validation

Target: `da_price_eur_mwh`  
Rows after feature drop: 14803  
TimeSeriesSplit folds (requested): 5

## Mean MAE over folds (EUR/MWh)

| Model | MAE |
|---|--:|
| Same-hour last week (naive) | 35.39 |
| Linear regression | 20.21 |
| Hist gradient boosting | 19.00 |

## High-price hours (HGBR)

| Metric | Value (EUR/MWh) |
|---|--:|
| Mean tail MAE (top decile of realised price per fold) | 35.85 |
