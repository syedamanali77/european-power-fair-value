# European power (DE/LU) — project notes

This document matches how the code in `src/` actually runs. Figures live under `outputs/figures/`; QA text under `outputs/qa/qa_report.md`.

## Data

Market: **Germany / DE-LU**, hourly.

**Source:** SMARD (Bundesnetzagentur), public HTTP API, no key. Filter IDs:  
https://github.com/bundesAPI/smard-api/blob/main/openapi.yaml  

URLs (hourly):

- `https://www.smard.de/app/chart_data/{filter}/{region}/index_hour.json`
- `https://www.smard.de/app/chart_data/{filter}/{region}/{filter}_{region}_hour_{timestamp_ms}.json`

**Series pulled (from `config/market.yaml`):**

| Column | Filter | Region | Note |
|--------|--------|--------|------|
| `da_price_eur_mwh` | 4169 | DE-LU | Day-ahead price (EUR/MWh) |
| `load_mw` | 410 | DE | Load |
| `wind_onshore_mw` | 4067 | DE | Wind |
| `wind_offshore_mw` | 1225 | DE | Wind |
| `solar_mw` | 4068 | DE | Solar |
| `wind_onshore_fc_mw` | 123 | DE | Wind forecast |
| `solar_fc_mw` | 126 | DE | Solar forecast |

After download, ingest adds **`wind_total_mw`** = onshore + offshore (NaNs treated as 0 in that sum).

Timestamps: SMARD gives Unix **ms** → converted to timezone-aware **`Europe/Berlin`**. Duplicate timestamps are dropped (`keep=last`). Tables are merged with an **outer** join on time, so edge periods can have extra NaNs.

Run: `python -m src.pipeline.ingest` → `data/processed/panel.parquet`.

## QA

`python -m src.pipeline.qa` → `outputs/qa/qa_report.md` and `outputs/qa/qa_summary.json`.

Checks: row count, duplicate times, missing share per numeric column, simple IQR-based tail rate, usual hour spacing, time range.

Optional: `python -m src.pipeline.qa --llm-rules` — the model proposes JSON rules; Python runs them. Log: `outputs/logs/llm_qa_rules.jsonl`.

**LLM:** Put `OLLAMA_MODEL=…` in `.env` for local Ollama, or `OPENAI_API_KEY=…` for OpenAI. If both are set, Ollama is used unless `LLM_BACKEND=openai`. See `.env.example` and the repo README.

## Forecasting

**Target:** `da_price_eur_mwh` (in code this is the constant `Y_COL` in `src/pipeline/train.py`).

**Features** are built in `features()` in the same file: hour, weekday, month (calendar in Berlin); price lagged 24h and 168h; `load_mw` lagged 24h if present; `wind_onshore_fc_mw` and `solar_fc_mw`. Same-hour **actual** wind/solar generation are **not** used as inputs (only forecasts), so the setup is closer to something you could use before the hour is over.

**Models compared (walk-forward):**

- **Naive:** same hour last week (168h lag of price).
- **Linear regression** on the feature matrix.
- **`HistGradientBoostingRegressor`** (sklearn), fixed hyperparameters in `train.py`.

**Validation:** `TimeSeriesSplit` with 5 folds (no shuffle). Metrics per fold: MAE, RMSE, and MAE on test points where the realised price is in the top decile (`tail_mae_p90`). Summary: `outputs/models/metrics.json`.

**Final saved model:** after CV, a single HGBR is fit on the **first 85%** of rows (ordered in time) and saved as `outputs/models/hgb_final.pkl` together with the feature name list.

## Report and plots

`python -m src.pipeline.report` loads `hgb_final.pkl`, builds the same features, and evaluates on the **last 15%** of rows (default `--test-frac 0.15`) for charts and CSV output.

Figures (defaults):

- `outputs/figures/fig_actual_vs_pred_holdout.png`
- `outputs/figures/fig_residual_hist.png`

CSV: `outputs/predictions.csv` (`id` timestamp string, `y_pred`).

The same `report` step refreshes **`run_snapshot/`** (plots, QA + metrics copies, `submission.csv`, `hold_out_meta.json`, `validation_summary.md`). See `run_snapshot/README.md`.

`outputs/models/prompt_curve_view.json` — short summary stats on the hold-out predictions (means over last week / ~month of hours, p10–p90). Handy to compare to a forward month or quarter on a desk, or to paste into a prompt.

## LLM commentary

`python -m src.ai.commentary` reads `metrics.json`, `prompt_curve_view.json`, and `qa_summary.json` (if present), calls the model once, writes `outputs/commentary_latest.txt`, and appends to `outputs/logs/llm_commentary.jsonl`.

## Repo layout (where to look)

| Piece | Location |
|--------|----------|
| SMARD download / merge | `src/data/smard.py`, `src/pipeline/ingest.py` |
| QA | `src/pipeline/qa.py` |
| Features, CV, pickle | `src/pipeline/train.py` |
| Plots, predictions CSV, run snapshot | `src/pipeline/report.py` |
| Ollama / OpenAI wrapper | `src/ai/llm_client.py` |
