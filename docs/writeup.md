# European power fair value: day-ahead forecast and prompt-curve view

## 1. Data and quality

**Market.** Germany / DE-LU hourly day-ahead power, with fundamentals at the same resolution.

**Source.** All series are downloaded from **SMARD** (public HTTP, no API key). The request shape is documented in the OpenAPI description used by the community client:  
`https://github.com/bundesAPI/smard-api/blob/main/openapi.yaml`  

Concretely, hourly chunks are fetched from:

- `https://www.smard.de/app/chart_data/{filter}/{region}/index_hour.json`
- `https://www.smard.de/app/chart_data/{filter}/{region}/{filter}_{region}_hour_{timestamp_ms}.json`

**Series (see `config/market.yaml`).** Day-ahead price (DE-LU); system load; onshore/offshore wind and solar generation; SMARD wind and solar **forecasts**. Ingest also derives `wind_total_mw` from onshore + offshore.

**Time alignment.** SMARD timestamps are Unix milliseconds in UTC. Ingest converts them to a single **`timestamp` column in `Europe/Berlin`**, so local hour and DST are handled in one place. Series are outer-joined on time; duplicate hour labels are dropped (`keep=last`). The panel used for the latest run has **15,144** hourly rows from **2024-07-14** through **2026-04-06** (UTC range in the QA report).

**QA (automated).** The pipeline reports missingness by column, duplicate timestamps, IQR-based tail rates as a simple outlier screen, and checks that the dominant timestep is one hour. For the snapshot in `run_snapshot/qa_report.md`: **no duplicate timestamps**, regular **1h** spacing, and missing fractions around **0–1.1%** on price and forecasts (edges from the outer join). I also run **optional LLM-proposed rules** (`python -m src.pipeline.qa --llm-rules`): the model suggests JSON checks (non-null, ranges, etc.), and Python executes them and appends results to the same QA report. Some suggested ranges fail on real data; that is expected and still useful as a second pair of eyes. Prompts and responses are logged under `outputs/logs/` and copied to `run_snapshot/logs/`.

---

## 2. Forecasting and validation

**Target choice (Option A).** I forecast **hourly day-ahead price** (`da_price_eur_mwh`) and then summarise the predicted path into **delivery-period means** and a **spread band** for curve-style discussion (next section). Hourly DA is the natural building block for a desk that marks prompt month/quarter from hourly expectations and shape.

**Features (leakage-aware).** Calendar effects in Berlin (hour, weekday, month); price lags **24h** and **168h** (same hour last week); load lagged 24h; SMARD **wind and solar forecasts** (not same-hour realised renewables), so the feature set is closer to what is knowable before the hour.

**Models.**

- **Baseline 1 — seasonal naive:** prediction = price **168 hours** ago (same hour last week).  
- **Baseline 2 — linear regression** on the feature matrix.  
- **Improved model — `HistGradientBoostingRegressor`** (scikit-learn), trained on the same features.

**Validation.** **Walk-forward:** `TimeSeriesSplit` with five expanding train / future test splits (no shuffle). Metrics on each test slice: **MAE**, **RMSE**, and **tail MAE** on hours where realised price is in the **top decile** of that slice (stress on high prices).

**Results (latest run, mean over folds — see `run_snapshot/validation_summary.md`).**

| Model | Mean MAE (EUR/MWh) |
|--------|---------------------:|
| Same-hour last week (naive) | 35.39 |
| Linear regression | 20.21 |
| Hist gradient boosting | 19.00 |

Mean **tail MAE (HGBR, top-decile hours)** ≈ **35.85 EUR/MWh**. Per-fold **RMSE** is in `run_snapshot/metrics.json`. After cross-validation, a final HGBR is fit on the **first 85%** of rows (time-ordered, after dropping incomplete feature rows) and saved as `hgb_final.pkl`. **14,803** rows enter training after the feature filter.

**Hold-out predictions.** The last **15%** of those rows (chronological) are used for plots and for **`run_snapshot/submission.csv`** (`id`, `y_pred`). **2,221** hours; the window is documented in **`run_snapshot/hold_out_meta.json`** (first/last `id` in Berlin local ISO format).

---

## 3. Prompt-curve translation

The file **`run_snapshot/prompt_curve_view.json`** turns the hold-out **predicted** hourly path into a small **tradable summary**:

- **Mean of predicted prices** over the last **168** hours of the hold-out ≈ **61.25 EUR/MWh** (proxy for “next week” delivery in that window).  
- Mean over the last **720** hours ≈ **78.11 EUR/MWh** (rough “next month” proxy on the same slice).  
- **p10–p90** of predicted levels ≈ **25.5–156.1 EUR/MWh** as a simple **band** on model output (not a full probabilistic forecast).

**Desk use (conceptual).** A trader could compare the week/month means to **broker forward/prompt** quotes for the relevant month or quarter and use the gap as a **directional or relative-value** hint, then express risk via **prompt buckets, shape, or spreads** (e.g. month vs front) depending on book constraints. The band is a crude **uncertainty / stress** read from the scenario path, not a calibrated confidence interval.

**Invalidation.** I would de-emphasise the signal if: **realised DA repeatedly diverges** from the model on the same fundamentals regime; **market structure or policy** shifts so lags and forecasts stop being representative; or **data quality** degrades (missing price/forecast spikes, clock issues). Systematic positive bias in high-price hours (see tail metric) would also warrant a model or feature review.

---

## 4. AI as an engineering lever

Everything is **called from code**, not pasted from a chat UI.

1. **LLM-assisted QA rules** (`src/ai/qa_llm.py`, `python -m src.pipeline.qa --llm-rules`). The model receives dtypes and a few sample rows, returns **JSON rules**, and the pipeline **executes** them and merges outcomes into the QA report. **Logs:** `llm_qa_rules.jsonl` (copy in `run_snapshot/logs/`).

2. **Commentary from numbers only** (`src/ai/commentary.py`, `python -m src.ai.commentary`). The model receives **only** serialised `metrics.json`, `prompt_curve_view.json`, and a trimmed `qa_summary.json`; the prompt asks for a short note **without inventing** statistics. **Logs:** `llm_commentary.jsonl`. Output: `run_snapshot/commentary_latest.txt` after refresh.

**Secrets.** API keys and model choice live in **`.env`** (see `.env.example`); they are **not** committed. Ollama or OpenAI is supported in `src/ai/llm_client.py`.

---

## 5. Reproducibility

**Setup and run order** are in the repo **`README.md`**: create a venv, `pip install -e .`, copy `.env.example` → `.env`, then run `ingest` → `qa` → `train` → `report` (optional `qa --llm-rules`, `commentary`, then `report` again to refresh **`run_snapshot/`**). Dependencies are in **`pyproject.toml`** and **`requirements.txt`**.

**Curated outputs** for figures, QA, metrics, prompt-curve JSON, optional LLM logs, **`submission.csv`**, and **`validation_summary.md`** live under **`run_snapshot/`** (see `run_snapshot/README.md`). The processed panel **`data/processed/panel.parquet`** is rebuilt by ingest and is gitignored because of size.

---

*Figures for this run: `run_snapshot/figures/fig_actual_vs_pred_holdout.png`, `run_snapshot/figures/fig_residual_hist.png`.*
