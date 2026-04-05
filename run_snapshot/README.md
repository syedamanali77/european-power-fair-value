# Run snapshot

This folder is a **pinned copy** of the latest end-to-end run: plots, QA exports, metrics, optional LLM logs, and out-of-sample predictions. It updates when you run:

```bash
python -m src.pipeline.report
```

| Path | Description |
|------|-------------|
| `submission.csv` | Hold-out predictions: `id` (hour, Berlin time), `y_pred` (EUR/MWh). Same rows as `outputs/predictions.csv`. |
| `hold_out_meta.json` | How that slice is defined (`test_frac`, row count, first/last `id`). |
| `validation_summary.md` | Walk-forward MAE table from `metrics.json`. |
| `figures/` | Actual vs predicted + residual histogram. |
| `qa_report.md`, `qa_summary.json` | From `outputs/qa/` if you ran `qa` first. |
| `metrics.json`, `prompt_curve_view.json` | From `outputs/models/`. |
| `commentary_latest.txt` | Present if you ran `python -m src.ai.commentary`. |
| `logs/` | `llm_*.jsonl` if present. |

Typical pipeline: `ingest` → `qa` → `train` → `report` (optional: `qa --llm-rules`, `commentary` before `report` so copies stay aligned).
