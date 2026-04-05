import argparse
import json
import math
import pickle
import shutil
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.pipeline.train import Y_COL, features

ROOT = Path(__file__).resolve().parents[2]
RUN_SNAPSHOT = ROOT / "run_snapshot"


def _fmt_num(x, nd=2):
    if x is None:
        return "—"
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "—"
    if math.isnan(v):
        return "—"
    return f"{v:.{nd}f}"


def _write_validation_summary(model_dir: Path, snap: Path) -> None:
    mp = model_dir / "metrics.json"
    if not mp.is_file():
        return
    m = json.loads(mp.read_text(encoding="utf-8"))
    mo = m.get("mean_over_folds") or {}
    lines = [
        "# Walk-forward validation\n\n",
        f"Target: `{m.get('target')}`  \n",
        f"Rows after feature drop: {m.get('n_rows_used')}  \n",
        f"TimeSeriesSplit folds (requested): {m.get('n_splits_requested')}\n\n",
        "## Mean MAE over folds (EUR/MWh)\n\n",
        "| Model | MAE |\n|---|--:|\n",
        f"| Same-hour last week (naive) | {_fmt_num(mo.get('naive_mae'))} |\n",
        f"| Linear regression | {_fmt_num(mo.get('linear_mae'))} |\n",
        f"| Hist gradient boosting | {_fmt_num(mo.get('hgb_mae'))} |\n\n",
        "## High-price hours (HGBR)\n\n",
        "| Metric | Value (EUR/MWh) |\n|---|--:|\n",
        f"| Mean tail MAE (top decile of realised price per fold) | {_fmt_num(mo.get('hgb_tail_mae_p90'))} |\n",
    ]
    (snap / "validation_summary.md").write_text("".join(lines), encoding="utf-8")


def _export_run_snapshot(pred_df: pd.DataFrame, args) -> None:
    """Copy the latest hold-out run into run_snapshot/ for sharing alongside the repo."""
    RUN_SNAPSHOT.mkdir(parents=True, exist_ok=True)
    (RUN_SNAPSHOT / "figures").mkdir(exist_ok=True)
    (RUN_SNAPSHOT / "logs").mkdir(exist_ok=True)

    pred_df.to_csv(RUN_SNAPSHOT / "submission.csv", index=False)

    first_id = str(pred_df["id"].iloc[0]) if len(pred_df) else None
    last_id = str(pred_df["id"].iloc[-1]) if len(pred_df) else None
    hold_out = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "definition": "Chronological tail after dropping rows with missing features; not used to fit hgb_final.pkl.",
        "test_frac": args.test_frac,
        "n_rows": int(len(pred_df)),
        "id_format": "Hour start, Europe/Berlin, ISO-8601 with numeric offset.",
        "first_id": first_id,
        "last_id": last_id,
        "columns": ["id", "y_pred"],
    }
    (RUN_SNAPSHOT / "hold_out_meta.json").write_text(
        json.dumps(hold_out, indent=2), encoding="utf-8"
    )

    _write_validation_summary(args.model_dir, RUN_SNAPSHOT)

    copies: list[tuple[Path, Path]] = [
        (args.out_fig / "fig_actual_vs_pred_holdout.png", RUN_SNAPSHOT / "figures" / "fig_actual_vs_pred_holdout.png"),
        (args.out_fig / "fig_residual_hist.png", RUN_SNAPSHOT / "figures" / "fig_residual_hist.png"),
        (ROOT / "outputs" / "qa" / "qa_report.md", RUN_SNAPSHOT / "qa_report.md"),
        (ROOT / "outputs" / "qa" / "qa_summary.json", RUN_SNAPSHOT / "qa_summary.json"),
        (args.model_dir / "metrics.json", RUN_SNAPSHOT / "metrics.json"),
        (args.model_dir / "prompt_curve_view.json", RUN_SNAPSHOT / "prompt_curve_view.json"),
        (ROOT / "outputs" / "commentary_latest.txt", RUN_SNAPSHOT / "commentary_latest.txt"),
        (ROOT / "outputs" / "logs" / "llm_commentary.jsonl", RUN_SNAPSHOT / "logs" / "llm_commentary.jsonl"),
        (ROOT / "outputs" / "logs" / "llm_qa_rules.jsonl", RUN_SNAPSHOT / "logs" / "llm_qa_rules.jsonl"),
    ]
    for src, dst in copies:
        if src.is_file():
            shutil.copy2(src, dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", type=Path, default=ROOT / "data" / "processed" / "panel.parquet")
    ap.add_argument("--model-dir", type=Path, default=ROOT / "outputs" / "models")
    ap.add_argument("--out-fig", type=Path, default=ROOT / "outputs" / "figures")
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--out-csv", type=Path, default=ROOT / "outputs" / "predictions.csv")
    args = ap.parse_args()

    if not args.panel.is_file():
        raise SystemExit(f"missing panel: {args.panel} (run: python -m src.pipeline.ingest)")
    pkl = args.model_dir / "hgb_final.pkl"
    if not pkl.is_file():
        raise SystemExit(f"missing model: {pkl} (run: python -m src.pipeline.train)")

    d, fcols = features(pd.read_parquet(args.panel))
    d = d.dropna(subset=[Y_COL] + fcols)

    with pkl.open("rb") as f:
        model = pickle.load(f)["model"]

    cut = int(len(d) * (1.0 - args.test_frac))
    test = d.iloc[cut:]
    y = test[Y_COL].to_numpy(float)
    pred = model.predict(test[fcols].to_numpy(float))

    args.out_fig.mkdir(parents=True, exist_ok=True)
    loc = pd.to_datetime(test["timestamp"]).dt.tz_convert("Europe/Berlin")
    x = np.arange(len(test))

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x, y, label="actual")
    ax.plot(x, pred, label="model", alpha=0.85)
    ax.set_title("DE/LU DA price (hold-out)")
    ax.set_xlabel("hour index")
    ax.set_ylabel("EUR/MWh")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out_fig / "fig_actual_vs_pred_holdout.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(pred - y, bins=40, color="steelblue", alpha=0.85)
    ax.axvline(0, color="black")
    ax.set_title("Residuals")
    ax.set_xlabel("pred - actual (EUR/MWh)")
    fig.tight_layout()
    fig.savefig(args.out_fig / "fig_residual_hist.png", dpi=150)
    plt.close(fig)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    pred_df = pd.DataFrame({"id": loc.dt.strftime("%Y-%m-%dT%H:%M:%S%z"), "y_pred": pred})
    pred_df.to_csv(args.out_csv, index=False)

    n = len(pred)
    wk, mo = min(n, 24 * 7), min(n, 24 * 30)
    desk = {
        "held_out_hours": n,
        "forecast_delivery_mean_next_week_eur_mwh": float(np.mean(pred[-wk:])),
        "forecast_delivery_mean_next_month_proxy_eur_mwh": float(np.mean(pred[-mo:])),
        "held_out_pred_p10_p90_eur_mwh": [float(np.percentile(pred, 10)), float(np.percentile(pred, 90))],
        "interpretation": "Compare week/month means to prompt month or quarter.",
        "invalidation_examples": [
            "Forecasts vs actuals keep wrong.",
            "Market regime change.",
            "Bad or missing data.",
        ],
    }
    (args.model_dir / "prompt_curve_view.json").write_text(json.dumps(desk, indent=2), encoding="utf-8")

    _export_run_snapshot(pred_df, args)


if __name__ == "__main__":
    main()
