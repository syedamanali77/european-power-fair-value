import argparse
import json
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.pipeline.train import Y_COL, features

ROOT = Path(__file__).resolve().parents[2]


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
    pd.DataFrame({"id": loc.dt.strftime("%Y-%m-%dT%H:%M:%S%z"), "y_pred": pred}).to_csv(
        args.out_csv, index=False
    )

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


if __name__ == "__main__":
    main()
