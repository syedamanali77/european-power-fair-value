import argparse
import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]

Y_COL = "da_price_eur_mwh"


def features(df: pd.DataFrame):
    d = df.sort_values("timestamp").reset_index(drop=True).copy()
    ts = pd.to_datetime(d["timestamp"])
    if ts.dt.tz is None:
        raise ValueError("timestamp needs timezone (from ingest)")
    loc = ts.dt.tz_convert("Europe/Berlin")
    d["hour"] = loc.dt.hour.astype(float)
    d["dow"] = loc.dt.dayofweek.astype(float)
    d["month"] = loc.dt.month.astype(float)
    if Y_COL not in d.columns:
        raise SystemExit(f"missing {Y_COL}")

    d["price_lag_24h"] = d[Y_COL].shift(24)
    d["price_lag_168h"] = d[Y_COL].shift(168)
    d["load_lag_24h"] = d["load_mw"].shift(24) if "load_mw" in d.columns else np.nan
    for c in ("wind_onshore_fc_mw", "solar_fc_mw"):
        if c not in d.columns:
            d[c] = np.nan

    cols = [
        "hour",
        "dow",
        "month",
        "price_lag_24h",
        "price_lag_168h",
        "load_lag_24h",
        "wind_onshore_fc_mw",
        "solar_fc_mw",
    ]
    return d, cols


def tail_mae(y, pred, q=0.9):
    cut = np.quantile(y, q)
    m = y >= cut
    return float(mean_absolute_error(y[m], pred[m])) if np.any(m) else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", type=Path, default=ROOT / "data" / "processed" / "panel.parquet")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "outputs" / "models")
    ap.add_argument("--n-splits", type=int, default=5)
    args = ap.parse_args()

    if not args.panel.is_file():
        raise SystemExit(f"missing panel: {args.panel} (run: python -m src.pipeline.ingest)")

    d, cols = features(pd.read_parquet(args.panel))
    d = d.dropna(subset=[Y_COL] + cols)
    X, y = d[cols].to_numpy(float), d[Y_COL].to_numpy(float)

    folds = []
    for fold_i, (tr, te) in enumerate(TimeSeriesSplit(n_splits=args.n_splits).split(X)):
        if len(tr) < 24 * 30:
            continue
        Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]
        naive = d[Y_COL].shift(168).iloc[te].to_numpy()
        ok = ~np.isnan(naive)
        if ok.sum() == 0:
            continue
        yte, naive, Xte = yte[ok], naive[ok], Xte[ok]

        def pack(pred):
            return {
                "mae": float(mean_absolute_error(yte, pred)),
                "rmse": float(np.sqrt(mean_squared_error(yte, pred))),
                "tail_mae_p90": tail_mae(yte, pred),
            }

        lr = LinearRegression().fit(Xtr, ytr)
        hgb = HistGradientBoostingRegressor(
            max_depth=6, learning_rate=0.06, max_iter=300, random_state=42
        )
        hgb.fit(Xtr, ytr)

        folds.append(
            {
                "fold": fold_i,
                "n_train": len(tr),
                "n_test": int(ok.sum()),
                "naive_same_week_hour": pack(naive),
                "linear_regression": pack(lr.predict(Xte)),
                "hist_gradient_boosting": pack(hgb.predict(Xte)),
            }
        )

    if not folds:
        raise SystemExit("no folds — need more rows or lower --n-splits")

    def avg(key, sub):
        return float(np.mean([f[key][sub] for f in folds]))

    summary = {
        "target": Y_COL,
        "features": cols,
        "n_rows_used": len(d),
        "n_splits_requested": args.n_splits,
        "folds": folds,
        "mean_over_folds": {
            "naive_mae": avg("naive_same_week_hour", "mae"),
            "linear_mae": avg("linear_regression", "mae"),
            "hgb_mae": avg("hist_gradient_boosting", "mae"),
            "hgb_tail_mae_p90": avg("hist_gradient_boosting", "tail_mae_p90"),
        },
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    mp = args.out_dir / "metrics.json"
    mp.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("wrote %s", mp)

    cut = int(len(X) * 0.85)
    final = HistGradientBoostingRegressor(
        max_depth=6, learning_rate=0.06, max_iter=300, random_state=42
    )
    final.fit(X[:cut], y[:cut])
    with (args.out_dir / "hgb_final.pkl").open("wb") as f:
        pickle.dump({"model": final, "feature_names": cols}, f)


if __name__ == "__main__":
    main()
