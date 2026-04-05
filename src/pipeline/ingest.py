import argparse
import logging
from pathlib import Path

import requests
import yaml

from src.data.smard import download_series, merge_on_timestamp, ms_to_timestamp

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=ROOT / "config" / "market.yaml")
    ap.add_argument("--max-chunks", type=int, default=90)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "processed" / "panel.parquet")
    args = ap.parse_args()

    if not args.config.is_file():
        raise SystemExit(f"missing config: {args.config}")

    cfg = yaml.safe_load(args.config.read_text())
    tz = cfg.get("timezone", "Europe/Berlin")
    sm = cfg.get("smard") or {}

    sess = requests.Session()
    sess.headers["User-Agent"] = "power-forecast/0.1"

    dfs = []
    for col, spec in sm.items():
        if not isinstance(spec, dict):
            continue
        fid, reg = int(spec["filter"]), str(spec["region"])
        logger.info("fetch %s (%s %s)", col, fid, reg)
        dfs.append(download_series(col, fid, reg, "hour", sess, args.max_chunks))

    if not dfs:
        raise SystemExit("nothing in config smard:")

    panel = ms_to_timestamp(merge_on_timestamp(dfs), tz)
    if "timestamp" not in panel.columns:
        raise SystemExit("empty panel")

    if "wind_onshore_mw" in panel.columns and "wind_offshore_mw" in panel.columns:
        panel["wind_total_mw"] = panel["wind_onshore_mw"].fillna(0) + panel["wind_offshore_mw"].fillna(0)

    panel = panel.sort_values("timestamp").reset_index(drop=True)
    if panel["timestamp"].duplicated().any():
        panel = panel.drop_duplicates("timestamp", keep="last")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(args.out, index=False)
    logger.info("saved %s rows -> %s", len(panel), args.out)


if __name__ == "__main__":
    main()
