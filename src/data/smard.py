# SMARD filters: https://github.com/bundesAPI/smard-api/blob/main/openapi.yaml

import json
import logging
import time

import pandas as pd
import requests

logger = logging.getLogger(__name__)
BASE = "https://www.smard.de/app/chart_data"


def get_json(url: str, session: requests.Session) -> dict:
    time.sleep(0.12)
    r = session.get(url, timeout=60)
    r.raise_for_status()
    return r.json()


def download_series(name, filter_id, region, resolution, session, max_chunks=80):
    url = f"{BASE}/{filter_id}/{region}/index_{resolution}.json"
    idx = list((get_json(url, session).get("timestamps") or []))
    if not idx:
        logger.warning("no index for %s", name)
        return pd.DataFrame(columns=["ts_utc", name])

    idx = sorted(idx)[-max_chunks:] if max_chunks else sorted(idx)
    parts = []
    for t0 in idx:
        u = f"{BASE}/{filter_id}/{region}/{filter_id}_{region}_{resolution}_{t0}.json"
        try:
            data = get_json(u, session)
        except (requests.HTTPError, json.JSONDecodeError) as e:
            logger.warning("chunk skip %s: %s", name, e)
            continue
        rows = [(int(a), float(b)) for a, b in (data.get("series") or []) if b is not None]
        if not rows:
            continue
        df = pd.DataFrame(rows, columns=["ts_utc", "value"]).rename(columns={"value": name})
        parts.append(df)

    if not parts:
        return pd.DataFrame(columns=["ts_utc", name])
    out = pd.concat(parts, ignore_index=True).drop_duplicates("ts_utc").sort_values("ts_utc")
    return out


def merge_on_timestamp(dfs, how="outer"):
    dfs = [d for d in dfs if not d.empty and "ts_utc" in d.columns]
    if not dfs:
        return pd.DataFrame()
    m = dfs[0]
    for d in dfs[1:]:
        m = m.merge(d, on="ts_utc", how=how)
    return m.sort_values("ts_utc").reset_index(drop=True)


def ms_to_timestamp(df: pd.DataFrame, tz: str) -> pd.DataFrame:
    if df.empty or "ts_utc" not in df.columns:
        return df
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["ts_utc"], unit="ms", utc=True).dt.tz_convert(tz)
    return out.drop(columns=["ts_utc"])
