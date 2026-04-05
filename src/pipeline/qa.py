import argparse
import json
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]


def iqr_tail_frac(s: pd.Series) -> float:
    x = pd.to_numeric(s, errors="coerce").dropna()
    if len(x) < 20:
        return 0.0
    q1, q3 = x.quantile([0.25, 0.75])
    iqr = q3 - q1
    if not iqr:
        return 0.0
    lo, hi = q1 - 3 * iqr, q3 + 3 * iqr
    return float(((x < lo) | (x > hi)).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", type=Path, default=ROOT / "data" / "processed" / "panel.parquet")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "outputs" / "qa")
    ap.add_argument("--llm-rules", action="store_true", help="extra rules from LLM (.env)")
    args = ap.parse_args()

    if not args.panel.is_file():
        raise SystemExit(f"missing panel: {args.panel} (run: python -m src.pipeline.ingest)")

    df = pd.read_parquet(args.panel)
    if "timestamp" not in df.columns:
        raise SystemExit("need timestamp column")

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    lines = ["# Data QA\n", f"- rows: {len(df)}\n", f"- columns: {list(df.columns)}\n"]
    dups = int(df["timestamp"].duplicated().sum())
    lines.append(f"- duplicate timestamps: {dups}\n")

    nums = [c for c in df.columns if c != "timestamp" and pd.api.types.is_numeric_dtype(df[c])]
    miss = df[nums].isna().mean().sort_values(ascending=False)

    lines.append("\n## Missing (numeric)\n\n| col | frac |\n|---|--:|\n")
    for c, v in miss.items():
        lines.append(f"| {c} | {v:.4f} |\n")

    lines.append("\n## Outliers (IQR)\n\n| col | frac |\n|---|--:|\n")
    for c in nums:
        lines.append(f"| {c} | {iqr_tail_frac(df[c]):.4f} |\n")

    if len(df) >= 2:
        dt = df["timestamp"].diff().dropna()
        mode = dt.mode().iloc[0] if len(dt.mode()) else None
        lines.append(f"\n## Spacing\n\n- usual step: `{mode}`\n")
        lines.append(f"- not 1h: {(dt != pd.Timedelta(hours=1)).sum()}\n")

    t0, t1 = df["timestamp"].min(), df["timestamp"].max()
    lines.append(f"\n## Range\n\n- {t0} .. {t1} (UTC)\n")

    summary = {
        "n_rows": len(df),
        "duplicate_timestamps": dups,
        "missingness": {str(k): float(v) for k, v in miss.items()},
        "iqr_outlier_frac": {c: iqr_tail_frac(df[c]) for c in nums},
        "time_start_utc": t0.isoformat() if pd.notna(t0) else None,
        "time_end_utc": t1.isoformat() if pd.notna(t1) else None,
    }

    if args.llm_rules:
        try:
            from src.ai.qa_llm import execute_rules, propose_rules

            rules = propose_rules(df)
            results = execute_rules(df, rules)
            lines.append("\n## LLM extra checks\n\n")
            for r in results:
                lines.append(f"- {'OK' if r['ok'] else 'FAIL'} {r['rule']} — {r['detail']}\n")
            summary["llm_rules"] = rules
            summary["llm_rule_results"] = results
        except Exception as e:
            logger.warning("llm rules: %s", e)
            summary["llm_rules_error"] = repr(e)
            lines.append(f"\n## LLM extra checks\n\n- error: {e!r}\n")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "qa_report.md").write_text("".join(lines), encoding="utf-8")
    (args.out_dir / "qa_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("wrote %s", args.out_dir / "qa_report.md")


if __name__ == "__main__":
    main()
