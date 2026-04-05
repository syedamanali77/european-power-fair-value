import json
from pathlib import Path

import pandas as pd

from src.ai.llm_client import call_llm_chat

ROOT = Path(__file__).resolve().parents[2]
LOG = ROOT / "outputs" / "logs" / "llm_qa_rules.jsonl"


def propose_rules(df: pd.DataFrame, n_sample=8) -> list:
    sample = df.head(n_sample)
    payload = {
        "dtypes": {c: str(df[c].dtype) for c in df.columns},
        "sample": json.loads(sample.to_json(orient="records", date_format="iso")),
    }
    system = (
        'Reply with only JSON: {"rules":[...]}. Each rule is '
        '{"type":"non_null","column":"..."} or '
        '{"type":"unique_index","column":"..."} or '
        '{"type":"range","column":"...","min":null,"max":null} or '
        '{"type":"max_missing_frac","column":"...","max_frac":0.05}. '
        "Use real column names. 5–10 rules."
    )
    raw = call_llm_chat(system, json.dumps(payload, indent=2), LOG, ollama_json=True)
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.removeprefix("```json").removeprefix("```").strip()
        if "```" in raw:
            raw = raw[: raw.index("```")].strip()
    i, j = raw.find("{"), raw.rfind("}") + 1
    if i < 0 or j <= i:
        raise ValueError("no json from model")
    data = json.loads(raw[i:j])
    rules = data.get("rules")
    if not isinstance(rules, list):
        raise ValueError("rules should be a list")
    return rules


def run_rules(df: pd.DataFrame, rules: list) -> list:
    n = len(df)
    out = []
    for i, rule in enumerate(rules):
        t = rule.get("type")
        ok, detail = True, ""
        try:
            if t == "non_null":
                c = rule["column"]
                nz = df[c].isna().sum()
                ok = nz == 0
                detail = f"nulls={int(nz)}"
            elif t == "unique_index":
                c = rule["column"]
                d = df[c].duplicated().sum()
                ok = d == 0
                detail = f"dups={int(d)}"
            elif t == "range":
                c = rule["column"]
                lo, hi = rule.get("min"), rule.get("max")
                s = pd.to_numeric(df[c], errors="coerce")
                bad = 0
                if lo is not None:
                    bad += int((s < float(lo)).sum())
                if hi is not None:
                    bad += int((s > float(hi)).sum())
                ok = bad == 0
                detail = f"out_of_range={bad}"
            elif t == "max_missing_frac":
                c = rule["column"]
                fr = float(df[c].isna().mean()) if n else 0.0
                ok = fr <= float(rule["max_frac"])
                detail = f"missing={fr:.4f}"
            else:
                ok = False
                detail = f"unknown {t}"
        except Exception as e:
            ok, detail = False, repr(e)
        out.append({"rule_index": i, "rule": rule, "ok": bool(ok), "detail": detail})
    return out


# used by qa.py
execute_rules = run_rules
