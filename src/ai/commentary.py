import argparse
import json
import logging
from pathlib import Path

from src.ai.llm_client import call_llm_chat

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]
LOG = ROOT / "outputs" / "logs" / "llm_commentary.jsonl"

_QA_KEYS_FOR_NOTE = (
    "n_rows",
    "duplicate_timestamps",
    "missingness",
    "iqr_outlier_frac",
    "time_start_utc",
    "time_end_utc",
)


def _slim_qa_summary(data):
    if not isinstance(data, dict):
        return data
    return {k: data[k] for k in _QA_KEYS_FOR_NOTE if k in data}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, default=ROOT / "outputs" / "models")
    ap.add_argument("--qa-dir", type=Path, default=ROOT / "outputs" / "qa")
    ap.add_argument("--model", default=None)
    args = ap.parse_args()

    blob = {}
    for path, key in [
        (args.model_dir / "metrics.json", "metrics"),
        (args.model_dir / "prompt_curve_view.json", "prompt_curve_view"),
        (args.qa_dir / "qa_summary.json", "qa_summary"),
    ]:
        if path.exists():
            raw = json.loads(path.read_text(encoding="utf-8"))
            blob[key] = _slim_qa_summary(raw) if key == "qa_summary" else raw
        else:
            blob[key] = None

    system = (
        "You write a short desk note for power markets (DE/LU day-ahead). "
        "Output plain English only: a few short paragraphs, no lists unless necessary. "
        "Do not print JSON, code fences, or keys. Do not repeat or dump the input. "
        "Use only numbers that appear in the data; round sensibly (e.g. one decimal for EUR/MWh). "
        "If metrics.mean_over_folds exists, treat naive_mae / linear_mae / hgb_mae as the walk-forward headline — do not say MAEs are missing. "
        "Mention hold-out curve summary from prompt_curve_view if present, and one or two QA facts if present "
        "(for duplicates, only the duplicate_timestamps field counts — not duplicate values in load or price columns). "
        "If a section is missing, say so in one phrase. Max 180 words. No markdown headings."
    )
    user = (
        "Summarize the following run artifacts for a colleague. "
        "Infer meaning from field names; do not echo the structure.\n\n"
        + json.dumps(blob, indent=2)
    )

    try:
        text = call_llm_chat(system, user, LOG, model=args.model)
    except Exception as e:
        logger.error("%s", e)
        text = f"(commentary failed: {e!r})"

    out = ROOT / "outputs" / "commentary_latest.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n", encoding="utf-8")
    logger.info("wrote %s", out)


if __name__ == "__main__":
    main()
