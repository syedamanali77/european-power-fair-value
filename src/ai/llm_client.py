import json
import os
from datetime import datetime, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_ROOT / ".env")


def _log(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _want_ollama() -> bool:
    b = (os.environ.get("LLM_BACKEND") or "").lower().strip()
    if b == "openai":
        return False
    if b == "ollama":
        return True
    return bool((os.environ.get("OLLAMA_MODEL") or "").strip())


def _ollama_base() -> str:
    h = (os.environ.get("OLLAMA_HOST") or "http://127.0.0.1:11434").strip().rstrip("/")
    for suf in ("/api/chat", "/v1/chat/completions", "/v1", "/api"):
        if h.lower().endswith(suf.lower()):
            h = h[: -len(suf)].rstrip("/")
    return h or "http://127.0.0.1:11434"


def _ollama_has_model(base: str, model: str) -> bool:
    try:
        r = requests.get(f"{base}/api/tags", timeout=15)
        r.raise_for_status()
        for m in r.json().get("models") or []:
            n = m.get("name") or ""
            if n.split(":")[0] == model or n.startswith(model + ":"):
                return True
    except Exception:
        return True
    return False


def call_llm_chat(
    system: str,
    user: str,
    log_path: Path,
    model: str | None = None,
    *,
    ollama_json: bool = False,
) -> str:
    full = system + "\n---\n" + user
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    now = datetime.now(timezone.utc).isoformat()

    if _want_ollama():
        m = model or os.environ.get("OLLAMA_MODEL") or "llama3.2"
        base = _ollama_base()
        if not _ollama_has_model(base, m):
            raise RuntimeError(f"Ollama has no '{m}'. Run: ollama pull {m}")

        body = {
            "model": m,
            "messages": msgs,
            "stream": False,
            "options": {"temperature": 0.2},
        }
        if ollama_json:
            body["format"] = "json"

        try:
            r = requests.post(f"{base}/api/chat", json=body, timeout=600)
            reply = ""
            if r.ok:
                reply = ((r.json() or {}).get("message") or {}).get("content") or ""
            reply = reply.strip()
            if not reply:
                body2 = {"model": m, "messages": msgs, "temperature": 0.2, "stream": False}
                if ollama_json:
                    body2["response_format"] = {"type": "json_object"}
                r2 = requests.post(f"{base}/v1/chat/completions", json=body2, timeout=600)
                r2.raise_for_status()
                ch = (r2.json().get("choices") or [{}])[0]
                reply = ((ch.get("message") or {}).get("content") or "").strip()
            _log(
                log_path,
                {
                    "time_utc": now,
                    "backend": "ollama",
                    "model": m,
                    "prompt_preview": full[:2000],
                    "answer_preview": reply[:4000] if reply else None,
                    "error": None,
                },
            )
            return reply
        except Exception as e:
            _log(
                log_path,
                {
                    "time_utc": now,
                    "backend": "ollama",
                    "model": m,
                    "prompt_preview": full[:2000],
                    "answer_preview": None,
                    "error": repr(e),
                },
            )
            raise

    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("Set OPENAI_API_KEY or OLLAMA_MODEL in .env (see README).")

    from openai import OpenAI

    m = model or "gpt-4o-mini"
    client = OpenAI(api_key=key)
    try:
        out = client.chat.completions.create(
            model=m,
            temperature=0.2,
            messages=msgs,
        )
        text = (out.choices[0].message.content or "").strip()
        _log(
            log_path,
            {
                "time_utc": now,
                "backend": "openai",
                "model": m,
                "prompt_preview": full[:2000],
                "answer_preview": text[:4000],
                "error": None,
            },
        )
        return text
    except Exception as e:
        _log(
            log_path,
            {
                "time_utc": now,
                "backend": "openai",
                "model": m,
                "prompt_preview": full[:2000],
                "answer_preview": None,
                "error": repr(e),
            },
        )
        raise
