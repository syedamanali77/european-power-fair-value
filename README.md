# european-power-fair-value

DE/LU hourly day-ahead price: download SMARD data, QA, sklearn model, plots, optional LLM (Ollama or OpenAI).

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
cp .env.example .env
```

## LLM

**Ollama:** install [ollama.com](https://ollama.com), run `ollama pull llama3.2`, keep the app open. In `.env`: `OLLAMA_MODEL=llama3.2`.

**OpenAI:** `OPENAI_API_KEY=sk-...` in `.env`.

If both are set, Ollama is used unless `LLM_BACKEND=openai`.

## Run (one command per line)

```bash
python -m src.pipeline.ingest
python -m src.pipeline.qa
python -m src.pipeline.train
python -m src.pipeline.report
python -m src.ai.commentary
```

Optional: `python -m src.pipeline.qa --llm-rules`

Working outputs: `outputs/` and `data/processed/` (see `data/README.md`). **SMARD URL patterns and filters** are in `docs/notes.md`.

## Run snapshot

After `report`, **`run_snapshot/`** is refreshed: `submission.csv`, `hold_out_meta.json`, `validation_summary.md`, figures, QA + metrics JSON, prompt-curve summary, and LLM logs if you generated them. See `run_snapshot/README.md`.

Longer design notes: `docs/notes.md`.
