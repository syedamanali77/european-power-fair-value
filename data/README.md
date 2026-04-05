# Data layout

- **`processed/`** — built locally as `panel.parquet` by `python -m src.pipeline.ingest`. Not stored in git (large binary; regenerate anytime).
- **`raw/`** — optional scratch space if you extend the pipeline.

Hourly series come from **SMARD** (public HTTP). Endpoint shape and filter IDs are documented in `docs/notes.md` and `config/market.yaml`.

Timestamps are stored timezone-aware (**`Europe/Berlin`**) after conversion from SMARD’s Unix milliseconds (UTC).
