# Home Credit Default Risk

Modular, script-driven ML for borrower default risk: table aggregation and feature engineering, **K-fold LightGBM + optional CatBoost** blend, **fold-fitted preprocessors**, **FastAPI** inference, and **Locust (web UI)** load testing. Training and batch scoring write to `outputs/` and `reports/`; the serialized **ensemble bundle** is the single artifact for serving.

---

## Architecture overview

| Layer | Role |
|--------|------|
| **`config.py`** | Canonical paths, hyperparameters, `MODEL_BUNDLE_PATH`. |
| **`configs/config.yaml`** | API bind, logging, model path override, Locust defaults (loaded by `src.runtime_config`). |
| **`src/`** | Data loading, preprocessing, aggregation, feature engineering, training, evaluation, **`src.inference`** ensemble prediction. |
| **`app/`** | FastAPI app: validation, health, `/predict` wrapping **`EnsembleInferenceEngine`**. |
| **`scripts/`** | Stage scripts + **`pipeline/`** (orchestration) + **`load/`** (Locust UI only). |
| **`tests/`** | Unit/integration tests; **`tests/load/locustfile.py`** defines Locust scenarios. |

**Training flow:** raw CSVs (`data/raw/`) → preprocess → `merged_*.pkl` → K-fold train → **`models/ensemble_model.pkl`** + metrics/plots under `outputs/` and `reports/figures/`.

**Inference flow:** JSON records → Pandas → per-fold preprocessor transform → LightGBM / CatBoost blend → mean across folds → optional probability sanitization → JSON response (`TARGET` in `[0, 1]`).

---

## Repository layout

```text
├── config.py                 # Paths & modeling constants (single source for paths)
├── configs/config.yaml       # API / logging / model path / load-test defaults
├── docs/                     # Ancillary documentation (see docs/README.md)
├── src/
│   ├── config.py             # Re-exports root config.py
│   ├── runtime_config.py     # Loads configs/config.yaml + env
│   └── …                     # ML pipeline & inference
├── app/                      # FastAPI service
├── scripts/
│   ├── pipeline/
│   │   ├── run_pipeline.py   # preprocess → train → submission
│   │   └── run_all.py        # pipeline + temp API + smoke test
│   ├── load/
│   │   └── run_locust.py     # Locust web UI (interactive load testing)
│   ├── preprocess_data.py
│   ├── train_model.py
│   └── generate_submission.py
├── tests/
├── notebooks/                # EDA / walkthroughs
├── notebooks/archive/        # historical notebooks (e.g. legacy_monolith.ipynb)
├── examples/payloads/        # Sample API JSON bodies
├── models/                   # ensemble_model.pkl (gitignored by default)
├── outputs/                  # submissions/, metrics/, logs/
├── reports/                  # figures/, final_report.md
├── docker-compose.yml
├── Dockerfile
├── Makefile
└── README.md
```

---

## Configuration (what lives where)

- **`config.py` (root)** — Filesystem layout (`DATA_DIR`, `MODEL_BUNDLE_PATH`, etc.), LightGBM defaults, blend weights, fold count. **Training scripts save the bundle to `MODEL_BUNDLE_PATH`.**
- **`src/config.py`** — Imports root `config.py` so code can use `from src import config`.
- **`configs/config.yaml`** — Operational knobs for **inference** (API host/port/workers, bundle path string, logging, Locust-related defaults). Values may be overridden by environment variables (see `src/runtime_config.py`).
- **`src/runtime_config.py`** — Parses YAML + env into typed settings consumed by **`app/main.py`** and Locust helpers.

No Hydra or duplicate constants beyond the deliberate YAML override of `model.bundle_path` (defaults align with `MODEL_BUNDLE_PATH`).

---

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Unix:    source .venv/bin/activate
pip install -r requirements.txt
```

Place competition CSVs under `data/raw/` before training.

---

## Training (pipeline)

End-to-end **preprocess → train → submission**:

```bash
python scripts/pipeline/run_pipeline.py
# or: make pipeline
```

**Artifacts:** `models/ensemble_model.pkl` (fold ensemble bundle), `reports/figures/*.png`, `outputs/metrics/*`, `outputs/submissions/submission.csv`.

Stages may also be run individually:

```bash
python scripts/preprocess_data.py
python scripts/train_model.py
python scripts/generate_submission.py
```

**Migrating from an older bundle name:** if you still have `models/lightgbm_model.pkl`, rename it to **`models/ensemble_model.pkl`** or retrain so the filename matches `config.py` / `configs/config.yaml`.

---

## Full automation (pipeline + smoke API)

```bash
python scripts/pipeline/run_all.py
# or: make all
```

Runs the ML pipeline, starts a **temporary** Uvicorn on `127.0.0.1` (port from config), waits for `/health`, then smoke-tests `POST /predict`. **Load testing is manual** — start Locust separately (below).

---

## Inference API (FastAPI)

```bash
make api
# or: bash scripts/run_api.sh
```

- **`GET /health`** — liveness + bundle readiness hints  
- **`POST /predict`** — body `{ "records": [ { "SK_ID_CURR": …, … } ] }`  

Example payloads: `examples/payloads/`.

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  --data "@examples/payloads/minimal_request.json"
```

---

## Load testing (Locust web UI only)

1. Run the API (see above).  
2. From the repo root:

```bash
python scripts/load/run_locust.py
# or: make locust
```

3. Open **http://127.0.0.1:8089** (default UI port), set users/spawn rate, **Start swarming**.

Scenarios: `tests/load/locustfile.py`. Override API URL with `--target-host` or `HOME_CREDIT_LOAD_HOST`.

---

## Docker

```bash
docker compose up --build api
```

Mount `./models` read-only; ensure **`ensemble_model.pkl`** exists before inference. Run Locust **on the host** against `http://127.0.0.1:8000` (or map ports as needed).

---

## Testing

```bash
python -m pytest -q
# or: make test
```

---

## Makefile reference

| Target | Action |
|--------|--------|
| `make pipeline` | `scripts/pipeline/run_pipeline.py` |
| `make all` | `scripts/pipeline/run_all.py` |
| `make api` | `scripts/run_api.sh` |
| `make test` | pytest |
| `make locust` | Locust web UI |
| `make smoke` | `scripts/smoke_test_api.sh` |
| `make clean` | `.pytest_cache`, `htmlcov`, `__pycache__`, `*.pyc` |

---

## Notebooks

Exploration and reporting only; **`scripts/`** is the source of truth for automation. Active notebooks live under `notebooks/`; **`notebooks/archive/`** holds legacy materials (e.g. `legacy_monolith.ipynb`).

---

## Evaluation & outputs

- **Primary metric:** ROC-AUC (with supporting PR curves and feature-importance artifacts).  
- **Reports:** `reports/figures/`, `reports/final_report.md`.  
- **Metrics:** `outputs/metrics/`.  
- **Submissions:** `outputs/submissions/`.

---

## API validation

Pydantic enforces request shape and types; errors return JSON with `error_type`. Response probabilities are validated into `[0, 1]` for stable JSON.
