# Home Credit Default Risk Project

Production-style machine learning repository for Home Credit Default Risk with modular code, reproducible pipelines, and script-first execution.

## Project Architecture

This project is organized into layers:

- `config.py` (repo root): single source of truth for paths and hyperparameters
- `src/`: reusable ML logic (data loading, preprocessing, feature engineering, aggregation, training, evaluation, inference)
- `scripts/`: command-line entry points; production runs use these, not notebooks
- `notebooks/`: EDA, explanation, visualization, and experiments only
- `data/raw/`: immutable source datasets
- `data/processed/`: generated model-ready artifacts (`merged_train.pkl`, `merged_test.pkl`)
- `models/`: serialized trained model bundle
- `outputs/`: run artifacts — `submissions/`, `metrics/`, `logs/`
- `reports/figures/`: ROC and feature-importance plots for documentation

Runtime paths are configured centrally in **`config.py`** at the repository root. The `src/config` module re-exports that configuration so existing `from src import config` imports keep working.

## Repository Structure

```text
home-credit-risk-project/
├── config.py
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── outputs/
│   ├── submissions/
│   ├── metrics/
│   └── logs/
├── reports/
│   ├── figures/
│   └── final_report.md
├── scripts/
│   ├── preprocess_data.py
│   ├── train_model.py
│   ├── generate_submission.py
│   ├── run_all.py
│   └── run_pipeline.py
├── notebooks/
├── src/
├── tests/
├── requirements.txt
├── run_all.bat
├── run_locust_ui.bat
└── README.md
```

## Setup

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
# source .venv/bin/activate
pip install -r requirements.txt
```

## One command: pipeline + API smoke + quick load test

From the repository root (after `pip install -r requirements.txt`):

```bash
python scripts/run_all.py
```

On Windows, **`make` is usually not installed**. Use Python directly (above), or double‑click / run:

```bat
run_all.bat
```

On macOS or Linux with GNU Make: `make all` runs the same thing.

This runs in order:

1. Preprocess → train → submission (same stages as `run_pipeline.py`; merged pickles are reused unless you pass `--force-recompute`)
2. Starts a temporary Uvicorn server bound to `127.0.0.1` (port from `configs/config.yaml`)
3. Waits for `GET /health`, then `POST /predict` using `examples/payloads/minimal_request.json`
4. Runs a short headless Locust session (default: 5 users, 30 seconds)

Useful flags:

- `--pipeline-only` — ML pipeline only (no API or Locust)
- `--skip-load-test` — run smoke `/predict` only (skip Locust)
- `--force-recompute` — rebuild `merged_train.pkl` / `merged_test.pkl` from raw CSVs
- `--api-startup-timeout 600` — allow more time while the model loads on slower machines
- `--load-users`, `--load-spawn-rate`, `--load-run-time` — tune the quick Locust phase

Logs default to `outputs/logs/run_all_YYYYMMDD_HHMMSS.log`.

## End-to-end ML pipeline only

From the repository root:

```bash
python scripts/run_pipeline.py
```

With GNU Make: `make pipeline` runs the same thing (optional on Windows).

This runs preprocessing → K-fold training + evaluation → submission generation. Progress and per-stage timings are logged to the console; the default log file is `outputs/logs/pipeline_YYYYMMDD_HHMMSS.log`.

Useful options:

- `--export-csv` — during preprocessing, also write large `merged_train.csv` / `merged_test.csv` under `data/processed/`
- `--no-memory-opt` — load raw tables without memory downcasting
- `--force-recompute` — ignore cached merged pickles and recompute preprocessing
- `--log-file PATH` — override the default pipeline log path
- `--no-timestamp-copy` — skip writing an extra timestamped submission file

## Step-by-step scripts (independent runs)

Each stage can be run on its own:

1. **Preprocess + feature engineer**

```bash
python scripts/preprocess_data.py
```

Creates `data/processed/merged_train.pkl` and `merged_test.pkl`. Optional: `--export-csv`.

2. **Train + evaluate**

```bash
python scripts/train_model.py
```

Creates:

- `models/lightgbm_model.pkl`
- `reports/figures/roc_curve.png`
- `reports/figures/feature_importance.png`
- `outputs/metrics/training_metrics.json`
- `outputs/metrics/feature_importance_top15.csv`

3. **Generate submission**

```bash
python scripts/generate_submission.py
```

Writes `outputs/submissions/submission.csv` and a timestamped copy `outputs/submissions/submission_YYYYMMDD_HHMMSS.csv`. Override destination with `--output PATH`; disable the timestamped copy with `--no-timestamp-copy`.

## Notebook purposes

Notebooks are for exploration and reporting only:

- `notebooks/01_eda.ipynb` — dataset overview, target distribution, missingness, correlations
- `notebooks/02_feature_engineering.ipynb` — aggregation + feature engineering walkthrough
- `notebooks/03_model_training.ipynb` — training and validation narrative
- `notebooks/04_submission.ipynb` — inference walkthrough

Automated execution should use `scripts/` (including `run_all.py` / `run_pipeline.py`), not notebook runs.

## Evaluation

Primary metric: **ROC-AUC**

Reported metrics include:

- Holdout ROC-AUC (LightGBM)
- Holdout ROC-AUC + CV summary (logistic baseline)
- OOF ROC-AUC (K-fold LightGBM ensemble)

## Testing

```bash
python -m pytest -q
```

Test modules validate aggregation, feature engineering, and model training on synthetic data.

## Inference API (FastAPI)

The repository includes a deployable inference service under `app/`:

- `app/main.py` — API app factory, startup model loading, health and predict routes
- `app/predict.py` — lightweight inference service wrapper around `src.inference.EnsembleInferenceEngine`
- `app/schemas.py` — Pydantic request/response schemas and validation contracts

### Start the API

```bash
make api
```

or

```bash
bash scripts/run_api.sh
```

By default:

- base URL: `http://127.0.0.1:8000`
- health endpoint: `GET /health`
- prediction endpoint: `POST /predict`

The API loads the trained model bundle on startup (configurable through `configs/config.yaml`).

### Predict request and response

Request body schema:

```json
{
  "records": [
    {
      "SK_ID_CURR": 100001,
      "AMT_INCOME_TOTAL": 135000.0,
      "AMT_CREDIT": 450000.0
    }
  ]
}
```

Response schema:

```json
{
  "predictions": [
    {
      "SK_ID_CURR": 100001,
      "TARGET": 0.132
    }
  ],
  "model_version": "lightgbm_ensemble",
  "request_count": 1
}
```

## Example Payloads

Ready-to-run payloads are provided in `examples/payloads/`:

- `minimal_request.json`
- `full_application_request.json`
- `batch_request.json`

You can use them directly:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  --data "@examples/payloads/minimal_request.json"
```

## API Validation and Error Handling

Pydantic schemas validate:

- required fields (`SK_ID_CURR`)
- datatypes (for example, integer IDs)
- malformed JSON request bodies

Invalid requests return structured error responses with an `error_type` field.

## Load Testing (Locust)

Locust scenarios are defined in `tests/load/locustfile.py` and simulate:

- repeated single-record requests
- richer full-feature requests
- batch prediction requests

Run headless load test:

```bash
make load-test
```

or

```bash
bash scripts/run_load_test.sh
```

### Locust web UI (interactive)

1. Start the API in **another** terminal:

```powershell
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

2. Start Locust with the browser UI (Windows):

```powershell
.\run_locust_ui.bat
```

or cross-platform:

```bash
python scripts/run_locust_ui.py
```

3. Open **http://127.0.0.1:8089** (default). Set users and spawn rate in the UI, then **Start swarming**.

Optional: `--target-host http://127.0.0.1:8000`, `--web-port 8089`, `--no-browser`. Defaults match `configs/config.yaml` (`load_test.locust_web_*`).

Useful environment overrides:

- `HOME_CREDIT_LOAD_HOST`
- `HOME_CREDIT_LOAD_USERS`
- `HOME_CREDIT_LOAD_SPAWN_RATE`
- `HOME_CREDIT_LOAD_RUN_TIME`
- `HOME_CREDIT_LOAD_MIN_WAIT_SECONDS`
- `HOME_CREDIT_LOAD_MAX_WAIT_SECONDS`
- `HOME_CREDIT_LOCUST_WEB_HOST` / `HOME_CREDIT_LOCUST_WEB_PORT` — Locust UI bind (for `run_locust_ui.py`)

## API Tests

API-focused tests are available in:

- `tests/test_api.py`
- `tests/test_api_payloads.py`

They cover:

- valid and invalid payload validation
- malformed JSON handling
- batch inference behavior
- response schema contract
- health endpoint checks

## Automation Scripts

Operational scripts under `scripts/`:

- `run_api.sh` — start FastAPI server with config defaults
- `run_load_test.sh` — execute Locust in headless mode
- `run_locust_ui.py` — Locust **web UI** (with `run_locust_ui.bat` at repo root on Windows)
- `smoke_test_api.sh` — run health + prediction smoke checks

## Docker Usage

Build and run API container:

```bash
docker compose up --build api
```

Run API plus headless load test:

```bash
docker compose up --build
```

> Ensure a trained model exists at `models/lightgbm_model.pkl` before running inference containers.

## Makefile Commands

Makefile targets assume **GNU Make** (common on macOS/Linux; on Windows install [Chocolatey `make`](https://community.chocolatey.org/packages/make) or use `python scripts/...` / `run_all.bat` instead).

- `make all` — preprocess → train → submission → temporary API smoke → short Locust (`scripts/run_all.py`)
- `make pipeline` — preprocess → train → submission only (`scripts/run_pipeline.py`)
- `make train` — preprocess then train (does not run submission)
- `make api` — start FastAPI service
- `make test` — run test suite
- `make load-test` — run Locust load test (headless)
- `make locust-ui` — open Locust web UI (`python scripts/run_locust_ui.py`)
- `make smoke` — smoke test against live API
- `make clean` — remove local test/cache artifacts

## Runtime Configuration

Operational settings are centralized in `configs/config.yaml`, including:

- API host/port/startup model loading behavior
- model artifact path for inference service
- structured logging preferences
- load test defaults (host/users/spawn-rate/runtime/wait times)
