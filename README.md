# Home Credit Default Risk

Production-grade machine learning system for the [Kaggle Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) competition. The project demonstrates end-to-end ML engineering: multi-table feature engineering, stratified ensemble training, calibration, explainability, automated reporting, and a hardened FastAPI inference service.

## Review Guide

Start here — all key reports are one click away. Metrics below match `artifacts/metrics/training_metrics.json` (single source of truth).

| Report | Markdown | HTML |
| --- | --- | --- |
| **Executive Summary** | [executive_summary.md](artifacts/reports/executive_summary.md) | [executive_summary.html](artifacts/reports/executive_summary.html) |
| **Model Comparison** | [model_comparison.md](artifacts/reports/model_comparison.md) | [model_comparison.html](artifacts/reports/model_comparison.html) |
| **SHAP Report** | [shap_report.md](artifacts/reports/shap_report.md) | [shap_report.html](artifacts/reports/shap_report.html) |
| **Model Card** | [model_card.md](artifacts/reports/model_card.md) | [model_card.html](artifacts/reports/model_card.html) |
| **Business Insights** | [business_insights.md](artifacts/reports/business_insights.md) | [business_insights.html](artifacts/reports/business_insights.html) |
| **Defense Preparation** | [defense_prep.md](artifacts/reports/defense_prep.md) | — |
| Training Summary | [training_summary.md](artifacts/reports/training_summary.md) | [training_summary.html](artifacts/reports/training_summary.html) |
| Calibration Report | [calibration_report.md](artifacts/reports/calibration_report.md) | [calibration_report.html](artifacts/reports/calibration_report.html) |
| Feature Ranking | [feature_report.md](artifacts/reports/feature_report.md) | [feature_report.html](artifacts/reports/feature_report.html) |
| Report index | [index.json](artifacts/reports/index.json) | — |
| Evaluation snapshot | [evaluation_report.json](artifacts/reports/evaluation_report.json) | — |

## How To Evaluate This Project In 10 Minutes

1. **Read Executive Summary** — [artifacts/reports/executive_summary.md](artifacts/reports/executive_summary.md) (business problem, key findings, recommended actions).
2. **Review Results** — OOF metrics table below and [model_comparison.md](artifacts/reports/model_comparison.md).
3. **Open SHAP Report** — [artifacts/reports/shap_report.md](artifacts/reports/shap_report.md) (global importance and risk direction).
4. **Review Model Comparison** — [artifacts/reports/model_comparison.md](artifacts/reports/model_comparison.md) (LightGBM vs CatBoost vs stacking vs calibration).
5. **Run Tests** — `make test` (95%+ coverage gate).
6. **Review API Example** — `make serve` then `curl -X POST http://127.0.0.1:8000/v1/predict -H "Content-Type: application/json" -d @examples/payloads/minimal_request.json`

Optional depth: [defense_prep.md](artifacts/reports/defense_prep.md) (25 trainer Q&A), [model_card.md](artifacts/reports/model_card.md) (intended use and limitations).

## Deployed Model

The API and submission pipeline serve the **calibrated stacking ensemble** — not raw LightGBM, not the weighted blend alone.

| Property | Value |
| --- | --- |
| **Architecture** | 5-fold stratified CV → per-fold LightGBM + CatBoost → fold-averaged predictions → logistic stacking meta-learner → **isotonic calibration** |
| **Artifact** | `artifacts/models/ensemble_model.pkl` |
| **OOF ROC-AUC (deployed)** | **0.7916410756948241** |
| **Calibration method** | Isotonic regression (selected over Platt scaling by lower Brier score) |
| **Features** | 723 |
| **Version** | 1.0.0 |

**Inference path:** For each fold, score with LightGBM and CatBoost using fold-specific preprocessors → average fold predictions → stack with logistic meta-learner → apply isotonic calibrator → return calibrated default probability.

Blend weights (`lgb: 1.0, cat: 0.0`) apply only to intermediate fold predictions before stacking; the deployed output is always the calibrated stack.

## Business Problem

Home Credit Group extends loans to financially excluded populations. The goal is to predict whether an applicant will default on a loan (`TARGET=1`) using application data and behavioral credit history from subsidiary tables. Accurate risk scoring supports underwriting decisions, portfolio monitoring, and loss prevention.

## Results

All values from `artifacts/metrics/training_metrics.json`:

| Model | OOF ROC-AUC |
| --- | --- |
| LightGBM | 0.790255159327977 |
| CatBoost | 0.7894775044564407 |
| Weighted Blend | 0.790255159327977 |
| Stacking | 0.7911954612578193 |
| **Calibrated (deployed)** | **0.7916410756948241** |

Full reports are under `artifacts/reports/` (regenerate with `make reports` after retraining).

## Architecture

```text
data/raw/  ──►  preprocess  ──►  data/processed/merged_{train,test}.pkl
                                        │
                                        ▼
                                 train (5-fold CV)
                                        │
                    ┌───────────────────┼───────────────────┐
                    ▼                   ▼                   ▼
           artifacts/models/   artifacts/metrics/   artifacts/reports/
           ensemble_model.pkl   training_metrics.json
                                        │
                                        ▼
                              FastAPI /v1/predict
```

### Module Layout

```text
src/
├── api/           # FastAPI app, schemas, prediction service
├── config/        # Typed YAML config loader + path constants
├── data/          # Loading, aggregation, preprocessing
├── features/      # Feature engineering, selection, schema
├── models/        # Train, predict, evaluate, calibration
├── reporting/     # Automated MD/HTML report generation
└── utils/         # Logging, serialization, memory optimization
```

## Setup

### Conda (recommended)

```bash
conda create -n abhishek-ml python=3.11 pip -y
conda activate abhishek-ml
make install          # core deps; never overwrites an existing LightGBM install
```

Verify you are using the env interpreter (not system Python):

```bash
which python pip   # should point to .../envs/abhishek-ml/bin/
```

### venv (alternative)

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
make install
```

Place Kaggle CSVs in `data/raw/` (see [data/README.md](data/README.md)). **Without raw data**, the pipeline still runs end-to-end using built-in synthetic sample tables (preprocess → train → API smoke tests auto-bootstrap a demo model).

### GPU training (optional)

Boosters (LightGBM, CatBoost, XGBoost) use GPU when `training.device` is `auto` or `cuda` and an NVIDIA GPU is present. Set in `configs/config.yaml`:

```yaml
training:
  device: auto          # auto | cpu | cuda
  gpu_device_id: 0
```

| `device` | Behavior |
| --- | --- |
| `auto` | CUDA/GPU for all boosters when `nvidia-smi` succeeds and LightGBM CUDA build is installed; otherwise CPU |
| `cpu` | Force CPU (CI default via `TRAINING_DEVICE=cpu`) |
| `cuda` | CUDA/GPU for all boosters (fails fast if LightGBM CUDA build is missing) |

Training logs the resolved backend at startup (`LIGHTGBM BACKEND: CUDA`, etc.).

**Dependencies:** `make install` installs everything in `requirements.txt` and **never reinstalls LightGBM** if it is already present — this preserves a CUDA-enabled LightGBM build. LightGBM is excluded from `requirements.txt` for the same reason.

| Environment | Command |
| --- | --- |
| GPU workstation (LightGBM already installed) | `make install` |
| CPU / CI | `make install-cpu` |
| First-time CUDA LightGBM install | `make install-cuda` |

`make install-cuda` builds LightGBM from source with `-DUSE_CUDA=ON` (requires NVIDIA CUDA toolkit). Prebuilt PyPI wheels are CPU-only and will not work for GPU training.

CatBoost and XGBoost GPU support is included in `requirements.txt`. Preprocessing, inference, and reporting remain on CPU.

## Workflow

```bash
make preprocess    # Aggregate + engineer features → data/processed/
make train         # 5-fold LGB+Cat ensemble, stacking, calibration, reports
make evaluate      # Write evaluation_report.json
make reports       # Regenerate reports from saved metrics (no retrain)
make submission    # Kaggle submission CSV
make serve         # Start API on :8000
make test          # 95%+ coverage test suite
```

## Feature Engineering

### Aggregation (per `SK_ID_CURR`)
- **Bureau / bureau_balance:** delinquency streaks, recency-weighted status, windowed trends (W3/W6/W12/W13), extended stats (mean/std/median/skew)
- **Previous applications:** approval/refusal rates, temporal patterns
- **Installments:** late payment flags, payment ratio, recency-weighted lateness
- **POS / credit card:** DPD frequency, utilization, balance volatility

### Engineered Features
- Capacity ratios (credit/income, annuity/credit, credit term)
- Interaction features (income×credit, employment×age, credit×bureau activity)
- Stability indicators (payment acceleration, utilization CV, credit velocity)
- Composite delinquency index

### Preprocessing
- Missing indicators for high-null columns
- Target encoding (fold-safe in CV) for high-cardinality categoricals
- One-hot encoding for low-cardinality categoricals

## Modeling

- **Base learners:** LightGBM + CatBoost (+ optional XGBoost)
- **Validation:** Stratified 5-fold CV with per-fold preprocessors
- **Ensemble:** OOF-optimized blend weights + logistic stacking meta-learner
- **Calibration:** Isotonic vs Platt scaling — auto-selected by Brier score
- **Feature selection:** Composite ranking (LightGBM gain, mutual information, SHAP)
- **Optional:** Optuna hyperparameter tuning (`enable_optuna: true` in config)

## API

```bash
make serve
curl http://127.0.0.1:8000/v1/health
curl -X POST http://127.0.0.1:8000/v1/predict \
  -H "Content-Type: application/json" \
  -d @examples/payloads/minimal_request.json
```

### Endpoints
| Endpoint | Description |
|----------|-------------|
| `GET /v1/health` | Model load status, bundle path |
| `POST /v1/predict` | Batch probability scoring |
| `GET /health`, `POST /predict` | Legacy aliases |

Responses include `model_version`, `schema_version`, `calibration_method`, and `threshold`.

**Note:** Best accuracy requires the full engineered feature vector. The API accepts sparse payloads and imputes missing columns, but production deployments should score preprocessed feature rows.

## Docker

Image and container name: **`abhishek-ml-project`**

```bash
make train
make docker-run     # builds abhishek-ml-project if missing, then runs on :8000
```

To rebuild the image explicitly:

```bash
make docker-build
```

Or with Compose:

```bash
docker compose up --build
```

## Testing

```bash
make test          # pytest with 95% coverage gate
make smoke         # auto-starts local API if needed, then health + predict
make docker-smoke  # frees port 8000, rebuilds image, starts Docker API, smoke test
make load-test     # Locust load test (UI at http://127.0.0.1:8089)
```

Load-test defaults in `configs/config.yaml` target **1000 users**. For high user counts (e.g. 20,000 in the Locust UI), think time is **auto-scaled** at test start to match single-node inference capacity (~8 RPS with 16 workers). Without this, the OS runs out of TCP connections and you see `Connection reset by peer`.

```bash
python scripts/run_load_test.py --headless
```

**Locust UI tips:** After clicking Start, check the terminal for `[load-test] Scaled think time` and `First-request stagger window`. For 20,000 users the stagger spreads the ramp burst over ~40 minutes. Use `make load-test USE_DOCKER=1` to test against Docker.

## Configuration

Key settings in `configs/config.yaml`:

```yaml
training:
  device: auto
  gpu_device_id: 0
  n_folds: 5
  enable_stacking: true
  enable_calibration: true
  enable_shap: true
  enable_optuna: false
  enable_xgboost: false
```

Environment override (optional): `TRAINING_DEVICE=cpu|cuda|auto` takes precedence over `training.device` in the YAML file.

## Artifacts

### What is committed vs generated

| Path | Status | Contents |
| --- | --- | --- |
| `artifacts/models/ensemble_model.pkl` | **Generated** (excluded by `.gitignore` if large) | Full model bundle — run `make train` |
| `artifacts/metrics/training_metrics.json` | Generated at train time | OOF AUCs, blend weights — **metric source of truth** |
| `artifacts/metrics/fold_metrics.csv` | Generated | Per-fold performance |
| `artifacts/reports/` | Generated (may be committed for reviewer convenience) | MD/HTML reports, `evaluation_report.json`, `index.json` |
| `artifacts/figures/` | Generated | ROC, PR, feature importance plots |
| `artifacts/submissions/` | Generated | Kaggle submission CSV |
| `data/raw/*` | **Excluded** (Kaggle terms + ~2.5 GB) | Download per [data/README.md](data/README.md), or omit for synthetic demo mode |
| `data/processed/*` | **Excluded** | Run `make preprocess` (auto-generated from synthetic data when raw CSVs are absent) |
| `catboost_info/` | **Excluded** | Legacy CatBoost log dir at repo root — not needed; training writes to `artifacts/tmp/catboost/` if required |

`.gitignore` excludes `artifacts/**` and `data/raw/*` / `data/processed/*` by default, keeping only `.gitkeep` placeholders. If reports and metrics are present in the repo, they were generated locally and committed for reviewer access — regenerate with `make train` + `make reports` for a fresh run.

## Notebooks

Exploratory notebooks live in `notebooks/` with status documented in [notebooks/README.md](notebooks/README.md). Production does not depend on them.

## Business Recommendations

1. Prioritize applicants with strong `EXT_SOURCE` scores and low installment late-payment rates.
2. Flag revolving utilization above 80% for manual review.
3. Monitor monthly drift in top-10 SHAP features.
4. Use calibrated probabilities for tiered lending limits.

## License & Data

Competition data is subject to Kaggle terms. Do not commit raw or processed datasets.
