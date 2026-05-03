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
│   └── run_pipeline.py
├── notebooks/
├── src/
├── tests/
├── requirements.txt
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

## End-to-end pipeline (single command)

From the repository root:

```bash
python scripts/run_pipeline.py
```

This runs, in order: preprocessing → K-fold training + evaluation → submission generation. Progress and per-stage timings are logged to the console; a full run log is written to `outputs/logs/pipeline_YYYYMMDD_HHMMSS.log` by default.

Useful options:

- `--export-csv` — during preprocessing, also write large `merged_train.csv` / `merged_test.csv` under `data/processed/`
- `--no-memory-opt` — load raw tables without memory downcasting
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

Automated execution should use `scripts/` (including `run_pipeline.py`), not notebook runs.

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
