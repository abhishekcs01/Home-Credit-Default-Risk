# Notebook Governance

Notebooks are for exploration and communication. **Production workflows run via `make` and `scripts/` — not notebooks.**

## Notebook Status

| Notebook | Status | Notes |
| --- | --- | --- |
| `01_eda.ipynb` | **Exploratory** | Early EDA walkthrough. Imports reference removed modules (`src.data_loading`) — use `src.data.load_data` if re-running, or treat as archival context. |
| `02_feature_engineering.ipynb` | **Exploratory** | Feature engineering prototype. Imports reference removed modules (`src.aggregation`, `src.feature_engineering`) — superseded by `src/features/` and `make preprocess`. |
| `03_model_training.ipynb` | **Exploratory** | Training prototype. Imports reference removed modules (`src.model_training`) — superseded by `src/models/train.py` and `make train`. |
| `04_submission.ipynb` | **Exploratory** | Submission prototype. Imports reference removed modules (`src.inference`) — superseded by `src/models/predict.py` and `make submission`. |
| `archive/legacy_monolith.ipynb` | **Archived** | Pre-refactor monolithic notebook. Historical reference only — do not run for evaluation. |

## Naming Convention

- `01_<topic>.ipynb` — primary exploratory sequence
- `NN_<topic>_draft.ipynb` — temporary work before cleanup
- `archive/` — obsolete or superseded notebooks

## Governance Rules

- Keep notebooks reproducible from repository code and `data/raw/` when possible.
- Do not duplicate production logic from `src/` or `scripts/`.
- Prefer importing stable modules instead of copying code cells.
- Remove heavy outputs before saving when practical.

## For Reviewers

To evaluate this project, use the production pipeline documented in the root `README.md` (`make preprocess`, `make train`, `make test`). Notebooks illustrate early exploration but are not required for reproduction.
