# AGENT.md

**Home Credit Default Risk — AI Agent Governance Manual**


| Field                | Value                                                                           |
| -------------------- | ------------------------------------------------------------------------------- |
| **Repository**       | Home Credit Default Risk ML System                                              |
| **Primary language** | Python 3.11                                                                     |
| **Status**           | Authoritative for all AI coding agents                                          |
| **Scope**            | `src/`, `scripts/`, `tests/`, `configs/`, `examples/`, `Dockerfile`, `Makefile` |


This document governs all AI coding agents operating in this repository (Cursor, Claude Code, Cline, Windsurf, Devin, OpenAI Codex, GPT Agents, and equivalents). Agents MUST read and comply with this file before making any change.

---

## 1. Purpose

The mission of an AI agent in this project is to **safely extend, fix, and maintain a production-grade machine learning system** for credit default risk prediction — without degrading model integrity, inference reliability, test coverage, or reviewer trust.

Agents are responsible for:

- Preserving the end-to-end pipeline: multi-table feature engineering → stratified ensemble training → calibration → explainability → automated reporting → hardened FastAPI inference.
- Upholding engineering standards equivalent to a professional ML platform team.
- Delivering minimal, verifiable, well-tested changes that respect existing architecture and conventions.
- Treating `artifacts/metrics/training_metrics.json` as the metric source of truth and `artifacts/models/ensemble_model.pkl` as the deployed model artifact.

Agents are **not** autonomous product owners. They implement scoped tasks; they do not redefine business objectives, model selection, or deployment policy without explicit human direction.

---

## 2. Core Responsibilities

Agents SHOULD perform the following when assigned work:

### Feature Implementation

- Implement features within the established module boundaries (`src/api`, `src/config`, `src/data`, `src/features`, `src/models`, `src/reporting`, `src/utils`).
- Wire new behavior through `configs/config.yaml` and `src/config/settings.py` when configuration is required.
- Preserve fold-safe cross-validation semantics for any preprocessing or encoding that touches training data.
- Maintain backward compatibility for API endpoints (`/v1/health`, `/v1/predict`, and legacy aliases `/health`, `/predict`).

### Bug Fixing

- Reproduce failures with tests or documented reproduction steps before changing code.
- Fix root causes, not symptoms.
- Add regression tests for every fixed defect unless the failure is purely environmental and cannot be tested deterministically.

### Refactoring

- Refactor only when it reduces complexity, improves clarity, or enables the assigned task.
- Preserve public interfaces, pickle compatibility (including legacy module remapping in `src/utils`), and artifact paths defined in `src/config/__init__.py`.
- Avoid drive-by refactors outside the task scope.

### Documentation

- Update `README.md`, `data/README.md`, `notebooks/README.md`, and relevant reports when behavior, setup, or artifacts change.
- Keep inline comments limited to non-obvious logic (fold-safe encoding, concurrency controls, calibration selection, synthetic-data fallback).

### Testing

- Maintain **≥ 95% line coverage** on `src/` per `pytest.ini`.
- Use synthetic fixtures in `tests/fixtures/`; tests MUST NOT depend on Kaggle CSVs or local `data/raw/` files.
- Force `TRAINING_DEVICE=cpu` in tests (see `tests/conftest.py`).

### Code Review Assistance

- Summarize diffs, risks, and verification steps suitable for human review.
- Flag changes that affect model outputs, API contracts, artifact formats, or metric reproducibility.

### Architecture Compliance

- Respect the pipeline topology documented in `README.md`:

```text
data/raw/ → preprocess → data/processed/ → train → artifacts/ → FastAPI /v1/predict
```

- Production code MUST NOT depend on `notebooks/`.
- The deployed inference path MUST remain: per-fold LightGBM + CatBoost → fold-averaged predictions → logistic stacking → isotonic calibration — unless explicitly directed to change the deployment architecture.

---

## 3. Non-Negotiable Rules

The following rules are **absolute**. Violation is grounds to reject the change.


| Rule                           | Requirement                                                                                                                              |
| ------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------- |
| **Preserve functionality**     | MUST NOT break existing preprocess, train, evaluate, report, API, Docker, or submission workflows.                                       |
| **Understand before deleting** | MUST NOT remove code, tests, config keys, or artifacts without tracing callers, imports, and downstream impact.                          |
| **No secrets in repo**         | MUST NOT commit `.env`, credentials, API keys, Kaggle tokens, or any secret-bearing file.                                                |
| **No dataset commits**         | MUST NOT commit `data/raw/`* or `data/processed/*` (Kaggle terms + size).                                                                |
| **No test bypass**             | MUST NOT skip tests, lower the coverage gate, disable assertions, or mark failing tests as expected without justification and approval.  |
| **Verify before assuming**     | MUST NOT assume file contents, function behavior, metric values, or API responses without reading code or running validation.            |
| **Minimal scope**              | MUST NOT modify unrelated files, reformat entire directories, or upgrade dependencies unless required by the task.                       |
| **No silent metric drift**     | MUST NOT change model training, feature engineering, or calibration logic without updating tests and documenting expected metric impact. |
| **No unauthorized commits**    | MUST NOT create git commits, push, or open PRs unless explicitly requested by the user.                                                  |
| **No destructive git ops**     | MUST NOT run `git push --force`, hard reset, or other irreversible git commands unless explicitly requested.                             |


---

## 4. Repository Awareness

Before writing code, agents MUST build context in this order:

### 4.1 Project Structure

```text
src/
├── api/           # FastAPI app, Pydantic schemas, prediction service
├── config/        # Typed YAML config loader + path constants
├── data/          # Loading, aggregation, preprocessing, synthetic fallback
├── features/      # Feature engineering, selection, schema
├── models/        # Train, predict, evaluate, calibration
├── reporting/     # Automated MD/HTML report generation
└── utils/         # Logging, serialization, memory optimization

scripts/           # CLI entry points (preprocess, train, API, smoke, load test)
tests/             # Unit, integration, API, load, and coverage-gap tests
configs/           # config.yaml — single runtime configuration file
examples/          # API payload examples
artifacts/         # Generated models, metrics, reports (mostly gitignored)
data/              # Raw (external) and processed (generated) datasets
notebooks/         # Exploratory only — not production dependencies
```

### 4.2 Read Before Write

Agents MUST:

1. Read `README.md` for architecture, workflow, and evaluation expectations.
2. Read `data/README.md` when touching data ingestion or synthetic mode.
3. Inspect the target module and its direct dependencies.
4. Search for existing patterns (logging via `get_logger`, paths via `src.config`, config via `load_config()`).
5. Review related tests in `tests/` before changing behavior.

### 4.3 Architectural Decisions (Do Not Violate Without Approval)


| Decision                                       | Rationale                                                          |
| ---------------------------------------------- | ------------------------------------------------------------------ |
| **Calibrated stacking ensemble is deployed**   | Business-facing probabilities; documented in README and model card |
| **Fold-safe target encoding in CV**            | Prevents leakage across folds                                      |
| **Synthetic data fallback**                    | Enables CI and reviewer walkthrough without Kaggle download        |
| **Typed frozen dataclasses for config**        | `AppConfig`, `ApiSettings`, `TrainingSettings` in `settings.py`    |
| **Path constants in `src/config/__init__.py`** | Single canonical artifact and data locations                       |
| **Pydantic strict schemas for API**            | Input validation at inference boundary                             |
| **LightGBM excluded from `requirements.txt`**  | Preserves CUDA-enabled local builds; use `make install`            |
| **Coverage gate at 95%**                       | Quality bar for production ML engineering                          |


### 4.4 Conventions to Follow

- `from __future__ import annotations` in new Python modules.
- Line length: **120** (Black, Ruff, isort).
- Logger namespace: `home_credit.<module>` via `src.utils.get_logger`.
- Random seed: **42** unless fold-specific seeding is required.
- Scripts invoke modules (`python -m src.models.train`), not ad-hoc duplicated logic.
- Environment override: `TRAINING_DEVICE=cpu|cuda|auto` takes precedence over YAML for training device.

---

## 5. Development Workflow

Agents MUST follow this sequence for every non-trivial task:

### Step 1 — Understand the Task

- Restate the goal, acceptance criteria, and constraints.
- Identify whether the change affects training, inference, API, data, reports, or infrastructure.
- Ask clarifying questions when requirements are ambiguous or mutually exclusive.

### Step 2 — Analyze the Codebase

- Locate entry points (`Makefile` targets, `scripts/*.py`, `src/**/*.py`).
- Trace data flow from input to output.
- Read existing tests that cover the affected behavior.

### Step 3 — Identify Affected Modules

- List files that will change and files that MUST NOT change.
- Note downstream consumers: API service, report generators, submission script, Docker image.

### Step 4 — Create an Implementation Plan

- Prefer the smallest correct diff.
- Specify test additions or updates.
- Call out risks: metric drift, pickle incompatibility, API contract changes, concurrency behavior.

### Step 5 — Implement Changes

- Edit only planned files.
- Match naming, typing, and error-handling style of surrounding code.
- Add configuration to `configs/config.yaml` + `settings.py` when introducing tunables.

### Step 6 — Run Validation

```bash
make lint          # ruff check src scripts tests
make test          # pytest with 95% coverage gate
```

When touching API or inference:

```bash
make smoke         # health + predict against local API
```

When touching Docker or container config:

```bash
make docker-smoke  # rebuild + containerized smoke test
```

When touching training or preprocessing (optional but recommended):

```bash
make preprocess
make train
make evaluate
```

### Step 7 — Verify Edge Cases

- Empty or sparse API payloads (imputation path).
- Missing raw data (synthetic fallback).
- CPU-only execution (`TRAINING_DEVICE=cpu`).
- Batch size and concurrency limits (`max_batch_size`, `max_concurrent_inferences`).
- Legacy endpoint aliases remain functional.

### Step 8 — Update Documentation

- Update `README.md` and section-specific docs if setup, commands, metrics, or API contracts change.
- Regenerate reports (`make reports`) only when metrics or model outputs change due to retraining.

---

## 6. Coding Standards

### Readability

- Prefer explicit names over abbreviations except established domain terms (`SK_ID_CURR`, `OOF`, `SHAP`).
- Keep functions focused; extract helpers only when reused or when complexity warrants it.
- Use type hints on public functions and dataclass fields.

### Maintainability

- Centralize paths in `src/config/__init__.py`; centralize runtime settings in `load_config()`.
- Avoid duplicating hyperparameters that already exist in `LGBM_BASE_PARAMS` or `configs/config.yaml`.
- Preserve pickle backward compatibility when changing module paths (update `legacy_module_map` in `src/utils`).

### Consistency

- Run formatters and linters before completion: **Ruff**, **Black**, **isort** (pre-commit hooks available).
- Import order: standard library → third party → `src` → `tests`.
- Match existing async/sync patterns in `src/api/main.py` (thread pool for inference, semaphores for backpressure).

### Performance

- Use `reduce_mem_usage()` for large DataFrames in data pipelines.
- Respect `inference_batch_size` and API concurrency limits; do not block the event loop with CPU-bound inference.
- Avoid unnecessary full retrains or SHAP recomputation in tests; use mocks and synthetic fixtures.

### Scalability

- Batch inference requests; avoid per-row model reloads.
- Load-test changes must respect `load_test` settings and auto-scaled think time (see `scripts/run_load_test.py`).

### Error Handling

- Raise specific exceptions with actionable messages in library code.
- Map service failures to structured API errors (`ErrorResponse` schema) with appropriate HTTP status codes.
- Fail fast on missing config (`FileNotFoundError` for missing `config.yaml`).

### Logging

- Use `get_logger("<module>")`; do not configure logging ad hoc.
- Log at INFO for lifecycle events; avoid logging PII or full request payloads at scale.
- Use `timer()` context manager for long-running pipeline stages.

### Type Safety

- mypy is configured in `pyproject.toml`; new code SHOULD be compatible with existing mypy settings.
- Pydantic models MUST use `strict=True` / `extra="forbid"` for API inputs where established.

---

## 7. Testing Requirements

### Unit Tests

- Place tests in `tests/test_<area>.py` mirroring `src/` structure.
- Use fixtures from `tests/conftest.py` and `tests/fixtures/`.
- Autouse fixtures MUST continue to enforce: deterministic seeds, CPU training device, no Kaggle data dependency.

### Integration Tests

- Cover pipeline stages: preprocessing → training artifacts → prediction → API responses.
- Validate artifact paths and JSON metric files when training logic changes.

### Regression Tests

- Any bug fix MUST include a test that fails before the fix and passes after.
- Any change to ensemble, calibration, or feature schema MUST update or add tests in `tests/test_model_training_advanced.py`, `tests/test_inference_advanced.py`, or `tests/test_api_advanced.py` as appropriate.

### Edge Cases (Required Coverage Areas)

- Synthetic data path when `data/raw/` is absent.
- Sparse API payloads vs full feature vectors.
- Invalid inputs: non-finite floats, out-of-range values, empty records, oversized batches.
- Config validation errors (`training.device` enum).
- Model bundle missing or corrupt (graceful API error).

### Validation Before Completion

- `make test` MUST pass with **≥ 95% coverage** (`--cov-fail-under=95`).
- Agents MUST NOT merge or declare completion with failing or skipped tests.
- New modules in `src/` MUST have corresponding test coverage.

---

## 8. Security Requirements

### Secrets Management

- MUST NOT hardcode credentials, tokens, or private URLs.
- MUST NOT commit `.env`, `*.secret`, or Kaggle API credentials.
- Secrets MUST be loaded from environment variables when needed; document required env vars in README.

### Input Validation

- All external input MUST pass through Pydantic schemas in `src/api/schemas.py`.
- Reject unknown fields (`extra="forbid"`), non-finite numerics, and oversized payloads (`max_request_body_bytes`, `max_batch_size`).
- Validate file paths from user input; do not allow arbitrary filesystem access.

### Authentication & Authorization

- The API is currently unauthenticated for local/demo use. Agents MUST NOT remove existing safety limits (concurrency caps, body size limits, timeouts) without explicit approval.
- If adding auth, MUST use established patterns and document breaking changes.

### Data Protection

- MUST NOT log raw applicant features at INFO level in production paths.
- MUST NOT commit competition datasets or derived processed files.
- Treat inference payloads as sensitive financial data in documentation and examples.

### Dependency Safety

- Pin or bound dependencies in `requirements.txt` when adding packages.
- MUST NOT add dependencies without justification.
- MUST NOT reinstall LightGBM in a way that overwrites a CUDA build (follow `make install` conventions).
- Review transitive risk for packages that execute network calls at import time.

---

## 9. Documentation Requirements

Agents MUST update documentation when changes affect behavior, setup, or operator workflows.


| Document                 | Update when                                                                 |
| ------------------------ | --------------------------------------------------------------------------- |
| `README.md`              | Commands, architecture, metrics, API, Docker, or setup change               |
| `data/README.md`         | Data layout, synthetic mode, or file naming changes                         |
| `notebooks/README.md`    | Notebook status or purpose changes                                          |
| `configs/config.yaml`    | New or renamed configuration keys (also update `settings.py`)               |
| `examples/payloads/`     | API request shape changes                                                   |
| `artifacts/reports/*.md` | Only via `make reports` / `make train` — do not hand-edit generated metrics |
| Inline comments          | Non-obvious algorithms, concurrency, or data-leakage prevention             |


### API Documentation

- Endpoint table in README MUST stay synchronized with `src/api/main.py` routes.
- Document response fields: `model_version`, `schema_version`, `calibration_method`, `threshold`.

### Architecture Documentation

- Significant structural changes MUST update the ASCII pipeline diagram in README.
- New modules MUST be listed in the Module Layout section.

### Metric Documentation

- Performance claims MUST cite `artifacts/metrics/training_metrics.json`.
- Do not invent or round metric values in prose without verifying the JSON source.

---

## 10. Communication Protocol

Agents MUST communicate in a professional, concise, evidence-based manner.

### Required Practices

- **Be concise** — Lead with outcome, then supporting detail.
- **Explain reasoning** — State why an approach was chosen over alternatives when non-obvious.
- **Highlight risks** — Model drift, API breakage, pickle incompatibility, coverage drops, performance regression.
- **Surface assumptions** — Explicitly list anything not verified (e.g., "assumed retrain not required").
- **Ask questions** — When requirements are ambiguous, conflicting, or missing acceptance criteria.

### Response Structure for Completed Work

1. Summary of what changed and why.
2. Files touched (grouped by module).
3. Validation performed (`make test`, `make lint`, smoke tests).
4. Risks, limitations, or follow-ups.

### Escalation Guidelines

Escalate to the human operator (ask before proceeding) when:


| Condition                                             | Action                                                             |
| ----------------------------------------------------- | ------------------------------------------------------------------ |
| Task requires changing deployed model architecture    | Stop; confirm expected metric and business impact                  |
| Task requires committing datasets or large binaries   | Stop; violates Kaggle terms and `.gitignore`                       |
| Task requires lowering test coverage below 95%        | Stop; propose alternative test strategy                            |
| Task requires force-push, hard reset, or hook bypass  | Stop; wait for explicit approval                                   |
| Task requires new production secrets or auth provider | Stop; provide design options                                       |
| Tests fail after multiple targeted fix attempts       | Report failures, root-cause hypothesis, and proposed next steps    |
| Change affects public API contract                    | Document breaking changes and migration path before implementation |
| Pickle format or model bundle schema must change      | Plan backward compatibility or migration script                    |


---

## 11. Change Safety Policy

### Minimal Blast Radius

- One logical change per PR or commit series.
- Touch the fewest files required to satisfy the task.
- Prefer extending existing functions over creating parallel implementations.

### Backward Compatibility

- Maintain legacy API routes (`/health`, `/predict`).
- Maintain pickle module remapping for older artifacts.
- Do not rename artifact paths without migration support.
- Do not remove config keys without deprecation period unless explicitly instructed.

### Incremental Changes

- Land refactors separately from behavior changes when possible.
- Feature-flag risky behavior via `configs/config.yaml` toggles.
- Use synthetic/demo training paths for validation before full Kaggle retrains.

### Rollback Awareness

- Changes MUST be reversible via git revert without manual artifact surgery.
- Document any manual steps required to restore service (e.g., `make train` to regenerate `ensemble_model.pkl`).
- Do not delete artifacts or models that production/Docker mounts depend on without regeneration instructions.

---

## 12. Prohibited Behaviors

Agents MUST NOT:


| Prohibited behavior                                       | Why                                                                          |
| --------------------------------------------------------- | ---------------------------------------------------------------------------- |
| **Hallucinate functionality**                             | Claim modules, endpoints, or reports exist without verifying                 |
| **Invent APIs**                                           | Introduce undocumented public functions or routes not requested              |
| **Ignore failing tests**                                  | Erode the 95% coverage quality gate                                          |
| **Make silent breaking changes**                          | Alter response schemas, metric formats, or model outputs without notice      |
| **Modify production configuration without justification** | Change `configs/config.yaml` defaults that affect deployed behavior casually |
| **Commit without request**                                | Violates user workflow and review expectations                               |
| **Train on committed secrets/data**                       | Legal and security violation                                                 |
| **Disable lint/type checks**                              | Masks defects; use `# noqa` sparingly with justification                     |
| **Broad dependency upgrades**                             | Unrelated version bumps introduce unpredictable risk                         |
| **Use notebooks as production entry points**              | Violates architecture; notebooks are exploratory                             |
| **Hand-edit generated reports/metrics**                   | Use `make train` / `make reports` for consistency                            |
| **Bypass API backpressure controls**                      | Removing semaphores, limits, or timeouts risks OOM and connection exhaustion |


---

## 13. Completion Checklist

Before marking any task complete, agents MUST verify every item:

- [ ] **Requirements satisfied** — All stated acceptance criteria met.
- [ ] **Scope minimal** — No unrelated files modified.
- [ ] **Lint clean** — `make lint` passes (or equivalent `ruff check`).
- [ ] **Tests pass** — `make test` passes with ≥ 95% coverage on `src/`.
- [ ] **API validated** — `make smoke` passes if `src/api/` or inference path changed.
- [ ] **Docker validated** — `make docker-smoke` passes if `Dockerfile`, `docker-compose.yml`, or container runtime changed.
- [ ] **Config synchronized** — `configs/config.yaml` and `src/config/settings.py` updated together if needed.
- [ ] **Documentation updated** — README and relevant docs reflect new behavior.
- [ ] **No secrets committed** — No `.env`, credentials, or raw/processed data staged.
- [ ] **No security regressions** — Input validation, size limits, and concurrency guards intact.
- [ ] **No unnecessary complexity** — Simpler solution chosen when equally correct.
- [ ] **Metrics integrity** — Any performance claims match `training_metrics.json` or newly generated artifacts.
- [ ] **Assumptions disclosed** — Human reviewer informed of anything not fully verified.

---

## Quick Reference


| Command             | Purpose                                            |
| ------------------- | -------------------------------------------------- |
| `make install`      | Install dependencies (preserves existing LightGBM) |
| `make install-cpu`  | CPU-only LightGBM install for CI                   |
| `make preprocess`   | Feature engineering → `data/processed/`            |
| `make train`        | Train ensemble, write artifacts                    |
| `make evaluate`     | Write `evaluation_report.json`                     |
| `make reports`      | Regenerate reports from saved metrics              |
| `make serve`        | Start FastAPI on port 8000                         |
| `make test`         | Full test suite + 95% coverage gate                |
| `make lint`         | Ruff static analysis                               |
| `make smoke`        | API health + predict smoke test                    |
| `make docker-smoke` | Docker build + smoke test                          |
| `make load-test`    | Locust load test                                   |



| Canonical paths        | Location                                  |
| ---------------------- | ----------------------------------------- |
| Deployed model         | `artifacts/models/ensemble_model.pkl`     |
| Metric source of truth | `artifacts/metrics/training_metrics.json` |
| Runtime config         | `configs/config.yaml`                     |
| API example payload    | `examples/payloads/minimal_request.json`  |


---

*This document is the authoritative governance layer for AI agents in this repository. When in doubt, prefer safety, verifiability, and minimal scope over speed.*