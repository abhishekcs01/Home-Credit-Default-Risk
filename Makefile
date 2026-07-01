PYTHON ?= python
DOCKER_IMAGE ?= abhishek-ml-project
DOCKER_CONTAINER ?= abhishek-ml-project
PYTEST ?= $(PYTHON) -m pytest

ifeq ($(OS),Windows_NT)
COV_ENV = set COVERAGE_FILE=.coverage.testcov &&
else
COV_ENV = COVERAGE_FILE=.coverage.testcov
endif

.PHONY: install lint test train evaluate serve preprocess submission smoke load-test all docker-build docker-run docker-run-bg docker-smoke docker-up docker-down clean reports

install:
	bash scripts/install_dependencies.sh

install-cpu:
	INSTALL_LIGHTGBM=cpu bash scripts/install_dependencies.sh

install-cuda:
	INSTALL_LIGHTGBM=cuda bash scripts/install_dependencies.sh

lint:
	$(PYTHON) -m ruff check src scripts tests

preprocess:
	$(PYTHON) scripts/preprocess_data.py

train:
	$(PYTHON) -m src.models.train

evaluate:
	$(PYTHON) -m src.models.evaluate

submission:
	$(PYTHON) scripts/generate_submission.py

reports:
	PYTHONPATH=. $(PYTHON) scripts/regenerate_reports.py

serve:
	$(PYTHON) scripts/run_api.py

smoke:
	$(PYTHON) scripts/smoke_test_api.py

test:
	$(PYTEST) tests

load-test:
	$(PYTHON) scripts/run_load_test.py $(if $(USE_DOCKER),--use-docker,)

all:
	$(PYTHON) scripts/run_all.py

docker-build:
	docker build -t $(DOCKER_IMAGE) .

docker-run:
	@docker rm -f $(DOCKER_CONTAINER) 2>/dev/null || true
	@docker image inspect $(DOCKER_IMAGE) >/dev/null 2>&1 || $(MAKE) docker-build
	@$(PYTHON) -c "from scripts.docker_smoke import free_port; free_port(8000)"
	docker run --rm -p 8000:8000 \
		--name $(DOCKER_CONTAINER) \
		-v "$(CURDIR)/artifacts:/app/artifacts:ro" \
		$(DOCKER_IMAGE)

docker-run-bg:
	@docker rm -f $(DOCKER_CONTAINER) 2>/dev/null || true
	@docker image inspect $(DOCKER_IMAGE) >/dev/null 2>&1 || $(MAKE) docker-build
	@$(PYTHON) -c "from scripts.docker_smoke import free_port; free_port(8000)"
	docker run -d --rm -p 8000:8000 \
		--name $(DOCKER_CONTAINER) \
		-v "$(CURDIR)/artifacts:/app/artifacts:ro" \
		$(DOCKER_IMAGE)

docker-smoke:
	$(PYTHON) scripts/docker_smoke.py

docker-up:
	docker compose up --build

docker-down:
	docker compose down

clean:
	$(PYTHON) -c "from pathlib import Path; import shutil; [shutil.rmtree(p, ignore_errors=True) for p in (Path('.pytest_cache'), Path('.mypy_cache'), Path('.ruff_cache'), Path('htmlcov'), Path('catboost_info'), Path('artifacts/tmp'))]"
	$(PYTHON) -c "from pathlib import Path; [p.unlink() for p in Path('.').glob('.coverage*') if p.is_file()]"

# Backward-compatible aliases
api: serve
test-cov: test
