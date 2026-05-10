.PHONY: train api test load-test locust-ui clean smoke all pipeline

train:
	python scripts/preprocess_data.py
	python scripts/train_model.py

api:
	bash scripts/run_api.sh

test:
	python -m pytest -q

load-test:
	bash scripts/run_load_test.sh

locust-ui:
	python scripts/run_locust_ui.py

smoke:
	bash scripts/smoke_test_api.sh

clean:
	python -c "from pathlib import Path; import shutil; [shutil.rmtree(p, ignore_errors=True) for p in [Path('.pytest_cache'), Path('htmlcov')]]"
	python -c "from pathlib import Path; import shutil; [shutil.rmtree(p, ignore_errors=True) for p in Path('.').rglob('__pycache__')]"
	python -c "from pathlib import Path; [p.unlink() for p in Path('.').rglob('*.pyc')]"

all:
	python scripts/run_all.py

pipeline:
	python scripts/run_pipeline.py
