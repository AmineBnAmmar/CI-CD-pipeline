
VENV = venv
PYTHON = $(VENV)/bin/python3
PIP = $(VENV)/bin/pip

.DEFAULT_GOAL := all

all: install-deps check-code prepare-data train-model run-tests mlflow-ui run-pipeline  test-api

# 1. Setup virtual environment and install dependencies
venv:
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip

install-deps: venv
	$(PIP) install -r requierements.txt

# 2. Code verification targets
format: install-deps
	$(VENV)/bin/black model_pipeline.py main.py

lint: install-deps
	$(VENV)/bin/pylint --fail-under=5.0 model_pipeline.py main.py

security-check: install-deps
	$(VENV)/bin/bandit -r model_pipeline.py main.py

check-code: format lint security-check

# 3. Prepare data
prepare-data: install-deps
	$(PYTHON) -c "from model_pipeline import prepare_data; prepare_data()"

# 4. Train model
train-model: prepare-data
	$(PYTHON) -c "from model_pipeline import prepare_data, train_model; X_train, y_train, _, _, _, _ = prepare_data(); model = train_model(X_train, y_train)"

# 5. Run tests
run-tests: install-deps
	$(PYTHON) -m pytest tests/test_model.py -v

# 6. MLflow-related targets
mlflow-ui: install-deps
	@echo "\n=== Starting MLflow UI ==="
	mlflow ui --host localhost --port 5000 & \

run-pipeline: install-deps
	@echo "\n=== Running Full Pipeline ==="
	$(PYTHON) main.py

# Add to your existing Makefile
test-api: install-deps
	@echo "\n=== Testing API Endpoints ==="
	$(PYTHON) -m pytest test_api.py -v

# Cleanup
clean:
	rm -rf __pycache__
	rm -rf $(VENV)
	rm -f *.pkl

.PHONY: all venv install-deps check-code format lint security-check prepare-data train-model run-tests test-api clean mlflow-ui run-pipeline
