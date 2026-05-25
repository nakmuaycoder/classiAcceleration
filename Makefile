# --- Variables ---
PYTHON := uv run python
SRC_DIR := ml/src
ARDUINO_DIR := firmware/FitnessTracker

# --- Default Goal ---
.PHONY: help
help:
	@echo "TinyML Fitness Tracker - Modernized Workflow"
	@echo "-------------------------------------------"
	@echo "  make install     Setup environment and dependencies"
	@echo "  make data        Preprocess raw logs into clean CSVs"
	@echo "  make train       Train a single model with current config"
	@echo "  make nas         Launch Architecture Search (Optuna NAS)"
	@echo "  make tensorboard Start TensorBoard visualization"
	@echo "  make test        Execute all unit tests (pytest)"
	@echo "  make format      Format and lint all code (Ruff)"
	@echo "  make clean       Cleanup cached artifacts"

# --- Setup ---
.PHONY: install
install:
	@echo "Installing dependencies using uv..."
	uv sync
	@echo "Environment ready."

# --- Data Preprocessing ---
.PHONY: data
data:
	@echo "Preprocessing raw logs..."
	export PYTHONPATH=$${PYTHONPATH}:. && $(PYTHON) ml/src/preprocess.py
	@echo "Raw data processed in data/clean/"

# --- Clean up ---
.PHONY: clean
clean:
	@echo "Cleaning up..."
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.py[co]" -delete
	rm -rf .uv
	@echo "Done."

# --- Model Workflow ---
.PHONY: train
train:
	export PYTHONPATH=$${PYTHONPATH}:. && $(PYTHON) $(SRC_DIR)/train.py

# Automated NAS (Neural Architecture Search) with Optuna Sweeper
.PHONY: nas
nas:
	export PYTHONPATH=$${PYTHONPATH}:. && $(PYTHON) $(SRC_DIR)/train.py --multirun

.PHONY: nas-1d
nas-1d:
	export PYTHONPATH=$${PYTHONPATH}:. && $(PYTHON) $(SRC_DIR)/train.py --multirun \
		data.use_norm=true data.augment=false

.PHONY: nas-3d
nas-3d: nas-3d-aug

.PHONY: nas-3d-aug
nas-3d-aug:
	export PYTHONPATH=$${PYTHONPATH}:. && $(PYTHON) $(SRC_DIR)/train.py --multirun \
		data.use_norm=false data.augment=true

.PHONY: nas-3d-noaug
nas-3d-noaug:
	export PYTHONPATH=$${PYTHONPATH}:. && $(PYTHON) $(SRC_DIR)/train.py --multirun \
		data.use_norm=false data.augment=false data.use_pca=true


.PHONY: export
export:
	export PYTHONPATH=$${PYTHONPATH}:. && $(PYTHON) $(SRC_DIR)/export.py


.PHONY: tensorboard
tensorboard:
	uv run tensorboard --logdir multirun/

# --- Testing ---
.PHONY: test
test:
	export PYTHONPATH=$${PYTHONPATH}:. && uv run pytest ml/tests/

# --- Quality ---
.PHONY: format
format:
	uv run ruff check --fix .
	uv run ruff format .
