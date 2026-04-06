# --- Variables ---
PYTHON := uv run python
SRC_DIR := ml/src
ARDUINO_DIR := firmware/FitnessTracker

# --- Default Goal ---
.PHONY: help
help:
	@echo "Available commands:"
	@echo "  make install     Install dependencies and setup environment"
	@echo "  make clean       Remove cached files and temporary data"
	@echo "  make data        Preprocess raw logs into clean CSVs"
	@echo "  make test        Run unit tests (augmentation, etc.)"
	@echo "  make format      Format and lint code with Ruff"
	@echo "  make train       Train the model with 3D rotation augmentation"
	@echo "  make export      Convert trained model to TFLite (INT8)"
	@echo "  make firmware    Compile and upload (if arduino-cli is installed)"

# --- Installation ---
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

# --- Parameters (can be overridden from CLI) ---
AUGMENT ?= 0
QUANTIZE ?= 0
TRAIN_FILES ?=
VAL_FILES ?=

# --- Model Workflow ---
.PHONY: train
train:
	$(PYTHON) $(SRC_DIR)/train.py \
		$(if $(filter 1,$(AUGMENT)),--augment) \
		$(if $(filter 1,$(QUANTIZE)),--quantize) \
		$(if $(TRAIN_FILES),--train-files $(TRAIN_FILES)) \
		$(if $(VAL_FILES),--val-files $(VAL_FILES))

.PHONY: export
export:
	$(PYTHON) $(SRC_DIR)/export.py

# --- Testing ---
.PHONY: test
test:
	export PYTHONPATH=$${PYTHONPATH}:. && uv run pytest ml/src/

# --- Quality ---
.PHONY: format
format:
	uv run ruff check --fix
	uv run ruff format
