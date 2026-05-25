# 🏃‍♂️ classiAcceleration: TinyML Fitness Tracker

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-ee4c2c.svg)](https://pytorch.org/)
[![Hardware: Arduino Nano 33 BLE](https://img.shields.io/badge/Hardware-Arduino%20Nano%2033%20BLE-00979d.svg)](https://store.arduino.cc/arduino-nano-33-ble-sense)
[![Quantization: INT8](https://img.shields.io/badge/Quantization-INT8-green.svg)](https://tensorflow.org/lite/performance/post_training_quantization)
[![CI Status](https://github.com/nakmuaycoder/classiAcceleration/actions/workflows/ci.yml/badge.svg)](https://github.com/nakmuaycoder/classiAcceleration/actions)

## 🔄 Project Revival: 5 Years Later
This repository is a revival of a personal project originally created in **early 2021** (see tag `v1_02/21` for the legacy version).

The goal was to build a fitness tracker capable of distinguishing between **walking** and **running** using 3D accelerations on an **Arduino Nano 33 BLE Sense**.

### Why the upgrade?
TinyML has evolved significantly in the last 5 years. This revival aims to:
- **Modernize the Stack**: Transitioning from a script-based TensorFlow 1.x approach to a professional **PyTorch** workflow.
- **Robustness via Math**: Re-implementing **3D Rotation Augmentation** as a vectorised PyTorch layer to make the model invariant to sensor orientation.
- **Advanced Quantization**: Exploring **INT8 Full Integer Quantization** and **Quantization-Aware Training (QAT)** to maximize the tiny Cortex-M4 performance.
- **Seamless Export**: Utilizing Google's high-level **`ai-edge-torch`** for a robust PyTorch -> TFLite conversion path.

---

## 🏗 Architecture
The project follows a modern, modular structure to ensure maintainability and testability:

```text
.
├── firmware/         # Arduino C++ code (LSM9DS1 sensor logic)
├── ml/
│   ├── config/       # Hydra & Optuna configuration files
│   ├── src/          # Core Python modules (model, augmentation, parser)
│   ├── tests/        # Pytest unit testing suite
│   └── notebooks/    # EDA and experimental analysis
├── pyproject.toml    # Unified dependency & tool configuration
├── data/
│   ├── raw/          # LightBlue BLE Sniffer logs captured from Arduino (.txt)
│   └── clean/        # Processed CSVs for model training
└── Makefile          # Unified project workflow (Data -> Test -> Train)
```

---

## 🛠 Features

### 📐 Robust Data Augmentations
One of the biggest challenges in wearable tech is sensor orientation. The training pipeline supports configurable augmentations managed dynamically via Hydra:
- **Random 3D Rotation**: Spatial rotations using the Rodrigues Formula to make the model invariant to sensor placement.
- **Add Gaussian Noise**: Inserts zero-mean Gaussian noise to improve robustness.
- **Random Scaling / Bias Shift**: Simulates sensor sensitivity differences and offsets.

### 📐 PCA Coordinate Alignment
Alternatively, the dataset supports deterministic alignment using a scikit-learn pipeline (`StandardScaler` + `PCA`) on each window to test if the model learns better on pre-aligned coordinate axes.

### 🧠 Modern 1D-CNN Architecture
Optimized for tiny ARM Cortex-M4 processors:
- **Param Count**: ~1,000 parameters.
- **Accuracy**: High precision even under INT8 quantization.

---

## 🚀 Getting Started

### Prerequisites
- **`uv`**: The modern Python package manager.
- **Arduino IDE / CLI**: For firmware deployment.

### Installation
```bash
# Install dependencies, setup env and git hooks
make install
```

### Development Workflow
```bash
# 1. Preprocess LightBlue sniffer logs with strict alignment verification
make data

# 2. Run quality checks & unit tests
make format
make test

# 3. Train a single model using the active configuration
make train

# 4. Neural Architecture Search (NAS) configurations
make nas-1d          # NAS search with 1D vector magnitude input (norm)
make nas-3d-aug      # NAS search with 3D input and stochatic augmentations
make nas-3d-noaug    # NAS search with 3D input and deterministic PCA axis alignment
```

---

## ✨ Code Quality & Security
To ensure a production-ready codebase, we integrated:
- **Ruff**: Ultra-fast linting and formatting.
- **Pytest**: Automated testing of the augmentation math.
- **Pre-commit**: Automatic quality checks before every commit.
- **Secret Detection**: Protection against accidental leaks of credentials.

---

## 📜 Historical Context (v1)
The original implementation (available at tag `v1_02/21`) focused on:
- Data collection via BLE and LightBlue app.
- TensorFlow Lite for Microcontrollers (Legacy API).

Check out the [Legacy README](https://github.com/nakmuaycoder/classiAcceleration/tree/v1_02/21) context in the git history or the `v1_02/21` tag for the original 2021 code.
