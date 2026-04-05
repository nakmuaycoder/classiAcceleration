# 🏃‍♂️ classiAcceleration: TinyML Fitness Tracker

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-ee4c2c.svg)](https://pytorch.org/)
[![Hardware: Arduino Nano 33 BLE](https://img.shields.io/badge/Hardware-Arduino%20Nano%2033%20BLE-00979d.svg)](https://store.arduino.cc/arduino-nano-33-ble-sense)
[![Quantization: INT8](https://img.shields.io/badge/Quantization-INT8-green.svg)](https://tensorflow.org/lite/performance/post_training_quantization)

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

## 🛠 Features

### 📐 Robust 3D Rotation Augmentation
One of the biggest challenges in wearable tech is sensor orientation. If the user wears the device upside down or on a different limb, raw $(x, y, z)$ values change.
- Our custom `Random3DRotation` layer applies random spatial rotations using the **Rodrigues Formula**.
- This ensures the model learns the "physics" of the movement (accelerations patterns) rather than fixed directional magnitudes.

### 🧠 Modern 1D-CNN Architecture
Unlike the original 2D approach, we now use a **1D Convolutional Neural Network** which is more biologically and physically aligned with time-series acceleration data.
- **Param Count**: ~1,000 parameters.
- **Latency**: Sub-millisecond inference on MCU.

---

## 🚀 Getting Started

### Prerequisites
- **`uv`**: The modern Python package manager.
- **Arduino IDE / CLI**: For firmware deployment.

### Installation
Setting up the environment is automated via the `Makefile`:

```bash
# Install dependencies and sync environment
make install
```

### Training & Export
```bash
# Train the model with 3D rotation augmentation
make train

# Convert to TFLite (INT8 Quantized)
make export
```

---

## 📜 Historical Context (v1)
The original implementation (available at tag `v1_02/21`) focused on:
- Data collection via BLE and LightBlue app.
- TensorFlow Lite for Microcontrollers (Legacy API).
- Basic Jupyter Notebook analysis.

Check out the [Legacy README](https://github.com/nakmuaycoder/classiAcceleration/tree/v1_02/21) context in the git history or the `v1_02/21` tag for the original 2021 code.

