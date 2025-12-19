# Householder Quantization (HH) Method

This directory contains the implementation of the Householder Quantization method, designed to reduce quantization error in Large Language Models (LLMs) by applying learnable orthogonal transformations (Householder reflections).

## Key Components

### Core Modules
*   **`hh_modules.py`**: Defines the `HouseholderRotation` class, which implements the learnable rotation $Q = H_k \dots H_1$ and its initialization from activation data (PCA-based).
*   **`hh_linear.py`**: Implements `HHLinear`, a wrapper around `nn.Linear` that integrates the Householder rotation and quantization logic. It supports "reparameterization" to fold rotations into weights for inference.
*   **`quant_utils.py`**: Contains quantization utilities (`WeightQuantizer`, `ActivationQuantizer`) adapted from FlatQuant/QS, supporting learnable scales and zero-points.

### Model Integration
*   **`hh_layers.py`**: Custom Qwen2 layers (`HHQwen2Attention`, `HHQwen2MLP`) that replace standard layers. They handle shared rotations (e.g., QKV sharing, Up/Gate sharing) and integrate `HHLinear`.
*   **`hh_model_utils.py`**: Utilities to replace model layers with HH versions (`apply_hh_to_qwen`), reparameterize the model, and managed devices.

### Verification & Training
*   **`verify_on_real_model.py`**: The main script for verifying the method on real models (e.g., DeepSeek-R1-Distill-Qwen-1.5B). It performs:
    *   Layer-wise calibration using real data (WikiText-2).
    *   Initialization of HH parameters via SVD/PCA on activations.
    *   Optimization of rotation parameters using AdamW to minimize layer-wise reconstruction error (MSE).
*   **`optimization.py`** / **`hh_train_utils.py`**: (Optional) Helper modules for the training loop and optimization logic.

## Usage

To verify the method on a real model:

```bash
python verify_on_real_model.py
```

This script will:
1.  Load the model and WikiText-2 calibration data.
2.  Iterate through the first few layers.
3.  Optimize Householder rotations to align the activation space with quantization grid.
4.  Report the Mean Squared Error (MSE) improvement.
