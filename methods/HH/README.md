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
*   **`verify_on_real_model.py`**: Self-contained verification script that exercises the first layers of DeepSeek-R1-Distill-Qwen. It:
    * Captures layer-0 inputs through a catcher module to ensure rotary/positional state is preserved.
    * Loads WikiText-2 (fallbacks to random tokens if the dataset is unavailable) for calibration.
    * Initializes Householder rotations via PCA on observed activations (selecting the smallest h that explains ≥95% variance), calibrates quantizers, and optimizes rotations to reduce layer-wise MSE without introducing NaNs.
    * Reports baseline vs. final MSE per layer so regressions are easy to spot.
*   **`optimization.py`** / **`hh_train_utils.py`**: Helper modules for end-to-end training and scaling logic.

## Verification quickstart

```bash
python methods/HH/verify_on_real_model.py \
  --model-path deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  --nsamples 8 \
  --cali-bsz 2 \
  --num-layers 2
```

Key flags:

* `--model-path`: Hugging Face repo or local checkpoint for a Qwen-based DeepSeek model. An environment variable `DEEPSEEK_MODEL_PATH` is also respected.
* `--nsamples` / `--cali-bsz` / `--max-seq-len`: Control calibration volume to fit the available GPU memory.
* `--steps` / `--lr`: Tune Householder optimization effort; defaults are conservative for quick runs.
* `--reflections`: Number of Householder reflections per rotation (applied to QKV, O, and MLP projections).

What the script checks:

1. The catcher mechanism collects the correct hidden states and positional kwargs for the first layer.
2. WikiText-2 loading works (with a random-data fallback for offline runs).
3. Layer-wise optimization reduces reconstruction MSE on the first `--num-layers` without producing NaNs; final metrics are logged per layer.

Use `--device cuda:0` to pin execution to a specific GPU or rely on automatic selection.
