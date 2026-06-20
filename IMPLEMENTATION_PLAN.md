# Implementation Plan

This plan follows `SPEC.md` from foundational modules toward downstream training,
evaluation, and production inference.

## 1. Project and Configuration Foundation

- Create Python package structure under `intent_classifier/`.
- Add `pyproject.toml` and manage dependencies with `uv`.
- Add default YAML files:
  - `intent_classifier/config/model_config.yaml`
  - `intent_classifier/config/train_config.yaml`
  - `intent_classifier/config/settings.yaml`
- Implement typed config loading and validation in `intent_classifier/settings.py`.

## 2. Model and Tokenizer Foundation

- Implement tokenizer export and local loading in `preprocessing.py`.
- Implement light text normalization.
- Implement `TextClassifier` in `model.py`:
  - Shared transformer encoder.
  - Configurable per-head hidden layers.
  - Generic named output heads.
- Implement ONNX load/export preprocessing and postprocessing helpers.

## 3. Dataset Layer

- Implement CSV loading and validation in `dataset.py`.
- Validate required metadata columns: `id`, `text`, `origin`, `domain`.
- Validate `<head_name>__<intent_name>` label columns.
- Build multi-label and single-label targets from config.
- Implement batching, data loader creation, split generation, and label distributions.

## 4. Training, Metrics, Export, and Calibration

- Implement optimizer and scheduler creation.
- Implement multi-head loss:
  - BCE with logits for multi-label heads.
  - Cross entropy for single-label heads.
- Implement metrics per head and per label.
- Implement two-phase training scaffold.
- Implement checkpoint save/load.
- Implement Optuna study hooks.
- Implement ONNX quantization.
- Implement calibration and threshold save/load/tuning helpers.

## 5. Inference Runtime

- Implement `IntentEstimator` in `inference.py`.
- Load model config, local tokenizer, ONNX model, calibration, and thresholds from artifacts.
- Implement Torch and ONNX prediction helpers.
- Implement calibrated probability and active-label postprocessing.

## 6. Tests

- Add pytest tests for:
  - Config loading and validation.
  - CSV dataset validation and target construction.
  - Model output shapes with a mocked encoder.
  - Multi-head loss and metrics.
  - Inference postprocessing.

## 7. Production Hardening Still Needed

- Train with real CSV data.
- Export real tokenizer artifacts.
- Export and validate real ONNX and INT8 ONNX models.
- Fit real calibration parameters on deployed logits.
- Tune real thresholds on the calibration split.
- Fill `changelog.yaml` with real dataset hashes, distributions, and metrics.
- Add integration tests for ONNX Runtime artifacts.
- Decide artifact storage backend if local artifacts are not enough.

