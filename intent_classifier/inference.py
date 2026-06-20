"""Runtime prediction helpers and ONNX-backed estimator."""


import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from intent_classifier.model import TextClassifier, load_onnx, postprocess_onnx_outputs
from intent_classifier.preprocessing import load_tokenizer, tokenize_texts
from intent_classifier.settings import HeadConfig, HeadMode, ModelConfig, load_model_config
from intent_classifier.utils import load_json, text_hash

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HeadPrediction:
    mode: str
    probabilities: dict[str, float]
    active_labels: list[str]


def predict_torch(
    model: TextClassifier,
    tokenizer: Any,
    config: ModelConfig,
    text: str,
    device: torch.device | None = None,
) -> dict[str, np.ndarray]:
    model.eval()
    runtime_device = device or torch.device("cpu")
    encoded = tokenize_texts(
        tokenizer,
        [text],
        max_length=config.text.max_length,
        truncation=config.text.truncation,
        padding=config.text.padding,
        return_tensors="pt",
    )
    with torch.no_grad():
        outputs = model(
            input_ids=encoded["input_ids"].to(runtime_device),
            attention_mask=encoded["attention_mask"].to(runtime_device),
        )
    return {name: tensor.detach().cpu().numpy() for name, tensor in outputs.items()}


def predict_onnx(
    session: Any,
    tokenizer: Any,
    config: ModelConfig,
    text: str,
) -> dict[str, np.ndarray]:
    encoded = tokenize_texts(
        tokenizer,
        [text],
        max_length=config.text.max_length,
        truncation=config.text.truncation,
        padding=config.text.padding,
        return_tensors="np",
    )
    inputs = {
        "input_ids": encoded["input_ids"].astype(np.int64),
        "attention_mask": encoded["attention_mask"].astype(np.int64),
    }
    output_names = [f"{head.name}_logits" for head in config.heads]
    outputs = session.run(output_names, inputs)
    return postprocess_onnx_outputs(outputs, config)


class IntentEstimator:
    """Production-oriented ONNX estimator loaded from an artifact directory."""

    def __init__(self, artifact_dir: str | Path) -> None:
        self.artifact_dir = Path(artifact_dir)
        self.config = load_model_config(self.artifact_dir / "model_config.yaml")
        self.tokenizer = load_tokenizer(
            self.artifact_dir / "tokenizer",
            local_files_only=True,
        )
        model_path = self._resolve_model_path()
        self.session = load_onnx(model_path)
        self.calibration = self._load_optional_json("calibration.json", default={"heads": {}})
        self.thresholds = self._load_optional_json("thresholds.json", default={"heads": {}})
        logger.info("Loaded IntentEstimator from %s", self.artifact_dir)

    def predict(self, text: str) -> dict[str, HeadPrediction]:
        logits = predict_onnx(self.session, self.tokenizer, self.config, text)
        return postprocess_predictions(logits, self.config, self.calibration, self.thresholds)

    def predict_with_metadata(self, text: str) -> dict[str, Any]:
        prediction = self.predict(text)
        return {
            "text_hash": text_hash(text),
            "predictions": prediction,
        }

    def _resolve_model_path(self) -> Path:
        int8 = self.artifact_dir / "model.int8.onnx"
        if int8.exists():
            return int8
        fp32 = self.artifact_dir / "model.onnx"
        if fp32.exists():
            return fp32
        raise FileNotFoundError(f"No ONNX model found in {self.artifact_dir}")

    def _load_optional_json(self, name: str, default: dict[str, Any]) -> dict[str, Any]:
        path = self.artifact_dir / name
        if not path.exists():
            return default
        return load_json(path)


def postprocess_predictions(
    logits: dict[str, np.ndarray],
    config: ModelConfig,
    calibration: dict[str, Any] | None = None,
    thresholds: dict[str, Any] | None = None,
) -> dict[str, HeadPrediction]:
    calibration = calibration or {"heads": {}}
    thresholds = thresholds or {"heads": {}}
    predictions: dict[str, HeadPrediction] = {}
    for head in config.heads:
        head_logits = np.asarray(logits[head.name])[0]
        probs = _probabilities_for_head(head_logits, head, calibration)
        probabilities = {label: float(probs[index]) for index, label in enumerate(head.labels)}
        active = _active_labels_for_head(probs, head, thresholds)
        predictions[head.name] = HeadPrediction(
            mode=head.mode.value,
            probabilities=probabilities,
            active_labels=active,
        )
    return predictions


def _probabilities_for_head(
    logits: np.ndarray,
    head: HeadConfig,
    calibration: dict[str, Any],
) -> np.ndarray:
    head_calibration = calibration.get("heads", {}).get(head.name, {})
    if head.mode == HeadMode.MULTI_LABEL:
        temperatures = head_calibration.get("temperatures", {})
        temp_values = np.array([float(temperatures.get(label, 1.0)) for label in head.labels])
        return sigmoid(logits / temp_values)
    temperature = float(head_calibration.get("temperature", 1.0))
    return softmax(logits / temperature)


def _active_labels_for_head(
    probabilities: np.ndarray,
    head: HeadConfig,
    thresholds: dict[str, Any],
) -> list[str]:
    if head.mode == HeadMode.SINGLE_LABEL:
        return [head.labels[int(np.argmax(probabilities))]]
    head_thresholds = thresholds.get("heads", {}).get(head.name, {})
    active = []
    for index, label in enumerate(head.labels):
        threshold = float(head_thresholds.get(label, {}).get("activate", 0.5))
        if probabilities[index] >= threshold:
            active.append(label)
    return active


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exp = np.exp(shifted)
    return exp / exp.sum()
