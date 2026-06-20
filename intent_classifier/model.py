"""PyTorch model and ONNX helpers."""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import torch
from torch import nn
from transformers import AutoModel

from intent_classifier.settings import ActivationName, HeadConfig, ModelConfig
from intent_classifier.utils import ensure_dir

logger = logging.getLogger(__name__)


class HeadMLP(nn.Module):
    """A per-head hidden layer followed by the head output projection."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        activation: ActivationName,
        dropout: float,
    ) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            _activation(activation),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layers(features)


class TextClassifier(nn.Module):
    """Shared encoder plus configurable per-head MLP classifiers."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = AutoModel.from_pretrained(
            config.backbone.name,
            revision=config.backbone.revision,
        )
        encoder_hidden_size = int(self.encoder.config.hidden_size)
        self.dropout = nn.Dropout(config.classifier.dropout)
        self.heads = nn.ModuleDict(
            {
                head.name: HeadMLP(
                    input_size=encoder_hidden_size,
                    hidden_size=head.hidden_layer.size,
                    output_size=len(head.labels),
                    activation=head.hidden_layer.activation,
                    dropout=config.classifier.dropout,
                )
                for head in config.heads
            }
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]
        features = self.dropout(pooled)
        logits = {name: head(features) for name, head in self.heads.items()}
        logits["features"] = features
        return logits

    def freeze_backbone(self) -> None:
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for parameter in self.encoder.parameters():
            parameter.requires_grad = True


class OnnxTextClassifierWrapper(nn.Module):
    """Wrapper that returns tuple outputs for ONNX export."""

    def __init__(self, model: TextClassifier, heads: tuple[HeadConfig, ...]) -> None:
        super().__init__()
        self.model = model
        self.head_names = tuple(head.name for head in heads)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, ...]:
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return tuple(outputs[name] for name in self.head_names)


def create_model(config: ModelConfig) -> TextClassifier:
    return TextClassifier(config)


def load_onnx(path: str | Path, providers: list[str] | None = None) -> ort.InferenceSession:
    return ort.InferenceSession(
        str(path),
        providers=providers or ["CPUExecutionProvider"],
    )


def export_onnx(
    model: TextClassifier,
    config: ModelConfig,
    output_path: str | Path,
    opset_version: int = 17,
    device: torch.device | None = None,
) -> Path:
    """Export raw logits for each configured head."""
    output = Path(output_path)
    ensure_dir(output.parent)
    export_device = device or torch.device("cpu")
    model = model.to(export_device)
    model.eval()
    wrapper = OnnxTextClassifierWrapper(model, config.heads).to(export_device)
    dummy_input_ids = torch.ones((1, config.text.max_length), dtype=torch.long, device=export_device)
    dummy_attention_mask = torch.ones_like(dummy_input_ids)
    output_names = [f"{head.name}_logits" for head in config.heads]
    dynamic_axes: dict[str, dict[int, str]] = {
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "attention_mask": {0: "batch_size", 1: "sequence_length"},
    }
    for name in output_names:
        dynamic_axes[name] = {0: "batch_size"}

    logger.info("Exporting ONNX model to %s", output)
    torch.onnx.export(
        wrapper,
        (dummy_input_ids, dummy_attention_mask),
        str(output),
        input_names=["input_ids", "attention_mask"],
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
    )
    return output


def preprocess_onnx_inputs(batch: dict[str, Any]) -> dict[str, np.ndarray]:
    return {
        "input_ids": np.asarray(batch["input_ids"], dtype=np.int64),
        "attention_mask": np.asarray(batch["attention_mask"], dtype=np.int64),
    }


def postprocess_onnx_outputs(
    outputs: list[np.ndarray],
    config: ModelConfig,
) -> dict[str, np.ndarray]:
    if len(outputs) != len(config.heads):
        raise ValueError(f"Expected {len(config.heads)} ONNX outputs, got {len(outputs)}")
    return {head.name: outputs[index] for index, head in enumerate(config.heads)}


def _activation(name: ActivationName) -> nn.Module:
    if name == ActivationName.GELU:
        return nn.GELU()
    if name == ActivationName.RELU:
        return nn.ReLU()
    if name == ActivationName.TANH:
        return nn.Tanh()
    raise ValueError(f"Unsupported activation: {name}")
