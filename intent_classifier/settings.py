"""Typed configuration loading and validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

HeadMode = Literal["multi_label", "single_label"]
ActivationName = Literal["gelu", "relu", "tanh"]
PaddingMode = Literal["max_length", "longest", "do_not_pad"]

ALLOWED_ORIGINS = {"synthetic", "manual", "production", "unknown"}
ALLOWED_DOMAINS = {"general", "electricista", "fontanero", "albañil", "carpintero", "unknown"}


@dataclass(frozen=True)
class BackboneConfig:
    name: str
    revision: str
    local_tokenizer_path: Path


@dataclass(frozen=True)
class TextConfig:
    max_length: int = 128
    truncation: bool = True
    padding: PaddingMode = "max_length"


@dataclass(frozen=True)
class ClassifierConfig:
    dropout: float = 0.2


@dataclass(frozen=True)
class HiddenLayerConfig:
    size: int
    activation: ActivationName = "gelu"


@dataclass(frozen=True)
class HeadConfig:
    name: str
    mode: HeadMode
    hidden_layer: HiddenLayerConfig
    labels: tuple[str, ...]

    @property
    def label_columns(self) -> tuple[str, ...]:
        return tuple(f"{self.name}__{label}" for label in self.labels)


@dataclass(frozen=True)
class ModelConfig:
    backbone: BackboneConfig
    text: TextConfig
    classifier: ClassifierConfig
    heads: tuple[HeadConfig, ...]

    @property
    def head_names(self) -> tuple[str, ...]:
        return tuple(head.name for head in self.heads)

    @property
    def label_columns(self) -> tuple[str, ...]:
        return tuple(column for head in self.heads for column in head.label_columns)

    def head_by_name(self, name: str) -> HeadConfig:
        for head in self.heads:
            if head.name == name:
                return head
        raise KeyError(f"Unknown head: {name}")


@dataclass(frozen=True)
class SplitConfig:
    train: float = 0.7
    validation: float = 0.1
    calibration: float = 0.1
    test: float = 0.1
    group_column: str | None = None


@dataclass(frozen=True)
class PhaseConfig:
    freeze_backbone: bool
    epochs: int
    learning_rate_classifier: float
    batch_size: int
    learning_rate_backbone: float | None = None


@dataclass(frozen=True)
class TrainingConfig:
    phase_1: PhaseConfig
    phase_2: PhaseConfig
    weight_decay: float = 0.01
    gradient_clipping: float = 1.0
    early_stopping_patience: int = 3


@dataclass(frozen=True)
class LossConfig:
    head_weights: dict[str, float]
    max_pos_weight: float = 10.0


@dataclass(frozen=True)
class OptunaConfig:
    enabled: bool = False
    n_trials: int = 30


@dataclass(frozen=True)
class TrainConfig:
    seed: int
    dataset_csv: Path
    model_config: Path
    artifact_dir: Path
    splits: SplitConfig
    training: TrainingConfig
    loss: LossConfig
    optuna: OptunaConfig


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    return data


def load_model_config(path: str | Path) -> ModelConfig:
    raw = load_yaml(path)
    backbone_raw = _required_mapping(raw, "backbone")
    text_raw = raw.get("text", {})
    classifier_raw = raw.get("classifier", {})

    backbone = BackboneConfig(
        name=_required_str(backbone_raw, "name"),
        revision=str(backbone_raw.get("revision", "main")),
        local_tokenizer_path=Path(_required_str(backbone_raw, "local_tokenizer_path")),
    )
    text = TextConfig(
        max_length=int(text_raw.get("max_length", 128)),
        truncation=bool(text_raw.get("truncation", True)),
        padding=_validate_padding(str(text_raw.get("padding", "max_length"))),
    )
    if text.max_length <= 0:
        raise ValueError("text.max_length must be positive")

    classifier = ClassifierConfig(dropout=float(classifier_raw.get("dropout", 0.2)))
    if not 0 <= classifier.dropout < 1:
        raise ValueError("classifier.dropout must be in [0, 1)")

    heads_raw = raw.get("heads")
    if not isinstance(heads_raw, list) or not heads_raw:
        raise ValueError("model_config.yaml must define at least one head")

    heads = tuple(_parse_head(item) for item in heads_raw)
    _validate_unique("head names", [head.name for head in heads])
    return ModelConfig(backbone=backbone, text=text, classifier=classifier, heads=heads)


def load_train_config(path: str | Path) -> TrainConfig:
    raw = load_yaml(path)
    training_raw = _required_mapping(raw, "training")
    phase_1 = _parse_phase(_required_mapping(training_raw, "phase_1"))
    phase_2 = _parse_phase(_required_mapping(training_raw, "phase_2"))
    splits = _parse_splits(raw.get("splits", {}))
    loss_raw = raw.get("loss", {})
    optuna_raw = raw.get("optuna", {})

    return TrainConfig(
        seed=int(raw.get("seed", 42)),
        dataset_csv=Path(_required_str(raw, "dataset_csv")),
        model_config=Path(_required_str(raw, "model_config")),
        artifact_dir=Path(_required_str(raw, "artifact_dir")),
        splits=splits,
        training=TrainingConfig(
            phase_1=phase_1,
            phase_2=phase_2,
            weight_decay=float(training_raw.get("weight_decay", 0.01)),
            gradient_clipping=float(training_raw.get("gradient_clipping", 1.0)),
            early_stopping_patience=int(training_raw.get("early_stopping_patience", 3)),
        ),
        loss=LossConfig(
            head_weights={
                str(k): float(v) for k, v in (loss_raw.get("head_weights") or {}).items()
            },
            max_pos_weight=float(loss_raw.get("max_pos_weight", 10.0)),
        ),
        optuna=OptunaConfig(
            enabled=bool(optuna_raw.get("enabled", False)),
            n_trials=int(optuna_raw.get("n_trials", 30)),
        ),
    )


def _parse_head(raw: Any) -> HeadConfig:
    if not isinstance(raw, dict):
        raise ValueError("Each head must be a mapping")
    labels_raw = raw.get("labels")
    if not isinstance(labels_raw, list) or not labels_raw:
        raise ValueError(f"Head {raw.get('name', '<unknown>')} must define labels")
    labels = tuple(str(label) for label in labels_raw)
    _validate_unique(f"labels for head {raw.get('name')}", labels)
    hidden_raw = _required_mapping(raw, "hidden_layer")
    hidden = HiddenLayerConfig(
        size=int(hidden_raw.get("size", 0)),
        activation=_validate_activation(str(hidden_raw.get("activation", "gelu"))),
    )
    if hidden.size <= 0:
        raise ValueError(f"Head {raw.get('name')} hidden_layer.size must be positive")
    return HeadConfig(
        name=_required_str(raw, "name"),
        mode=_validate_head_mode(_required_str(raw, "mode")),
        hidden_layer=hidden,
        labels=labels,
    )


def _parse_phase(raw: dict[str, Any]) -> PhaseConfig:
    learning_rate_backbone = raw.get("learning_rate_backbone")
    return PhaseConfig(
        freeze_backbone=bool(raw.get("freeze_backbone", False)),
        epochs=int(raw.get("epochs", 1)),
        learning_rate_classifier=float(raw.get("learning_rate_classifier", 5e-4)),
        batch_size=int(raw.get("batch_size", 16)),
        learning_rate_backbone=(
            None if learning_rate_backbone is None else float(learning_rate_backbone)
        ),
    )


def _parse_splits(raw: Any) -> SplitConfig:
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("splits must be a mapping")
    split = SplitConfig(
        train=float(raw.get("train", 0.7)),
        validation=float(raw.get("validation", 0.1)),
        calibration=float(raw.get("calibration", 0.1)),
        test=float(raw.get("test", 0.1)),
        group_column=raw.get("group_column"),
    )
    total = split.train + split.validation + split.calibration + split.test
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Split fractions must sum to 1.0, got {total}")
    return split


def _required_mapping(raw: dict[str, Any], key: str) -> dict[str, Any]:
    value = raw.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Missing or invalid mapping: {key}")
    return value


def _required_str(raw: dict[str, Any], key: str) -> str:
    value = raw.get(key)
    if value is None or str(value) == "":
        raise ValueError(f"Missing required string: {key}")
    return str(value)


def _validate_unique(name: str, values: list[str] | tuple[str, ...]) -> None:
    duplicates = sorted({value for value in values if values.count(value) > 1})
    if duplicates:
        raise ValueError(f"Duplicate {name}: {duplicates}")


def _validate_head_mode(value: str) -> HeadMode:
    if value not in {"multi_label", "single_label"}:
        raise ValueError(f"Invalid head mode: {value}")
    return value  # type: ignore[return-value]


def _validate_activation(value: str) -> ActivationName:
    if value not in {"gelu", "relu", "tanh"}:
        raise ValueError(f"Invalid activation: {value}")
    return value  # type: ignore[return-value]


def _validate_padding(value: str) -> PaddingMode:
    if value not in {"max_length", "longest", "do_not_pad"}:
        raise ValueError(f"Invalid padding mode: {value}")
    return value  # type: ignore[return-value]

