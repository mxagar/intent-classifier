"""Typed configuration loading and validation with Pydantic."""

from enum import Enum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class HeadMode(str, Enum):
    MULTI_LABEL = "multi_label"
    SINGLE_LABEL = "single_label"


class ActivationName(str, Enum):
    RELU = "relu"
    GELU = "gelu"
    TANH = "tanh"


class PaddingMode(str, Enum):
    MAX_LENGTH = "max_length"
    LONGEST = "longest"
    DO_NOT_PAD = "do_not_pad"


ALLOWED_ORIGINS = {"synthetic", "manual", "production", "unknown"}
ALLOWED_DOMAINS = {"general", "electricista", "fontanero", "albañil", "carpintero", "unknown"}


class FrozenConfigModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)


class LoggingConfig(FrozenConfigModel):
    level: str = "INFO"


class SettingsConfig(FrozenConfigModel):
    model_config_path: Path = Field(alias="model_config")
    train_config_path: Path = Field(alias="train_config")
    logging: LoggingConfig = LoggingConfig()


class BackboneConfig(FrozenConfigModel):
    name: str
    revision: str = "main"
    local_tokenizer_path: Path

    @field_validator("local_tokenizer_path")
    @classmethod
    def local_tokenizer_path_must_be_relative(cls, path: Path) -> Path:
        if path.is_absolute():
            raise ValueError("local_tokenizer_path must be relative to the artifact version directory")
        return path


class TextConfig(FrozenConfigModel):
    max_length: int = Field(default=128, gt=0)
    truncation: bool = True
    padding: PaddingMode = PaddingMode.MAX_LENGTH


class ClassifierConfig(FrozenConfigModel):
    dropout: float = Field(default=0.2, ge=0.0, lt=1.0)


class HiddenLayerConfig(FrozenConfigModel):
    size: int = Field(gt=0)
    activation: ActivationName = ActivationName.RELU


class HeadConfig(FrozenConfigModel):
    name: str
    mode: HeadMode
    hidden_layer: HiddenLayerConfig
    labels: tuple[str, ...]

    @field_validator("labels")
    @classmethod
    def labels_must_be_unique(cls, labels: tuple[str, ...]) -> tuple[str, ...]:
        duplicates = sorted({label for label in labels if labels.count(label) > 1})
        if duplicates:
            raise ValueError(f"Duplicate labels: {duplicates}")
        if not labels:
            raise ValueError("At least one label is required")
        return labels

    @property
    def label_columns(self) -> tuple[str, ...]:
        return tuple(f"{self.name}__{label}" for label in self.labels)


class ModelConfig(FrozenConfigModel):
    backbone: BackboneConfig
    text: TextConfig = TextConfig()
    classifier: ClassifierConfig = ClassifierConfig()
    heads: tuple[HeadConfig, ...]

    @model_validator(mode="after")
    def heads_must_be_unique(self) -> "ModelConfig":
        names = [head.name for head in self.heads]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"Duplicate head names: {duplicates}")
        if not self.heads:
            raise ValueError("At least one head is required")
        return self

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


class SplitConfig(FrozenConfigModel):
    train: float = Field(default=0.8, gt=0.0, lt=1.0)
    validation: float = Field(default=0.1, gt=0.0, lt=1.0)
    test: float = Field(default=0.1, gt=0.0, lt=1.0)
    group_column: str | None = None

    @model_validator(mode="after")
    def fractions_must_sum_to_one(self) -> "SplitConfig":
        total = self.train + self.validation + self.test
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Split fractions must sum to 1.0, got {total}")
        return self


class PhaseConfig(FrozenConfigModel):
    freeze_backbone: bool
    epochs: int = Field(ge=0)
    learning_rate_classifier: float = Field(gt=0.0)
    batch_size: int = Field(gt=0)
    learning_rate_backbone: float | None = Field(default=None, gt=0.0)


class TrainingConfig(FrozenConfigModel):
    phase_1: PhaseConfig
    phase_2: PhaseConfig
    weight_decay: float = Field(default=0.01, ge=0.0)
    gradient_clipping: float | None = Field(default=1.0, gt=0.0)
    early_stopping_patience: int = Field(default=3, ge=0)


class LossConfig(FrozenConfigModel):
    head_weights: dict[str, float] = Field(default_factory=dict)
    max_pos_weight: float = Field(default=10.0, gt=0.0)
    use_pos_weights: bool = True


class HpoSearchSpace(FrozenConfigModel):
    dropout: tuple[float, float] = (0.1, 0.4)
    max_length: tuple[int, ...] = (64, 128, 256)
    batch_size: tuple[int, ...] = (8, 16, 32)
    weight_decay: tuple[float, float] = (0.0, 0.05)
    learning_rate_backbone: tuple[float, float] = (1e-5, 5e-5)
    learning_rate_classifier: tuple[float, float] = (1e-4, 1e-3)
    max_pos_weight: tuple[float, float] = (2.0, 15.0)
    head_hidden_sizes: tuple[int, ...] = (64, 128, 256, 384)


class OptunaConfig(FrozenConfigModel):
    enabled: bool = False
    n_trials: int = Field(default=30, gt=0)
    metric: str = "mean_macro_f1"
    search_space: HpoSearchSpace = HpoSearchSpace()


class TrainConfig(FrozenConfigModel):
    seed: int = 42
    dataset_csv: Path
    artifact_dir: Path
    version_dir: Path
    splits: SplitConfig = SplitConfig()
    training: TrainingConfig
    loss: LossConfig = LossConfig()
    optuna: OptunaConfig = OptunaConfig()

    @field_validator("version_dir")
    @classmethod
    def version_dir_must_be_relative(cls, path: Path) -> Path:
        if path.is_absolute():
            raise ValueError("version_dir must be relative to artifact_dir")
        return path

    @property
    def artifacts_root(self) -> Path:
        return self.artifact_dir

    @property
    def current_artifact_dir(self) -> Path:
        return self.artifact_dir / self.version_dir


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    return data


def load_model_config(path: str | Path) -> ModelConfig:
    return ModelConfig.model_validate(load_yaml(path))


def load_train_config(path: str | Path) -> TrainConfig:
    return TrainConfig.model_validate(load_yaml(path))


def load_settings_config(path: str | Path = "intent_classifier/config/settings.yaml") -> SettingsConfig:
    return SettingsConfig.model_validate(load_yaml(path))
