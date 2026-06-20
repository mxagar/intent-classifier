"""CSV dataset validation, target construction, splitting, and batching."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import PreTrainedTokenizerBase

from intent_classifier.preprocessing import normalize_text
from intent_classifier.settings import (
    ALLOWED_DOMAINS,
    ALLOWED_ORIGINS,
    HeadConfig,
    ModelConfig,
    SplitConfig,
)

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = ("id", "text", "origin", "domain")


@dataclass(frozen=True)
class DatasetSplits:
    train: list[int]
    validation: list[int]
    calibration: list[int]
    test: list[int]


class RequestDataset(Dataset[dict[str, Any]]):
    """PyTorch dataset backed by the validated CSV schema from the spec."""

    def __init__(self, frame: pd.DataFrame, config: ModelConfig) -> None:
        self.frame = validate_dataset_frame(frame.reset_index(drop=True), config)
        self.config = config

    @classmethod
    def from_csv(cls, path: str | Path, config: ModelConfig) -> "RequestDataset":
        return cls(load_dataset_csv(path), config)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        return {
            "id": str(row["id"]),
            "text": normalize_text(str(row["text"])),
            "origin": str(row["origin"]),
            "domain": str(row["domain"]),
            "labels": build_targets_for_row(row, self.config),
        }


def load_dataset_csv(path: str | Path) -> pd.DataFrame:
    logger.info("Loading dataset CSV from %s", path)
    return pd.read_csv(path)


def validate_dataset_frame(frame: pd.DataFrame, config: ModelConfig) -> pd.DataFrame:
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    missing.extend(column for column in config.label_columns if column not in frame.columns)
    if missing:
        raise ValueError(f"Missing required dataset columns: {missing}")

    if frame[list(REQUIRED_COLUMNS)].isna().any().any():
        raise ValueError("Required metadata columns cannot contain missing values")

    invalid_origins = sorted(set(frame["origin"].astype(str)) - ALLOWED_ORIGINS)
    if invalid_origins:
        raise ValueError(f"Invalid origin values: {invalid_origins}")

    invalid_domains = sorted(set(frame["domain"].astype(str)) - ALLOWED_DOMAINS)
    if invalid_domains:
        raise ValueError(f"Invalid domain values: {invalid_domains}")

    label_frame = frame[list(config.label_columns)]
    if label_frame.isna().any().any():
        raise ValueError("Label columns cannot contain missing values")
    invalid_label_columns = [
        column for column in config.label_columns if not set(frame[column].unique()).issubset({0, 1})
    ]
    if invalid_label_columns:
        raise ValueError(f"Label columns must contain only 0/1: {invalid_label_columns}")

    for head in config.heads:
        _validate_head_targets(frame, head)

    return frame.copy()


def build_targets_for_row(row: pd.Series, config: ModelConfig) -> dict[str, torch.Tensor]:
    labels: dict[str, torch.Tensor] = {}
    for head in config.heads:
        values = [int(row[f"{head.name}__{label}"]) for label in head.labels]
        if head.mode == "multi_label":
            labels[head.name] = torch.tensor(values, dtype=torch.float32)
        else:
            labels[head.name] = torch.tensor(int(np.argmax(values)), dtype=torch.long)
    return labels


def build_batch(
    examples: list[dict[str, Any]],
    tokenizer: PreTrainedTokenizerBase,
    config: ModelConfig,
) -> dict[str, Any]:
    texts = [example["text"] for example in examples]
    encoded = tokenizer(
        texts,
        truncation=config.text.truncation,
        max_length=config.text.max_length,
        padding=config.text.padding,
        return_tensors="pt",
    )
    labels = {
        head.name: torch.stack([example["labels"][head.name] for example in examples])
        for head in config.heads
    }
    return {
        "ids": [example["id"] for example in examples],
        "texts": texts,
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        "labels": labels,
    }


def create_data_loaders(
    dataset: RequestDataset,
    tokenizer: PreTrainedTokenizerBase,
    config: ModelConfig,
    splits: DatasetSplits,
    batch_size: int,
    num_workers: int = 0,
) -> dict[str, DataLoader[dict[str, Any]]]:
    def collate(examples: list[dict[str, Any]]) -> dict[str, Any]:
        return build_batch(examples, tokenizer, config)

    return {
        "train": DataLoader(
            Subset(dataset, splits.train),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate,
        ),
        "validation": DataLoader(
            Subset(dataset, splits.validation),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate,
        ),
        "calibration": DataLoader(
            Subset(dataset, splits.calibration),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate,
        ),
        "test": DataLoader(
            Subset(dataset, splits.test),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate,
        ),
    }


def split_ids(frame: pd.DataFrame, split_config: SplitConfig, seed: int = 42) -> DatasetSplits:
    indices = list(range(len(frame)))
    stratify = None
    train_idx, remainder_idx = train_test_split(
        indices,
        train_size=split_config.train,
        random_state=seed,
        shuffle=True,
        stratify=stratify,
    )
    remainder_fraction = split_config.validation + split_config.calibration + split_config.test
    validation_fraction = split_config.validation / remainder_fraction
    validation_idx, calibration_test_idx = train_test_split(
        remainder_idx,
        train_size=validation_fraction,
        random_state=seed,
        shuffle=True,
    )
    calibration_test_fraction = split_config.calibration + split_config.test
    calibration_fraction = split_config.calibration / calibration_test_fraction
    calibration_idx, test_idx = train_test_split(
        calibration_test_idx,
        train_size=calibration_fraction,
        random_state=seed,
        shuffle=True,
    )
    return DatasetSplits(
        train=sorted(train_idx),
        validation=sorted(validation_idx),
        calibration=sorted(calibration_idx),
        test=sorted(test_idx),
    )


def compute_label_distributions(frame: pd.DataFrame, config: ModelConfig) -> dict[str, Any]:
    distributions: dict[str, Any] = {
        "rows_total": int(len(frame)),
        "origin_distribution": _value_counts(frame["origin"]),
        "domain_distribution": _value_counts(frame["domain"]),
        "heads": {},
    }
    for head in config.heads:
        distributions["heads"][head.name] = {
            label: int(frame[f"{head.name}__{label}"].sum()) for label in head.labels
        }
    return distributions


def show_batch(batch: dict[str, Any], max_examples: int = 3) -> str:
    lines = []
    for index, text in enumerate(batch["texts"][:max_examples]):
        lines.append(f"{batch['ids'][index]}: {text}")
    return "\n".join(lines)


def _validate_head_targets(frame: pd.DataFrame, head: HeadConfig) -> None:
    columns = list(head.label_columns)
    if head.mode == "single_label":
        positives = frame[columns].sum(axis=1)
        invalid_count = int((positives != 1).sum())
        if invalid_count:
            raise ValueError(
                f"Head {head.name} is single_label but {invalid_count} rows do not have "
                "exactly one positive label"
            )


def _value_counts(series: pd.Series) -> dict[str, int]:
    return {str(key): int(value) for key, value in series.value_counts().sort_index().items()}

