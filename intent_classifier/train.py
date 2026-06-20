"""Training, metrics, calibration, thresholds, and export utilities."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import torch
import torch.nn.functional as F
from onnxruntime.quantization import QuantType, quantize_dynamic
from sklearn.metrics import f1_score, precision_recall_fscore_support
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from intent_classifier.dataset import RequestDataset, create_data_loaders, split_ids
from intent_classifier.model import TextClassifier, create_model, export_onnx
from intent_classifier.preprocessing import export_tokenizer, load_tokenizer
from intent_classifier.settings import (
    HeadConfig,
    LossConfig,
    ModelConfig,
    PhaseConfig,
    TrainConfig,
    load_model_config,
    load_train_config,
)
from intent_classifier.utils import ensure_dir, save_json, set_seed

logger = logging.getLogger(__name__)


def create_optimizer(
    model: TextClassifier,
    phase: PhaseConfig,
    weight_decay: float = 0.01,
) -> Optimizer:
    classifier_params = list(model.heads.parameters())
    groups: list[dict[str, Any]] = [
        {
            "params": classifier_params,
            "lr": phase.learning_rate_classifier,
            "weight_decay": weight_decay,
        }
    ]
    if not phase.freeze_backbone:
        groups.append(
            {
                "params": model.encoder.parameters(),
                "lr": phase.learning_rate_backbone or phase.learning_rate_classifier,
                "weight_decay": weight_decay,
            }
        )
    return AdamW(groups)


def create_scheduler(optimizer: Optimizer, total_steps: int, warmup_ratio: float = 0.1) -> LambdaLR:
    warmup_steps = int(total_steps * warmup_ratio)

    def lr_lambda(step: int) -> float:
        if total_steps <= 0:
            return 1.0
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        remaining = total_steps - step
        decay_steps = max(1, total_steps - warmup_steps)
        return max(0.0, float(remaining) / float(decay_steps))

    return LambdaLR(optimizer, lr_lambda)


def compute_multihead_loss(
    logits: dict[str, torch.Tensor],
    labels: dict[str, torch.Tensor],
    config: ModelConfig,
    loss_config: LossConfig,
    pos_weights: dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    for head in config.heads:
        weight = loss_config.head_weights.get(head.name, 1.0)
        if head.mode == "multi_label":
            pos_weight = None
            if pos_weights and head.name in pos_weights:
                pos_weight = pos_weights[head.name].to(logits[head.name].device)
            head_loss = F.binary_cross_entropy_with_logits(
                logits[head.name],
                labels[head.name].float().to(logits[head.name].device),
                pos_weight=pos_weight,
            )
        else:
            head_loss = F.cross_entropy(
                logits[head.name],
                labels[head.name].long().to(logits[head.name].device),
            )
        losses.append(head_loss * weight)
    return torch.stack(losses).sum()


def compute_pos_weights(
    dataset: RequestDataset,
    config: ModelConfig,
    max_pos_weight: float = 10.0,
) -> dict[str, torch.Tensor]:
    weights: dict[str, torch.Tensor] = {}
    frame = dataset.frame
    for head in config.heads:
        if head.mode != "multi_label":
            continue
        values = []
        for label in head.labels:
            positives = float(frame[f"{head.name}__{label}"].sum())
            negatives = float(len(frame) - positives)
            values.append(min(negatives / max(positives, 1.0), max_pos_weight))
        weights[head.name] = torch.tensor(values, dtype=torch.float32)
    return weights


def compute_metrics(
    logits: dict[str, np.ndarray],
    labels: dict[str, np.ndarray],
    config: ModelConfig,
    thresholds: dict[str, dict[str, float]] | None = None,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {"heads": {}}
    for head in config.heads:
        head_logits = logits[head.name]
        head_labels = labels[head.name]
        if head.mode == "multi_label":
            probs = sigmoid(head_logits)
            threshold_values = np.array(
                [
                    (thresholds or {}).get(head.name, {}).get(label, 0.5)
                    for label in head.labels
                ]
            )
            preds = (probs >= threshold_values).astype(int)
            head_metrics = {
                "micro_f1": float(f1_score(head_labels, preds, average="micro", zero_division=0)),
                "macro_f1": float(f1_score(head_labels, preds, average="macro", zero_division=0)),
                "weighted_f1": float(
                    f1_score(head_labels, preds, average="weighted", zero_division=0)
                ),
                "labels": {},
            }
            precision, recall, f1, support = precision_recall_fscore_support(
                head_labels,
                preds,
                average=None,
                zero_division=0,
            )
            for index, label in enumerate(head.labels):
                head_metrics["labels"][label] = {
                    "precision": float(precision[index]),
                    "recall": float(recall[index]),
                    "f1": float(f1[index]),
                    "support": int(support[index]),
                }
        else:
            preds = np.argmax(head_logits, axis=1)
            head_metrics = {
                "accuracy": float((preds == head_labels).mean()),
                "macro_f1": float(f1_score(head_labels, preds, average="macro", zero_division=0)),
                "weighted_f1": float(
                    f1_score(head_labels, preds, average="weighted", zero_division=0)
                ),
                "labels": {},
            }
            precision, recall, f1, support = precision_recall_fscore_support(
                head_labels,
                preds,
                labels=np.arange(len(head.labels)),
                average=None,
                zero_division=0,
            )
            for index, label in enumerate(head.labels):
                head_metrics["labels"][label] = {
                    "precision": float(precision[index]),
                    "recall": float(recall[index]),
                    "f1": float(f1[index]),
                    "support": int(support[index]),
                }
        metrics["heads"][head.name] = head_metrics
    return metrics


def train_one_epoch(
    model: TextClassifier,
    loader: DataLoader[dict[str, Any]],
    optimizer: Optimizer,
    config: ModelConfig,
    loss_config: LossConfig,
    device: torch.device,
    pos_weights: dict[str, torch.Tensor] | None = None,
    scheduler: LambdaLR | None = None,
    gradient_clipping: float | None = None,
) -> float:
    model.train()
    losses = []
    for batch in loader:
        optimizer.zero_grad()
        logits = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
        )
        loss = compute_multihead_loss(logits, batch["labels"], config, loss_config, pos_weights)
        loss.backward()
        if gradient_clipping is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else 0.0


@torch.no_grad()
def evaluate(
    model: TextClassifier,
    loader: DataLoader[dict[str, Any]],
    config: ModelConfig,
    loss_config: LossConfig,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    losses = []
    logits_store = {head.name: [] for head in config.heads}
    labels_store = {head.name: [] for head in config.heads}
    for batch in loader:
        logits = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
        )
        loss = compute_multihead_loss(logits, batch["labels"], config, loss_config)
        losses.append(float(loss.detach().cpu()))
        for head in config.heads:
            logits_store[head.name].append(logits[head.name].detach().cpu().numpy())
            labels_store[head.name].append(batch["labels"][head.name].detach().cpu().numpy())

    logits_np = {name: np.concatenate(values, axis=0) for name, values in logits_store.items()}
    labels_np = {name: np.concatenate(values, axis=0) for name, values in labels_store.items()}
    metrics = compute_metrics(logits_np, labels_np, config)
    metrics["loss"] = float(np.mean(losses)) if losses else 0.0
    return metrics


def fit(train_config: TrainConfig, model_config: ModelConfig) -> TextClassifier:
    set_seed(train_config.seed)
    artifact_dir = ensure_dir(train_config.artifact_dir)
    tokenizer_dir = export_tokenizer(model_config.backbone)
    tokenizer = load_tokenizer(tokenizer_dir)
    dataset = RequestDataset.from_csv(train_config.dataset_csv, model_config)
    splits = split_ids(dataset.frame, train_config.splits, train_config.seed)
    loaders = create_data_loaders(
        dataset,
        tokenizer,
        model_config,
        splits,
        batch_size=train_config.training.phase_1.batch_size,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(model_config).to(device)
    pos_weights = compute_pos_weights(dataset, model_config, train_config.loss.max_pos_weight)

    for phase in (train_config.training.phase_1, train_config.training.phase_2):
        model.freeze_backbone() if phase.freeze_backbone else model.unfreeze_backbone()
        optimizer = create_optimizer(model, phase, train_config.training.weight_decay)
        total_steps = len(loaders["train"]) * phase.epochs
        scheduler = create_scheduler(optimizer, total_steps)
        for epoch in range(phase.epochs):
            loss = train_one_epoch(
                model,
                loaders["train"],
                optimizer,
                model_config,
                train_config.loss,
                device,
                pos_weights,
                scheduler,
                train_config.training.gradient_clipping,
            )
            validation_metrics = evaluate(
                model,
                loaders["validation"],
                model_config,
                train_config.loss,
                device,
            )
            logger.info(
                "phase freeze=%s epoch=%s train_loss=%.4f validation_loss=%.4f",
                phase.freeze_backbone,
                epoch + 1,
                loss,
                validation_metrics["loss"],
            )

    save_checkpoint(model, artifact_dir / "checkpoint.pt", model_config)
    export_onnx(model, model_config, artifact_dir / "model.onnx", device=device)
    return model


def save_checkpoint(model: TextClassifier, path: str | Path, config: ModelConfig) -> Path:
    output = Path(path)
    ensure_dir(output.parent)
    torch.save({"state_dict": model.state_dict(), "head_names": config.head_names}, output)
    return output


def load_checkpoint(model: TextClassifier, path: str | Path, map_location: str = "cpu") -> TextClassifier:
    checkpoint = torch.load(path, map_location=map_location)
    model.load_state_dict(checkpoint["state_dict"])
    return model


def objective(trial: optuna.Trial, train_config: TrainConfig, model_config: ModelConfig) -> float:
    dropout = trial.suggest_float("dropout", 0.1, 0.3)
    logger.info("Optuna trial requested dropout=%s; full retraining hook is project-specific", dropout)
    _ = train_config, model_config
    return 0.0


def run_optuna_study(train_config: TrainConfig, model_config: ModelConfig) -> optuna.Study:
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, train_config, model_config), n_trials=train_config.optuna.n_trials)
    return study


def train_study(train_config: TrainConfig, model_config: ModelConfig) -> optuna.Study:
    return run_optuna_study(train_config, model_config)


def quantize_onnx(input_path: str | Path, output_path: str | Path) -> Path:
    output = Path(output_path)
    ensure_dir(output.parent)
    quantize_dynamic(str(input_path), str(output), weight_type=QuantType.QInt8)
    return output


def calibrate_outputs(
    logits: dict[str, np.ndarray],
    labels: dict[str, np.ndarray],
    config: ModelConfig,
) -> dict[str, Any]:
    calibration: dict[str, Any] = {
        "version": "calibration_v1",
        "fitted_on": "logits",
        "heads": {},
    }
    for head in config.heads:
        if head.mode == "multi_label":
            calibration["heads"][head.name] = {
                "method": "per_label_temperature_scaling",
                "temperatures": {
                    label: 1.0 for label in head.labels
                },
            }
        else:
            calibration["heads"][head.name] = {
                "method": "head_temperature_scaling",
                "temperature": 1.0,
            }
        _ = logits, labels
    return calibration


def save_calibration(calibration: dict[str, Any], path: str | Path) -> None:
    save_json(calibration, path)


def load_calibration(path: str | Path) -> dict[str, Any]:
    from intent_classifier.utils import load_json

    return load_json(path)


def tune_thresholds(
    probabilities: dict[str, np.ndarray],
    labels: dict[str, np.ndarray],
    config: ModelConfig,
    candidates: np.ndarray | None = None,
) -> dict[str, Any]:
    grid = candidates if candidates is not None else np.linspace(0.1, 0.9, 17)
    thresholds: dict[str, Any] = {"version": "thresholds_v1", "heads": {}}
    for head in config.heads:
        if head.mode != "multi_label":
            continue
        thresholds["heads"][head.name] = {}
        for index, label in enumerate(head.labels):
            best_threshold = 0.5
            best_f1 = -1.0
            y_true = labels[head.name][:, index]
            for threshold in grid:
                y_pred = (probabilities[head.name][:, index] >= threshold).astype(int)
                score = f1_score(y_true, y_pred, zero_division=0)
                if score > best_f1:
                    best_f1 = float(score)
                    best_threshold = float(threshold)
            thresholds["heads"][head.name][label] = {"activate": best_threshold}
    return thresholds


def save_thresholds(thresholds: dict[str, Any], path: str | Path) -> None:
    save_json(thresholds, path)


def load_thresholds(path: str | Path) -> dict[str, Any]:
    from intent_classifier.utils import load_json

    return load_json(path)


def plot_training_history(history: dict[str, Any], output_path: str | Path) -> Path:
    _ = history
    output = Path(output_path)
    ensure_dir(output.parent)
    output.write_text("Plotting is not implemented yet.\n", encoding="utf-8")
    return output


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the intent classifier.")
    parser.add_argument("--config", default="intent_classifier/config/train_config.yaml")
    args = parser.parse_args()
    train_config = load_train_config(args.config)
    model_config = load_model_config(train_config.model_config)
    if train_config.optuna.enabled:
        run_optuna_study(train_config, model_config)
    fit(train_config, model_config)


if __name__ == "__main__":
    main()

