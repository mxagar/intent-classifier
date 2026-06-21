"""Training, HPO, metrics, calibration, thresholds, and export utilities."""

import argparse
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import torch
import torch.nn.functional as F
import yaml
from optuna.distributions import distribution_to_json, json_to_distribution
from pydantic import BaseModel, ConfigDict
from sklearn.metrics import f1_score, precision_recall_fscore_support
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import PreTrainedTokenizerBase

from intent_classifier.dataset import (
    RequestDataset,
    compute_label_distributions,
    create_data_loaders,
    split_ids,
)
from intent_classifier.model import TextClassifier, create_model, export_onnx
from intent_classifier.preprocessing import export_tokenizer, load_tokenizer
from intent_classifier.settings import (
    HeadMode,
    LossConfig,
    ModelConfig,
    PhaseConfig,
    SettingsConfig,
    TrainConfig,
    load_model_config,
    load_settings_config,
    load_train_config,
)
from intent_classifier.utils import ensure_dir, load_json, save_json, set_seed, sha256_file

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainingInputs:
    train_config: TrainConfig
    model_config: ModelConfig
    model: TextClassifier
    tokenizer: PreTrainedTokenizerBase
    dataset: RequestDataset
    loaders: dict[str, DataLoader[dict[str, Any]]]
    device: torch.device

    @classmethod
    def from_settings(
        cls,
        settings: SettingsConfig | None = None,
        device: torch.device | None = None,
    ) -> "TrainingInputs":
        resolved_settings = settings or load_settings_config()
        train_config = load_train_config(resolved_settings.train_config_path)
        model_config = load_model_config(resolved_settings.model_config_path)
        return cls.from_configs(train_config, model_config, device=device)

    @classmethod
    def from_configs(
        cls,
        train_config: TrainConfig,
        model_config: ModelConfig,
        device: torch.device | None = None,
    ) -> "TrainingInputs":
        set_seed(train_config.seed)
        artifact_dir = ensure_dir(train_config.current_artifact_dir)
        tokenizer_dir = export_tokenizer(
            model_config.backbone,
            artifact_dir / model_config.backbone.local_tokenizer_path,
        )
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
        resolved_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = create_model(model_config).to(resolved_device)
        return cls(
            train_config=train_config,
            model_config=model_config,
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            loaders=loaders,
            device=resolved_device,
        )

    @classmethod
    def from_objects(
        cls,
        train_config: TrainConfig,
        model_config: ModelConfig,
        model: TextClassifier,
        tokenizer: PreTrainedTokenizerBase,
        dataset: RequestDataset,
        loaders: dict[str, DataLoader[dict[str, Any]]],
        device: torch.device,
    ) -> "TrainingInputs":
        return cls(
            train_config=train_config,
            model_config=model_config,
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            loaders=loaders,
            device=device,
        )


class TrainingOutputs(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: TextClassifier
    history: dict[str, Any]
    calibration: dict[str, Any]
    thresholds: dict[str, Any]
    validation: dict[str, Any]
    test: dict[str, Any]
    artifact_dir: Path | None = None
    checkpoint_path: Path | None = None
    model_config_path: Path | None = None
    train_config_path: Path | None = None
    calibration_path: Path | None = None
    thresholds_path: Path | None = None
    evaluation_report_path: Path | None = None
    training_history_path: Path | None = None
    training_history_plot_path: Path | None = None
    onnx_path: Path | None = None
    quantized_onnx_path: Path | None = None


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
        if head.mode == HeadMode.MULTI_LABEL:
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
    """Compute BCE positive-label weights to reduce rare-label under-training.

    Multi-label datasets are usually sparse: most labels are negative in most rows. For each
    multi-label output, `pos_weight = negatives / positives` tells BCEWithLogitsLoss to penalize
    missed positives more heavily. The cap keeps rare labels from destabilizing training.
    """
    weights: dict[str, torch.Tensor] = {}
    frame = dataset.frame
    for head in config.heads:
        if head.mode != HeadMode.MULTI_LABEL:
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
        if head.mode == HeadMode.MULTI_LABEL:
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
    metrics["mean_macro_f1"] = float(
        np.mean([head_metrics["macro_f1"] for head_metrics in metrics["heads"].values()])
    )
    metrics["mean_weighted_f1"] = float(
        np.mean([head_metrics["weighted_f1"] for head_metrics in metrics["heads"].values()])
    )
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
def collect_logits_and_labels(
    model: TextClassifier,
    loader: DataLoader[dict[str, Any]],
    config: ModelConfig,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    model.eval()
    logits_store = {head.name: [] for head in config.heads}
    labels_store = {head.name: [] for head in config.heads}
    for batch in loader:
        logits = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
        )
        for head in config.heads:
            logits_store[head.name].append(logits[head.name].detach().cpu().numpy())
            labels_store[head.name].append(batch["labels"][head.name].detach().cpu().numpy())
    return (
        {name: np.concatenate(values, axis=0) for name, values in logits_store.items()},
        {name: np.concatenate(values, axis=0) for name, values in labels_store.items()},
    )


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
    for batch in loader:
        logits = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
        )
        loss = compute_multihead_loss(logits, batch["labels"], config, loss_config)
        losses.append(float(loss.detach().cpu()))

    logits_np, labels_np = collect_logits_and_labels(model, loader, config, device)
    metrics = compute_metrics(logits_np, labels_np, config)
    metrics["loss"] = float(np.mean(losses)) if losses else 0.0
    return metrics


def fit(
    model_config: ModelConfig | None = None,
    train_config: TrainConfig | None = None,
    training_inputs: TrainingInputs | None = None,
) -> TrainingOutputs:
    objects = _resolve_training_inputs(model_config, train_config, training_inputs)
    return fit_with_history(
        objects,
        export_artifacts=True,
    )


def _resolve_training_inputs(
    model_config: ModelConfig | None,
    train_config: TrainConfig | None,
    training_inputs: TrainingInputs | None,
) -> TrainingInputs:
    if training_inputs is not None:
        return training_inputs
    if model_config is None or train_config is None:
        raise ValueError("Pass either training_inputs or both model_config and train_config")
    return TrainingInputs.from_configs(train_config, model_config)


def fit_with_history(
    training_inputs: TrainingInputs,
    export_artifacts: bool,
) -> TrainingOutputs:
    model = training_inputs.model
    dataset = training_inputs.dataset
    loaders = training_inputs.loaders
    train_config = training_inputs.train_config
    model_config = training_inputs.model_config
    set_seed(train_config.seed)
    artifact_dir = ensure_dir(train_config.current_artifact_dir)
    resolved_device = training_inputs.device
    model = model.to(resolved_device)
    pos_weights = (
        compute_pos_weights(dataset, model_config, train_config.loss.max_pos_weight)
        if train_config.loss.use_pos_weights
        else None
    )

    epochs: list[dict[str, Any]] = []
    best_validation: dict[str, Any] | None = None
    total_epochs = train_config.training.phase_1.epochs + train_config.training.phase_2.epochs
    progress = tqdm(total=total_epochs, desc="Training", unit="epoch")
    global_epoch = 0
    for phase in (train_config.training.phase_1, train_config.training.phase_2):
        model.freeze_backbone() if phase.freeze_backbone else model.unfreeze_backbone()
        optimizer = create_optimizer(model, phase, train_config.training.weight_decay)
        total_steps = len(loaders["train"]) * phase.epochs
        scheduler = create_scheduler(optimizer, total_steps)
        for epoch in range(phase.epochs):
            global_epoch += 1
            loss = train_one_epoch(
                model,
                loaders["train"],
                optimizer,
                model_config,
                train_config.loss,
                resolved_device,
                pos_weights,
                scheduler,
                train_config.training.gradient_clipping,
            )
            validation_metrics = evaluate(
                model,
                loaders["validation"],
                model_config,
                train_config.loss,
                resolved_device,
            )
            best_validation = validation_metrics
            epochs.append(
                {
                    "freeze_backbone": phase.freeze_backbone,
                    "epoch": epoch + 1,
                    "global_epoch": global_epoch,
                    "train_loss": loss,
                    "validation": validation_metrics,
                }
            )
            progress.set_postfix(
                {
                    "train_loss": f"{loss:.4f}",
                    "val_loss": f"{validation_metrics['loss']:.4f}",
                    "val_macro_f1": f"{validation_metrics['mean_macro_f1']:.4f}",
                }
            )
            progress.update(1)
            logger.info(
                "phase freeze=%s epoch=%s train_loss=%.4f validation_loss=%.4f",
                phase.freeze_backbone,
                epoch + 1,
                loss,
                validation_metrics["loss"],
            )
    progress.close()

    validation_logits, validation_labels = collect_logits_and_labels(
        model,
        loaders["validation"],
        model_config,
        resolved_device,
    )
    validation_probs = {
        head_name: sigmoid(values) for head_name, values in validation_logits.items()
    }
    calibration = calibrate_outputs(validation_logits, validation_labels, model_config)
    thresholds = tune_thresholds(validation_probs, validation_labels, model_config)
    test_metrics = evaluate(model, loaders["test"], model_config, train_config.loss, resolved_device)
    history = {
        "epochs": epochs,
        "validation": best_validation or {},
        "test": test_metrics,
        "dataset": {
            "csv_path": str(train_config.dataset_csv),
            "csv_sha256": sha256_file(train_config.dataset_csv),
            **compute_label_distributions(dataset.frame, model_config),
            "split_rows": {
                "train": len(loaders["train"].dataset),
                "validation": len(loaders["validation"].dataset),
                "test": len(loaders["test"].dataset),
            },
        },
    }

    outputs = TrainingOutputs(
        model=model,
        history=history,
        calibration=calibration,
        thresholds=thresholds,
        validation=history["validation"],
        test=history["test"],
        artifact_dir=artifact_dir,
    )

    if export_artifacts:
        checkpoint_path = save_checkpoint(model, artifact_dir / "checkpoint.pt", model_config)
        save_runtime_configs(model_config, train_config, artifact_dir)
        calibration_path = artifact_dir / "calibration.json"
        thresholds_path = artifact_dir / "thresholds.json"
        evaluation_report_path = artifact_dir / "evaluation_report.json"
        training_history_path = artifact_dir / "training_history.json"
        training_history_plot_path = artifact_dir / "training_history.png"
        save_calibration(calibration, calibration_path)
        save_thresholds(thresholds, thresholds_path)
        save_json(history, evaluation_report_path)
        save_json(history, training_history_path)
        plot_training_history(history, training_history_plot_path)
        outputs = outputs.model_copy(
            update={
                "checkpoint_path": checkpoint_path,
                "model_config_path": artifact_dir / "model_config.yaml",
                "train_config_path": artifact_dir / "train_config.yaml",
                "calibration_path": calibration_path,
                "thresholds_path": thresholds_path,
                "evaluation_report_path": evaluation_report_path,
                "training_history_path": training_history_path,
                "training_history_plot_path": training_history_plot_path,
            }
        )
        update_changelog(train_config.artifacts_root / "changelog.yaml", artifact_dir, history)
    return outputs


def save_checkpoint(model: TextClassifier, path: str | Path, config: ModelConfig) -> Path:
    output = Path(path)
    ensure_dir(output.parent)
    torch.save({"state_dict": model.state_dict(), "head_names": config.head_names}, output)
    return output


def load_checkpoint(model: TextClassifier, path: str | Path, map_location: str = "cpu") -> TextClassifier:
    checkpoint = torch.load(path, map_location=map_location)
    model.load_state_dict(checkpoint["state_dict"])
    return model


def objective(
    trial: optuna.Trial,
    train_config: TrainConfig,
    model_config: ModelConfig,
    hpo_dir: str | Path,
) -> float:
    trial_train_config, trial_model_config = build_trial_configs(
        trial,
        train_config,
        model_config,
        hpo_dir,
    )
    training_inputs = TrainingInputs.from_configs(trial_train_config, trial_model_config)
    outputs = fit_with_history(
        training_inputs,
        export_artifacts=False,
    )
    history = outputs.history
    metric_name = train_config.optuna.metric
    value = float(history["validation"].get(metric_name, history["validation"].get("mean_macro_f1", 0.0)))
    trial.set_user_attr("train_config", trial_train_config.model_dump(mode="json"))
    trial.set_user_attr("model_config", trial_model_config.model_dump(mode="json"))
    return value


def build_trial_configs(
    trial: optuna.Trial,
    train_config: TrainConfig,
    model_config: ModelConfig,
    hpo_dir: str | Path | None = None,
) -> tuple[TrainConfig, ModelConfig]:
    space = train_config.optuna.search_space
    trial_number = int(getattr(trial, "number", 0))
    dropout = trial.suggest_float("dropout", *space.dropout)
    max_length = trial.suggest_categorical("max_length", list(space.max_length))
    batch_size = trial.suggest_categorical("batch_size", list(space.batch_size))
    weight_decay = trial.suggest_float("weight_decay", *space.weight_decay)
    lr_backbone = trial.suggest_float("learning_rate_backbone", *space.learning_rate_backbone, log=True)
    lr_classifier = trial.suggest_float(
        "learning_rate_classifier",
        *space.learning_rate_classifier,
        log=True,
    )
    max_pos_weight = trial.suggest_float("max_pos_weight", *space.max_pos_weight)
    head_updates = []
    for head in model_config.heads:
        hidden_size = trial.suggest_categorical(
            f"{head.name}_hidden_size",
            list(space.head_hidden_sizes),
        )
        head_updates.append(
            head.model_copy(
                update={"hidden_layer": head.hidden_layer.model_copy(update={"size": hidden_size})}
            )
        )
    trial_model_config = model_config.model_copy(
        update={
            "classifier": model_config.classifier.model_copy(update={"dropout": dropout}),
            "text": model_config.text.model_copy(update={"max_length": max_length}),
            "heads": tuple(head_updates),
        }
    )
    phase_1 = train_config.training.phase_1.model_copy(
        update={"batch_size": batch_size, "learning_rate_classifier": lr_classifier}
    )
    phase_2 = train_config.training.phase_2.model_copy(
        update={
            "batch_size": batch_size,
            "learning_rate_classifier": lr_classifier,
            "learning_rate_backbone": lr_backbone,
        }
    )
    trial_train_config = train_config.model_copy(
        update={
            "artifact_dir": Path(hpo_dir or train_config.artifacts_root) / "artifacts",
            "version_dir": f"trial_{trial_number:04d}",
            "training": train_config.training.model_copy(
                update={
                    "phase_1": phase_1,
                    "phase_2": phase_2,
                    "weight_decay": weight_decay,
                }
            ),
            "loss": train_config.loss.model_copy(update={"max_pos_weight": max_pos_weight}),
        }
    )
    return trial_train_config, trial_model_config


def run_optuna_study(train_config: TrainConfig, model_config: ModelConfig) -> optuna.Study:
    hpo_dir = create_hpo_run_dir(train_config.artifacts_root)
    study = optuna.create_study(direction="maximize")
    study.set_user_attr("hpo_dir", str(hpo_dir))
    study.optimize(
        lambda trial: objective(trial, train_config, model_config, hpo_dir),
        n_trials=train_config.optuna.n_trials,
    )
    save_study(study, hpo_dir / "study.json")
    return study


def save_study(study: optuna.Study, path: str | Path) -> None:
    payload = {
        "study_name": study.study_name,
        "direction": study.direction.name,
        "best_trial_number": study.best_trial.number if study.trials else None,
        "best_params": study.best_params if study.trials else {},
        "best_value": study.best_value if study.trials else None,
        "user_attrs": study.user_attrs,
        "trials": [_trial_to_json(trial) for trial in study.trials],
    }
    save_json(payload, path)


def load_study(path: str | Path) -> optuna.Study:
    payload = load_json(path)
    direction = payload.get("direction", "MAXIMIZE").lower()
    study = optuna.create_study(direction=direction)
    for key, value in payload.get("user_attrs", {}).items():
        study.set_user_attr(key, value)
    for trial_payload in payload.get("trials", []):
        study.add_trial(_trial_from_json(trial_payload))
    return study


def train_study(
    study_json: str | Path,
    train_config: TrainConfig,
    model_config: ModelConfig,
) -> TrainingOutputs:
    study_data = load_json(study_json)
    params = study_data.get("best_params")
    if not params:
        raise ValueError(f"No best_params found in study JSON: {study_json}")
    fixed_trial = optuna.trial.FixedTrial(params)
    final_train_config, final_model_config = build_trial_configs(fixed_trial, train_config, model_config)
    final_train_config = final_train_config.model_copy(
        update={
            "artifact_dir": train_config.artifact_dir,
            "version_dir": train_config.version_dir,
        }
    )
    training_inputs = TrainingInputs.from_configs(final_train_config, final_model_config)
    outputs = fit(training_inputs=training_inputs)
    onnx_path, quantized_onnx_path = export_runtime_onnx(training_inputs, quantize=True)
    return outputs.model_copy(
        update={"onnx_path": onnx_path, "quantized_onnx_path": quantized_onnx_path}
    )


def export_runtime_onnx(
    training_inputs: TrainingInputs,
    quantize: bool = True,
) -> tuple[Path, Path | None]:
    artifact_dir = ensure_dir(training_inputs.train_config.current_artifact_dir)
    onnx_path = artifact_dir / "model.onnx"
    quantized_onnx_path = artifact_dir / "model.int8.onnx" if quantize else None
    onnx_path = export_onnx(
        training_inputs.model,
        training_inputs.model_config,
        onnx_path,
        device=training_inputs.device,
        quantize=quantize,
        quantized_output_path=quantized_onnx_path,
    )
    return onnx_path, quantized_onnx_path


def quantize_onnx(input_path: str | Path, output_path: str | Path) -> Path:
    from onnxruntime.quantization import QuantType, quantize_dynamic

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
        "fitted_on": "validation_logits",
        "heads": {},
    }
    for head in config.heads:
        if head.mode == HeadMode.MULTI_LABEL:
            calibration["heads"][head.name] = {
                "method": "per_label_temperature_scaling",
                "temperatures": {label: 1.0 for label in head.labels},
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
        if head.mode != HeadMode.MULTI_LABEL:
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
    return load_json(path)


def plot_training_history(history: dict[str, Any], output_path: str | Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output_path)
    ensure_dir(output.parent)
    epochs = history.get("epochs", [])
    if not epochs:
        output.write_text("No epoch history available.\n", encoding="utf-8")
        return output

    x_values = [entry.get("global_epoch", index + 1) for index, entry in enumerate(epochs)]
    train_loss = [entry["train_loss"] for entry in epochs]
    validation_loss = [entry["validation"]["loss"] for entry in epochs]
    validation_macro_f1 = [entry["validation"]["mean_macro_f1"] for entry in epochs]
    validation_weighted_f1 = [entry["validation"]["mean_weighted_f1"] for entry in epochs]

    figure, (loss_axis, metric_axis) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    loss_axis.plot(x_values, train_loss, marker="o", label="train loss")
    loss_axis.plot(x_values, validation_loss, marker="o", label="validation loss")
    loss_axis.set_ylabel("Loss")
    loss_axis.grid(True, alpha=0.3)
    loss_axis.legend()

    metric_axis.plot(x_values, validation_macro_f1, marker="o", label="validation macro F1")
    metric_axis.plot(x_values, validation_weighted_f1, marker="o", label="validation weighted F1")
    metric_axis.set_xlabel("Epoch")
    metric_axis.set_ylabel("F1")
    metric_axis.set_ylim(0.0, 1.0)
    metric_axis.grid(True, alpha=0.3)
    metric_axis.legend()

    figure.suptitle("Training History")
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)
    return output


def create_hpo_run_dir(artifacts_root: str | Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return ensure_dir(Path(artifacts_root) / "hpo" / timestamp)


def make_hpo_artifact_dir(hpo_dir: str | Path, trial_number: int) -> Path:
    return Path(hpo_dir) / "artifacts" / f"trial_{trial_number:04d}"


def next_version_artifact_dir(artifacts_root: str | Path) -> Path:
    root = ensure_dir(artifacts_root)
    versions = [
        int(path.name[1:])
        for path in root.iterdir()
        if path.is_dir() and path.name.startswith("v") and path.name[1:].isdigit()
    ]
    next_version = max(versions, default=0) + 1
    return root / f"v{next_version}"


def save_runtime_configs(
    model_config: ModelConfig,
    train_config: TrainConfig,
    artifact_dir: str | Path,
) -> None:
    output = ensure_dir(artifact_dir)
    with (output / "model_config.yaml").open("w", encoding="utf-8") as file:
        yaml.safe_dump(model_config.model_dump(mode="json"), file, sort_keys=False, allow_unicode=True)
    with (output / "train_config.yaml").open("w", encoding="utf-8") as file:
        yaml.safe_dump(train_config.model_dump(mode="json"), file, sort_keys=False, allow_unicode=True)


def update_changelog(changelog_path: str | Path, artifact_dir: str | Path, report: dict[str, Any]) -> None:
    path = Path(changelog_path)
    if path.exists():
        with path.open("r", encoding="utf-8") as file:
            changelog = yaml.safe_load(file) or {}
    else:
        changelog = {}
    version = Path(artifact_dir).name
    changelog[version] = {
        "artifact_dir": str(artifact_dir),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": report.get("dataset", {}),
        "metrics": report.get("test", {}),
    }
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(changelog, file, sort_keys=False, allow_unicode=True)


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def _trial_to_json(trial: optuna.trial.FrozenTrial) -> dict[str, Any]:
    return {
        "number": trial.number,
        "state": trial.state.name,
        "value": trial.value,
        "params": trial.params,
        "distributions": {
            name: distribution_to_json(distribution)
            for name, distribution in trial.distributions.items()
        },
        "user_attrs": trial.user_attrs,
    }


def _trial_from_json(payload: dict[str, Any]) -> optuna.trial.FrozenTrial:
    distributions = {
        name: json_to_distribution(distribution_json)
        for name, distribution_json in payload.get("distributions", {}).items()
    }
    return optuna.trial.create_trial(
        params=payload.get("params", {}),
        distributions=distributions,
        value=payload.get("value"),
        user_attrs=payload.get("user_attrs", {}),
        state=getattr(optuna.trial.TrialState, payload.get("state", "COMPLETE")),
    )


def main() -> None:
    """Run training or HPO from the command line.

    Examples:
        uv run python -m intent_classifier.train --settings intent_classifier/config/settings.yaml
        uv run python -m intent_classifier.train --settings intent_classifier/config/settings.yaml --hpo
        uv run python -m intent_classifier.train --settings intent_classifier/config/settings.yaml --study-json intent_classifier/artifacts/hpo/20260101_120000/study.json
    """
    parser = argparse.ArgumentParser(description="Train the intent classifier.")
    parser.add_argument("--settings", default="intent_classifier/config/settings.yaml")
    parser.add_argument("--hpo", action="store_true", help="Run Optuna HPO instead of normal training.")
    parser.add_argument(
        "--study-json",
        default=None,
        help="Train final model using the best params from a saved study JSON.",
    )
    parser.add_argument(
        "--artifact-dir",
        default=None,
        help="Override artifact directory. Use this for explicit vN output paths.",
    )
    args = parser.parse_args()
    settings = load_settings_config(args.settings)
    train_config = load_train_config(settings.train_config_path)
    model_config = load_model_config(settings.model_config_path)
    if args.artifact_dir:
        train_config = train_config.model_copy(update={"artifact_dir": Path(args.artifact_dir)})
    if args.hpo:
        run_optuna_study(train_config, model_config)
    elif args.study_json:
        train_study(args.study_json, train_config, model_config)
    else:
        training_inputs = TrainingInputs.from_configs(train_config, model_config)
        outputs = fit(training_inputs=training_inputs)
        onnx_path, quantized_onnx_path = export_runtime_onnx(training_inputs, quantize=True)
        _ = outputs.model_copy(
            update={"onnx_path": onnx_path, "quantized_onnx_path": quantized_onnx_path}
        )


if __name__ == "__main__":
    main()
