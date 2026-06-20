import numpy as np
import optuna
import torch

from intent_classifier.settings import LossConfig, load_model_config, load_train_config
from intent_classifier.train import (
    compute_metrics,
    compute_multihead_loss,
    load_study,
    save_study,
    tune_thresholds,
)


def test_compute_multihead_loss() -> None:
    config = load_model_config("intent_classifier/config/model_config.yaml")
    logits = {
        "business": torch.zeros((2, 10)),
        "undesired": torch.zeros((2, 8)),
    }
    labels = {
        "business": torch.zeros((2, 10)),
        "undesired": torch.zeros((2, 8)),
    }

    loss = compute_multihead_loss(
        logits,
        labels,
        config,
        LossConfig(head_weights={"business": 1.0, "undesired": 1.5}),
    )

    assert loss.item() > 0


def test_compute_metrics_and_thresholds() -> None:
    config = load_model_config("intent_classifier/config/model_config.yaml")
    logits = {
        "business": np.array([[5.0] + [-5.0] * 9, [-5.0] * 10]),
        "undesired": np.array([[-5.0] * 8, [5.0] + [-5.0] * 7]),
    }
    labels = {
        "business": np.array([[1] + [0] * 9, [0] * 10]),
        "undesired": np.array([[0] * 8, [1] + [0] * 7]),
    }

    metrics = compute_metrics(logits, labels, config)
    thresholds = tune_thresholds(
        {name: 1 / (1 + np.exp(-value)) for name, value in logits.items()},
        labels,
        config,
        candidates=np.array([0.5]),
    )

    assert metrics["heads"]["business"]["micro_f1"] == 1.0
    assert thresholds["heads"]["business"]["create_budget"]["activate"] == 0.5


def test_train_config_has_no_calibration_split() -> None:
    train_config = load_train_config("intent_classifier/config/train_config.yaml")

    assert train_config.splits.train == 0.8
    assert train_config.splits.validation == 0.1
    assert train_config.splits.test == 0.1
    assert not hasattr(train_config.splits, "calibration")


def test_save_and_load_study_json(tmp_path) -> None:
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: trial.suggest_float("x", 0.0, 1.0), n_trials=1)
    path = tmp_path / "study.json"

    save_study(study, path)
    loaded = load_study(path)

    assert len(loaded.trials) == 1
    assert loaded.best_params.keys() == study.best_params.keys()
