import numpy as np
import torch

from intent_classifier.settings import LossConfig, load_model_config
from intent_classifier.train import compute_metrics, compute_multihead_loss, tune_thresholds


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

