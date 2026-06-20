import numpy as np

from intent_classifier.inference import postprocess_predictions
from intent_classifier.settings import load_model_config


def test_postprocess_predictions_applies_thresholds_and_calibration() -> None:
    config = load_model_config("intent_classifier/config/model_config.yaml")
    logits = {
        "business": np.array([[3.0] + [-3.0] * 9]),
        "undesired": np.array([[-3.0] * 8]),
    }
    calibration = {
        "heads": {
            "business": {
                "temperatures": {"create_budget": 1.0},
            }
        }
    }
    thresholds = {
        "heads": {
            "business": {
                "create_budget": {"activate": 0.5},
            }
        }
    }

    predictions = postprocess_predictions(logits, config, calibration, thresholds)

    assert predictions["business"].active_labels == ["create_budget"]
    assert predictions["undesired"].active_labels == []

