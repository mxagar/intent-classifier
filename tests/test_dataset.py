import pandas as pd
import pytest

from intent_classifier.dataset import (
    RequestDataset,
    build_targets_for_row,
    compute_label_distributions,
    validate_dataset_frame,
)
from intent_classifier.settings import load_model_config


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "id": "1",
                "text": "haz presupuesto",
                "origin": "manual",
                "domain": "electricista",
                "business__create_budget": 1,
                "business__create_invoice": 0,
                "business__schedule_visit": 0,
                "business__cancel_visit": 0,
                "business__modify_visit": 0,
                "business__send_document": 0,
                "business__add_customer": 0,
                "business__update_customer": 0,
                "business__ask_price": 0,
                "business__ask_status": 0,
                "undesired__prompt_injection": 0,
                "undesired__abuse": 0,
                "undesired__spam": 0,
                "undesired__fraud_attempt": 0,
                "undesired__unsafe_data_request": 0,
                "undesired__unsupported_request": 0,
                "undesired__irrelevant_request": 0,
                "undesired__ambiguous_request": 0,
            }
        ]
    )


def test_validate_dataset_frame_and_targets() -> None:
    config = load_model_config("intent_classifier/config/model_config.yaml")
    frame = validate_dataset_frame(_frame(), config)
    targets = build_targets_for_row(frame.iloc[0], config)

    assert targets["business"].shape[0] == 10
    assert targets["business"][0].item() == 1
    assert RequestDataset(frame, config)[0]["text"] == "haz presupuesto"


def test_validate_dataset_frame_rejects_invalid_origin() -> None:
    config = load_model_config("intent_classifier/config/model_config.yaml")
    frame = _frame()
    frame.loc[0, "origin"] = "bad"

    with pytest.raises(ValueError, match="Invalid origin"):
        validate_dataset_frame(frame, config)


def test_compute_label_distributions() -> None:
    config = load_model_config("intent_classifier/config/model_config.yaml")
    distributions = compute_label_distributions(_frame(), config)

    assert distributions["rows_total"] == 1
    assert distributions["origin_distribution"]["manual"] == 1
    assert distributions["heads"]["business"]["create_budget"] == 1

