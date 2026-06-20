"""Standalone evaluation entry point."""


import argparse
import logging
from pathlib import Path

from intent_classifier.settings import load_model_config, load_train_config
from intent_classifier.train import evaluate as evaluate_model
from intent_classifier.utils import save_json

logger = logging.getLogger(__name__)


def run_evaluation(config_path: str | Path) -> dict[str, object]:
    """Placeholder CLI hook for full artifact evaluation.

    Full evaluation needs a trained checkpoint and dataset. The reusable metric function lives in
    `intent_classifier.train.evaluate`.
    """
    train_config = load_train_config(config_path)
    model_config = load_model_config(train_config.model_config_path)
    report = {
        "status": "not_run",
        "reason": "Load a trained checkpoint and data loader to call train.evaluate.",
        "heads": list(model_config.head_names),
    }
    save_json(report, train_config.artifact_dir / "evaluation_report.json")
    _ = evaluate_model
    return report


def main() -> None:
    """Run standalone artifact evaluation.

    Example:
        uv run python -m intent_classifier.evaluate --config intent_classifier/config/train_config.yaml
    """
    parser = argparse.ArgumentParser(description="Evaluate trained intent classifier artifacts.")
    parser.add_argument("--config", default="intent_classifier/config/train_config.yaml")
    args = parser.parse_args()
    report = run_evaluation(args.config)
    logger.info("Evaluation report status: %s", report["status"])


if __name__ == "__main__":
    main()
