from pathlib import Path

import pytest

from intent_classifier.settings import load_model_config, load_settings_config, load_train_config


def test_load_settings_config_default_paths() -> None:
    config = load_settings_config("intent_classifier/config/settings.yaml")

    assert config.model_config_path == Path("intent_classifier/config/model_config.yaml")
    assert config.train_config_path == Path("intent_classifier/config/train_config.yaml")


def test_load_model_config_default() -> None:
    config = load_model_config("intent_classifier/config/model_config.yaml")

    assert config.backbone.name == "microsoft/Multilingual-MiniLM-L12-H384"
    assert config.backbone.local_tokenizer_path == Path("tokenizer")
    assert config.head_names == ("business", "undesired")
    assert config.head_by_name("business").hidden_layer.size == 256
    assert "business__create_budget" in config.label_columns


def test_load_train_config_resolves_versioned_artifact_dir() -> None:
    config = load_train_config("intent_classifier/config/train_config.yaml")

    assert config.artifact_dir == Path("intent_classifier/artifacts")
    assert config.version_dir == Path("v1")
    assert config.current_artifact_dir == Path("intent_classifier/artifacts/v1")


def test_load_model_config_rejects_duplicate_head_names(tmp_path: Path) -> None:
    path = tmp_path / "model_config.yaml"
    path.write_text(
        """
backbone:
  name: test
  revision: main
  local_tokenizer_path: tokenizer
text:
  max_length: 8
classifier:
  dropout: 0.1
heads:
  - name: business
    mode: multi_label
    hidden_layer:
      size: 4
      activation: relu
    labels: [a]
  - name: business
    mode: multi_label
    hidden_layer:
      size: 4
      activation: relu
    labels: [b]
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Duplicate head names"):
        load_model_config(path)
