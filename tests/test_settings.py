from pathlib import Path

import pytest

from intent_classifier.settings import load_model_config


def test_load_model_config_default() -> None:
    config = load_model_config("intent_classifier/config/model_config.yaml")

    assert config.backbone.name == "microsoft/Multilingual-MiniLM-L12-H384"
    assert config.head_names == ("business", "undesired")
    assert config.head_by_name("business").hidden_layer.size == 256
    assert "business__create_budget" in config.label_columns


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
      activation: gelu
    labels: [a]
  - name: business
    mode: multi_label
    hidden_layer:
      size: 4
      activation: gelu
    labels: [b]
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Duplicate head names"):
        load_model_config(path)

