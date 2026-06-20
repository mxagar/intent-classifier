from types import SimpleNamespace

import torch
from torch import nn

from intent_classifier.model import TextClassifier
from intent_classifier.settings import load_model_config


class FakeEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=6)
        self.proj = nn.Linear(1, 6)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> SimpleNamespace:
        values = input_ids.float().unsqueeze(-1)
        return SimpleNamespace(last_hidden_state=self.proj(values))


def test_text_classifier_outputs_configured_heads(monkeypatch) -> None:
    monkeypatch.setattr("intent_classifier.model.AutoModel.from_pretrained", lambda *a, **k: FakeEncoder())
    config = load_model_config("intent_classifier/config/model_config.yaml")
    model = TextClassifier(config)

    outputs = model(
        input_ids=torch.ones((2, 4), dtype=torch.long),
        attention_mask=torch.ones((2, 4), dtype=torch.long),
    )

    assert set(outputs) == {"business", "undesired"}
    assert outputs["business"].shape == (2, 10)
    assert outputs["undesired"].shape == (2, 8)

