# Intent Classifier

Spanish multi-head intent classifier for chatbot routing. The current default setup uses two
multi-label heads:

- `business`
- `undesired`

The model architecture, dataset format, artifact structure, and training flow are described in
[SPEC.md](SPEC.md). The build order and current implementation milestones are in
[IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md).

## Setup

This project uses `uv`.

```bash
uv sync --extra dev
```

Activate the virtual environment:

```bash
source .venv/bin/activate
```

You can also skip manual activation and prefix commands with `uv run`.

Run the full local verification suite:

```bash
uv run nox
```

Run individual sessions:

```bash
uv run nox -s lint
uv run nox -s typecheck
uv run nox -s tests
```

The example dataset is:

```text
dataset/example_dataset.csv
```

The default training config is:

```text
intent_classifier/config/train_config.yaml
```

The default model config is:

```text
intent_classifier/config/model_config.yaml
```

Example head definitions (extended in `model_config.yaml`):

```yaml
heads:
  - name: business
    mode: multi_label
    labels:
      - create_budget
      - create_invoice
      - schedule_visit
      - cancel_visit
      - modify_visit
      - send_document
      - add_customer
      - update_customer
      - ask_price
      - ask_status

  - name: undesired
    mode: multi_label
    labels:
      - prompt_injection
      - abuse
      - spam
      - fraud_attempt
      - unsafe_data_request
      - unsupported_request
      - irrelevant_request
      - ambiguous_request
```

## Training

Run the default training command:

```bash
uv run python -m intent_classifier.train \
  --settings intent_classifier/config/settings.yaml
```

Run hyperparameter optimization:

```bash
uv run python -m intent_classifier.train \
  --settings intent_classifier/config/settings.yaml \
  --hpo
```

Train a final model from a saved HPO study:

```bash
uv run python -m intent_classifier.train \
  --settings intent_classifier/config/settings.yaml \
  --study-json intent_classifier/artifacts/hpo/<run_timestamp>/study.json
```

## Artifacts

Default model artifacts are written to:

```text
intent_classifier/artifacts/v1/
```

Expected files include:

```text
checkpoint.pt
model.onnx
model.int8.onnx
model_config.yaml
train_config.yaml
calibration.json
thresholds.json
evaluation_report.json
training_history.json
training_history.png
tokenizer/
```

HPO runs are written under:

```text
intent_classifier/artifacts/hpo/<run_timestamp>/
```

Release metadata is tracked in:

```text
intent_classifier/artifacts/changelog.yaml
```

## Inference

Once a trained artifact directory exists, run inference with `IntentEstimator`:

```python
from intent_classifier.inference import IntentEstimator

estimator = IntentEstimator("intent_classifier/artifacts/v1")

prediction = estimator.predict(
    "hazme un presupuesto y agenda una visita para mañana"
)

for head_name, head_prediction in prediction.items():
    print(head_name)
    print("probabilities:", head_prediction.probabilities)
    print("active labels:", head_prediction.active_labels)
```

`estimator.predict(...)` returns a dictionary keyed by head name. The values are
`HeadPrediction` objects:

```python
{
    "business": HeadPrediction(
        mode="multi_label",
        probabilities={
            "create_budget": 0.91,
            "create_invoice": 0.04,
            "schedule_visit": 0.86,
            "cancel_visit": 0.01,
            "modify_visit": 0.03,
            "send_document": 0.02,
            "add_customer": 0.01,
            "update_customer": 0.01,
            "ask_price": 0.12,
            "ask_status": 0.05,
        },
        active_labels=["create_budget", "schedule_visit"],
    ),
    "undesired": HeadPrediction(
        mode="multi_label",
        probabilities={
            "prompt_injection": 0.01,
            "abuse": 0.01,
            "spam": 0.02,
            "fraud_attempt": 0.01,
            "unsafe_data_request": 0.01,
            "unsupported_request": 0.04,
            "irrelevant_request": 0.02,
            "ambiguous_request": 0.08,
        },
        active_labels=[],
    ),
}
```

Production inference loads the tokenizer from the artifact directory with local files only, so it
does not depend on Hugging Face connectivity at runtime.

## Authorship

Mikel Sagardia, 2026.  
No guarantees.
