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

Run tests:

```bash
uv run pytest -q
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

## Training

Run the default training command:

```bash
uv run python -m intent_classifier.train \
  --config intent_classifier/config/train_config.yaml
```

Run hyperparameter optimization:

```bash
uv run python -m intent_classifier.train \
  --config intent_classifier/config/train_config.yaml \
  --hpo
```

Train a final model from a saved HPO study:

```bash
uv run python -m intent_classifier.train \
  --config intent_classifier/config/train_config.yaml \
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

Production inference loads the tokenizer from the artifact directory with local files only, so it
does not depend on Hugging Face connectivity at runtime.

## Authorship

Mikel Sagardia, 2026.  
No guarantees.

