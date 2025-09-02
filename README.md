# radlab-ml-utils

Lightweight utilities to streamline machine learning experiments and training pipelines.
Focus areas:
- Weights & Biases (W&B) integration: easy run initialization (tags, merged configs, timestamped names), 
logging of datasets/models as artifacts, scalar metrics, confusion matrices, and prediction tables.
- Training and data prep: minimal helpers to ready tokenizer and build train/validation 
splits from JSON files compatible with Hugging Face Datasets.
- CLI ergonomics: reusable ArgumentParser field presets for consistent flags, 
types, and help texts across projects.

Designed to be plug-and-play, with minimal setup and clear, composable components for everyday ML workflows.

## Features
- `wandb_handler` 
  - Easy W&B run initialization with tags, merged configs, and timestamped names
  - Log datasets and models as W&B artifacts
  - Log scalar metrics and confusion matrices
  - Store prediction results in W&B tables
- `training_handler` 
  - Lightweight wrapper to prepare tokenizer and datasets for training/eval
  - Accepts JSON files using Hugging Face Datasets.
  - Tracks core paths/configs and prepares in-memory train/validation splits.
- `argument_parser` 
  - Build ArgumentParser instances from reusable field presets
  - Handle required/not-required inputs, output paths, models, and W&B flags.
  - Consistent short/long options, help texts, and types.

## Installation
- Python 3.10+
- Install in your virtualenv:
  - Clone the repository
  - Install the package in editable mode (if applicable): `pip install -e .`
  - Ensure W&B is installed: `pip install wandb`

## Requirements
- wandb
- transformers

## Quick start

### WB integration -- example

**Example**: tracking a training run with W&B

```python
from rdl_ml_utils.handlers.wandb_handler import WanDBHandler

# Minimal config objects (adapt as needed)
class WandbConfig:
    PROJECT_NAME = "my-project"
    PROJECT_TAGS = ["experiment", "baseline"]
    PREFIX_RUN = "run-"
    BASE_RUN_NAME = "training"

run_config = {"base_model": "distilbert-base-uncased"}
training_args = type("Args", (), {"epochs": 3, "batch_size": 32})()

# Initialize W&B
WanDBHandler.init_wandb(
    wandb_config=WandbConfig,
    run_config=run_config,
    training_args=training_args,
)

# Log metrics
WanDBHandler.log_metrics({"loss": 0.42, "accuracy": 0.88}, step=1)

# Finish run
WanDBHandler.finish_wand()
```

### Argument parser (argument_parser)

Minimal example:

```python
from rdl_ml_utils.utils.argument_parser import (
    prepare_parser_for_fields,
    INPUT_FILE_REQUIRED,
    BASE_MODEL_REQUIRED,
    WANDB_BOOLEAN_FULL,
)

parser = prepare_parser_for_fields(
    [INPUT_FILE_REQUIRED, BASE_MODEL_REQUIRED, WANDB_BOOLEAN_FULL],
    description="My CLI app"
)
args = parser.parse_args()
# args.input_file, args.base_model, args.wandb_full (bool)
```

### Training handler (training_handler)

Minimal example:

```python
from rdl_ml_utils.handlers.training_handler import TrainingHandler

th = TrainingHandler(
    train_dataset_file_path="data/train.json",
    eval_dataset_file_path="data/valid.json",
    base_model="model/path",
    train_batch_size=32,
    workdir="./workdir"
)

# Access prepared objects:
#   th.tokenizer, 
#   th.train_dataset, 
#   th.eval_dataset, 
#   th.train_batch_size
```

## License
Read [LICENSE](LICENSE)
