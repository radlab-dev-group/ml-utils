# radlab-ml-utils

Lightweight utilities to streamline machine learning workflows. 
Includes helpers for experiment tracking with Weights & Biases (W&B): 

- artifact logging, 
- metrics reporting, 
- simple plotting utilities.

## Features
- Easy W&B run initialization with tags, merged configs, and timestamped names
- Log datasets and models as W&B artifacts
- Log scalar metrics and confusion matrices
- Store prediction results in W&B tables

## Installation
- Python 3.10+
- Install in your virtualenv:
  - Clone the repository
  - Install the package in editable mode (if applicable): `pip install -e .`
  - Ensure W&B is installed: `pip install wandb`

## Quick start

``` python
# Example: tracking a training run with W&B
from radlab_ml_utils.wandb_handler import WanDBHandler

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

## Requirements
- wandb

## License
Read [LICENSE](LICENSE)
