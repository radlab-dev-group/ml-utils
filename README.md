# radlab-ml-utils

## 📖 Overview

`radlab-ml-utils` is a collection of helper utilities and handlers designed to simplify  
common machine‑learning workflows. The package includes:

- **OpenAPI handler** – thin client for LLM servers exposing an OpenAPI spec.
- **Training handler** – utilities for preparing datasets, tokenization and orchestrating training pipelines.
- **WandB handler** – convenient wrappers around Weights & Biases for experiment tracking,
  artifact management and rich logging.
- **Prompt handler** – loads and manages prompt files (`*.prompt`) from a directory tree,
  making it easy to reuse and reference prompts programmatically.

The library is built on Python 3.10 and can be installed via `pip install .` after cloning the repository.

---

## 📂 Project Structure

```

radlab-ml-utils/
│
├─ apps/
│   └─ __init__.py
│   └─ openapi_test.py          # Example script demonstrating the OpenAPI client
│
├─ configs/
│   └─ ollama_config.json       # Sample OpenAPI configuration file
│
├─ rdl_ml_utils/
│   ├─ handlers/
│   │   ├─ __init__.py
│   │   ├─ openapi_handler.py   # Core OpenAPI client implementation
│   │   ├─ training_handler.py  # Dataset loading & training helpers
│   │   ├─ wandb_handler.py     # W&B integration utilities
│   │   └─ prompt_handler.py    # Prompt loading and lookup utilities
│   ├─ open_api/
│   │   ├─ __init__.py
│   │   ├─ cache_api.py         # Request-response caching utilities for OpenAPI calls
│   │   └─ queue_manager.py     # Threaded queue/pool for multiple OpenAPI clients
│   └─ utils/
│       └─ __init__.py
│
├─ .gitignore
├─ CHANGELOG.md
├─ LICENSE
├─ README.md                    # *You are reading it right now*
├─ requirements.txt
└─ setup.py
```

---

## 🛠️ Handlers

### `openapi_handler.py`

The **OpenAPI client** (`OpenAPIClient`) provides a simple, opinionated interface for interacting with LLM servers that
follow the OpenAI‑compatible API schema.

#### Key Features

| Feature                     | Description                                                                                      |
|-----------------------------|--------------------------------------------------------------------------------------------------|
| **Flexible initialization** | Pass `base_url` / `model` directly **or** load them from a JSON config file (`open_api_config`). |
| **Authentication**          | Optional `api_key` is added as a `Bearer` token header when supplied.                            |
| **Prompt generation**       | `generate(prompt, …)` returns a plain‑text completion.                                           |
| **Chat completions**        | `chat(messages, …)` works with the standard `[{role, content}]` message format.                  |
| **System prompt handling**  | A global `system_prompt` can be set at client creation and overridden per call.                  |
| **Health check**            | `is_available()` performs a quick GET request to verify server reachability.                     |
| **Context‑manager**         | Use `with OpenAPIClient(...) as client:` for clean entry/exit semantics.                         |

#### Configuration File (`configs/ollama_config.json`)

```json
{
  "base_url": "http://localhost:11434",
  "model": "MODEL_NAME",
  "api_key": "YOUR_API_KEY_IF_NEEDED",
  "system_prompt": "You are a helpful AI assistant."
}
```

#### Example Usage

```python
from rdl_ml_utils.open_api.client import OpenAPIClient

# Load configuration from JSON (recommended for reproducibility)
with OpenAPIClient(open_api_config="configs/ollama_config.json") as client:
    # Verify the server is up
    if not client.is_available():
        raise RuntimeError("OpenAPI server is not reachable.")

    # Simple generation
    answer = client.generate(
        message="Explain logistic regression.",
        system_prompt="You are a statistics expert.",
        max_tokens=512,
    )
    print("Generation result:", answer)

    # Chat‑style interaction
    chat_messages = [
        {"role": "user", "content": "What are the biggest challenges in ML today?"},
    ]
    response = client.chat(
        messages=chat_messages,
        system_prompt="Speak like a senior data scientist.",
        max_tokens=256,
    )
    print("Chat response:", response)
```

#### Configuration File (`configs/ollama_config.json`)

```json
{
  "base_url": "http://localhost:11434",
  "model": "MODEL_NAME",
  "api_key": "YOUR_API_KEY_IF_NEEDED",
  "system_prompt": "You are a helpful AI assistant."
}
```

---

### `openapi_queue_manager.py`

The **OpenAPI queue manager** (`OpenAPIQueue`) provides a thin, thread‑safe wrapper around multiple
`OpenAPIClient` instances. It loads client configurations from JSON files, creates a pool of worker
threads, and processes `generate` and `chat` requests in a first‑in‑first‑out (FIFO) order.

#### Why use it?

| Benefit                          | Explanation                                                                                                                                  |
|----------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------|
| **Parallel client usage**        | Multiple LLM endpoints can be configured and used simultaneously, improving throughput.                                                      |
| **Automatic request scheduling** | Calls are enqueued and dispatched to the first free client, so you never have to manage locks yourself.                                      |
| **Simple synchronous API**       | Despite the internal concurrency, the public methods (`generate`, `chat`) block until a result is ready, making integration straightforward. |
| **Graceful shutdown**            | `close()` cleanly stops all background workers.                                                                                              |

#### Core API

```python
from pathlib import Path
from rdl_ml_utils.open_api.queue_manager import OpenAPIQueue

# Initialise the queue with one or more client config files
queue = OpenAPIQueue([
    Path("configs/ollama-config.json"),
    Path("configs/ollama-config_lab4_1.json"),
])

# Generate a completion (handled by the first available client)
answer = queue.generate(
    "Explain quantum entanglement.",
    max_tokens=128,
)

# Chat‑style interaction
reply = queue.chat(
    "What is the capital of France?",
    max_tokens=64,
)

# When finished, shut down the workers
queue.close()
```

The class handles all low‑level details (client selection, locking, task queuing)
so you can focus on the prompts and model logic.

---

### `open_api/cache_api.py`

The **OpenAPI cache utilities** provide a lightweight caching layer for request/response pairs
made via the OpenAPI client stack. This is useful when you want to avoid repeated calls
for identical inputs (e.g., during prompt iteration, evaluation, or notebook exploration).

#### Key capabilities

- Request signature hashing based on input content and selected generation parameters.
- In‑memory cache for fast repeat access and optional persistence to disk to survive process restarts.
- Transparent use with generation or chat‑style calls to reduce latency and cost.
- Simple controls to bypass cache for a specific call or to invalidate entries programmatically.

#### Typical usage patterns

- Wrap your OpenAPI calls with a cache lookup, and only call the backend if there is a cache miss.
- Scope caches by project/run folder so that experiments do not interfere with each other.
- Periodically clear or prune the cache (e.g., by a max size or time‑to‑live policy) according to your workflow.

Example flow (conceptual):

1) Build a request key from the prompt/messages and key parameters (e.g., model, temperature, max_tokens).
2) If the key is present, return the stored response.
3) Otherwise, call the OpenAPI client, store the response under that key, and return it.

This module is designed to be orthogonal to the queue manager: you can use caching with or without the queue,
and you can place caching either outside (per request) or inside (per worker) depending on your needs.

---

### `training_handler.py`

The **Training handler** (`TrainingHandler`) streamlines dataset preparation for transformer‑based models. It:

* Loads JSON‑line datasets using the 🤗 Datasets library.
* Instantiates a tokenizer from a Hugging‑Face model (e.g., `bert-base-uncased`).
* Stores useful metadata such as the number of unique labels.
* Exposes ready‑to‑use `train_dataset` and `eval_dataset` attributes
  (creation of the actual `DataLoader`s is left to the user, keeping the class framework‑agnostic).

#### Core API

```python
from rdl_ml_utils.handlers.training_handler import TrainingHandler

handler = TrainingHandler(
    train_dataset_file_path="data/train.jsonl",
    eval_dataset_file_path="data/valid.jsonl",
    base_model="distilbert-base-uncased",
    train_batch_size=16,
    workdir="./workdir",
)

# After initialization:
#   handler.tokenizer          -> AutoTokenizer instance
#   handler.train_dataset      -> 🤗 Dataset with training examples
#   handler.eval_dataset       -> 🤗 Dataset with validation examples
#   handler.uniq_labels        -> Set of label strings
```

#### What the class does internally

```python
# ... existing code ...

self.tokenizer = AutoTokenizer.from_pretrained(
    self.base_model, use_fast=True
)

data = load_dataset(
    "json",
    cache_dir="./cache",
    data_files={
        "train": self.train_dataset_file_path,
        "validation": self.eval_dataset_file_path,
    },
)

self.train_dataset = data["train"]
self.eval_dataset = data["validation"]

# ... existing code ...
```

The handler is deliberately lightweight: it only prepares raw datasets and tokenizers, leaving model definition,
optimizer setup and training loops to the user’s own script or training framework (PyTorch, TensorFlow, 🤗 Trainer,
etc.). This makes it easy to plug into existing pipelines while keeping reproducibility (datasets are cached under
`./cache`).

---

### `wandb_handler.py`

The **WandB handler** (`WanDBHandler`) centralises all interactions with the Weights & Biases service, providing a
high‑level API for:

| Action                       | Method                                                                                | Description                                                                                                     |
|------------------------------|---------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------|
| **Initialize a run**         | `init_wandb`                                                                          | Sets up a W&B run with project name, tags, and a merged configuration dict (run‑specific + training arguments). |
| **Log scalar metrics**       | `log_metrics`                                                                         | Sends a dictionary of metric name → value pairs, optionally with a step number.                                 |
| **Store datasets / models**  | `add_dataset`, `add_model`                                                            | Creates a `wandb.Artifact` of type *dataset* or *model* and uploads the supplied directory.                     |
| **Finish a run**             | `finish_wandb`                                                                        | Calls `wandb.run.finish()` to close the run cleanly.                                                            |
| **Prepare run metadata**     | `prepare_run_tags`, `prepare_run_name_with_date`, `prepare_simple_run_name_with_date` | Helper functions that add host information, timestamps and model identifiers to run names/tags.                 |
| **Merge configs**            | `prepare_run_config`                                                                  | Combines a user‑provided dict with attributes of a `training_args` object (e.g., from 🤗 Trainer).              |
| **Plot confusion matrix**    | `plot_confusion_matrix`                                                               | Uses `wandb.plot.confusion_matrix` to visualise classification performance.                                     |
| **Log detailed predictions** | `store_prediction_results`                                                            | Creates a `wandb.Table` with raw text, true label, predicted label and optional per‑class probabilities.        |

#### Example Usage

```python
from rdl_ml_utils.handlers.wandb_handler import WanDBHandler


# Simple config object (could be a dataclass or Namespace)
class WandbConfig:
    PROJECT_NAME = "ml-experiments"
    PROJECT_TAGS = ["nlp", "classification"]
    PREFIX_RUN = "run_"
    BASE_RUN_NAME = "experiment"


wandb_cfg = WandbConfig()
run_cfg = {"base_model": "distilbert-base-uncased", "learning_rate": 3e-5}
training_args = None  # could be an argparse.Namespace with many fields

# Initialise run (name will include timestamp and model name)
WanDBHandler.init_wandb(
    wandb_config=wandb_cfg,
    run_config=run_cfg,
    training_args=training_args,
    run_name=None,  # auto‑generated
)

# Log some metrics during training
for epoch in range(3):
    # ... training logic ...
    WanDBHandler.log_metrics({"epoch": epoch, "accuracy": 0.87 + epoch * 0.01})

# After training, store the model artifact
WanDBHandler.add_model(name="distilbert-finetuned", local_path="./workdir/model")

# Finish the run
WanDBHandler.finish_wandb()
```

#### Plotting a Confusion Matrix

```python
# ground_truth and predictions are list‑like, class_names is a list of label strings
WanDBHandler.plot_confusion_matrix(
    ground_truth=y_true,
    predictions=y_pred,
    class_names=["neg", "pos"],
    probs=prediction_probs,  # optional probability matrix
)
```

#### Storing Detailed Prediction Results

```python
WanDBHandler.store_prediction_results(
    texts_str=test_texts,
    ground_truth=y_true,
    pred_labels=y_pred,
    probs=prediction_probs,
)
```

All helper methods automatically add the host name to run tags, ensuring that runs from different machines are easily
distinguishable.

---

### `prompt_handler.py`

The **Prompt handler** (`PromptHandler`) offers a simple way to load, store, and retrieve textual prompts stored as
`*.prompt` files.  
Prompts are indexed by a *key* that corresponds to the file’s path **relative to the base directory**, using forward
slashes and **without the file extension**.

#### Core API

```python
from rdl_ml_utils.handlers.prompt_handler import PromptHandler

# Initialise the handler pointing at a directory that contains *.prompt files
prompt_dir = "configs/prompts"  # any directory you like
ph = PromptHandler(base_dir=prompt_dir)

# List all loaded prompts (key → content)
all_prompts = ph.list_prompts()
print("Available prompts:", list(all_prompts.keys()))

# Retrieve a specific prompt
key = "system/default"  # corresponds to configs/prompts/system/default.prompt
prompt_text = ph.get_prompt(key)
print("Prompt content:", prompt_text)
```

#### How It Works

* **Recursive loading** – The handler walks the `base_dir` recursively (`Path.rglob("*.prompt")`).
* **Key generation** – For each file, the relative path (POSIX style) is taken, the `.prompt` suffix is stripped, and
  the result becomes the dictionary key.
* **In‑memory storage** – Prompt contents are kept in a plain Python `dict[str, str]`, making subsequent look‑ups O(1).

#### Typical Use‑Cases

| Scenario                     | How PromptHandler helps                                                                                                                                  |
|------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Prompt engineering**       | Keep a library of reusable prompts (system, few‑shot examples, task‑specific templates) in a dedicated folder; retrieve them by logical name at runtime. |
| **Dynamic prompt selection** | Based on experiment configuration, select the appropriate prompt key (`"qa/simple"`, `"summarization/long"` etc.) without hard‑coding file paths.        |
| **Versioned prompts**        | Store multiple versions (`v1.prompt`, `v2.prompt`) in sub‑folders; the key reflects the version (`"summarization/v1"`).                                  |

#### Error handling

* `KeyError` is raised if a non‑existent key is requested.
* `RuntimeError` is raised if a prompt file cannot be read (e.g., permission issues).

---

## 🚀 Getting Started

1. **Clone the repository**

```shell script
git clone https://github.com/radlab-dev-group/ml-utils.git
cd ml-utils
```

2. **Create a virtual environment and install dependencies**

```shell script
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

3. **Run the OpenAPI demo**

```shell script
python apps/openapi_test.py
```

Tip: For higher throughput or multi-endpoint setups, consider using the queue manager described
in open_api/queue_manager.py. You can also add a local cache layer (see open_api/cache_api.py)
to avoid recomputing identical requests during experiments.

---

## 📦 Installation

```shell script
pip install git+https://github.com/radlab-dev-group/ml-utils.git
```

or, after cloning:

```shell script
pip install .
```

> The base install has **no heavy dependencies** (e.g. `wandb`, `transformers` are not
> pulled in). Install the optional dependencies (from `requirements.txt`) with:
>
> ```shell script
> pip install .[deps]
> ```

---

## 📜 License

This project is licensed under the Apache 2.0 License – see the [LICENSE](LICENSE) file for details.