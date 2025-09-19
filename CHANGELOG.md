## Changelog

| Version | Changelog                                                                                                                                                                                                           |
|---------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 0.0.1   | Initialization, License, setup, Weights and Biases handler                                                                                                                                                          |
| 0.0.2   | Added `utils.*` module, renamed `wandb` to `wandb_handler`, added `handlers.training_handler`                                                                                                                       | 
| 0.0.3   | Added `openapi_handler` handler -- a lightweight OpenAPI requests api. Added `prompt_handler` module. Add load balances `OpenAPIQueue` as simple queue management to multiple OpenAPI services. Readme update.      |
| 0.0.4   | Refactored code to `open_api` separated module. Add module `OpenApiHandlerWithCache` as simple manager to connect PromptHandler and OpenAPIQueue to multithread executions (with cached user query-response pairs). |



---

// ... existing code ...

### `training_handler.py`

The **Training handler** (`TrainingHandler`) streamlines dataset preparation for transformer‑based models. It:

* Loads JSON‑line datasets using the 🤗 Datasets library.
* Instantiates a tokenizer from a Hugging‑Face model (e.g., `bert-base-uncased`).
* Stores useful metadata such as the number of unique labels.
* Exposes ready‑to‑use `train_dataset` and `eval_dataset` attributes
  (creation of the actual `DataLoader`s is left to the user, keeping the class framework‑agnostic).

// ... existing code ...

### `wandb_handler.py`

The **WandB handler** (`WanDBHandler`) centralises all interactions with the Weights & Biases service, providing a
high‑level API for:

// ... existing code ...

### `prompt_handler.py`

The **Prompt handler** (`PromptHandler`) offers a simple way to load, store, and retrieve textual prompts stored as
`*.prompt` files.  
Prompts are indexed by a *key* that corresponds to the file’s path **relative to the base directory**, using forward
slashes and **without the file extension**.

// ... existing code ...

## 🚀 Getting Started

1. **Clone the repository**
```
shell script
git clone https://github.com/radlab-dev-group/ml-utils.git
cd ml-utils
```
2. **Create a virtual environment and install dependencies**
```
shell script
python -m venv .venv
source .venv/bin/activate
pip install -e .
```
3. **Run the OpenAPI demo**
```
shell script
python apps/openapi_test.py
```
Tip: For higher throughput or multi-endpoint setups, consider using the queue manager described in open_api/queue_manager.py.  
You can also add a local cache layer (see open_api/cache_api.py) to avoid recomputing identical requests during experiments.

---
