## Changelog

| Version | Changelog                                                                                                                                                                                                           |
|---------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 0.0.1   | Initialization, License, setup, Weights and Biases handler                                                                                                                                                          |
| 0.0.2   | Added `utils.*` module, renamed `wandb` to `wandb_handler`, added `handlers.training_handler`                                                                                                                       | 
| 0.0.3   | Added `openapi_handler` handler -- a lightweight OpenAPI requests api. Added `prompt_handler` module. Add load balances `OpenAPIQueue` as simple queue management to multiple OpenAPI services. Readme update.      |
| 0.0.4   | Refactored code to `open_api` separated module. Add module `OpenApiHandlerWithCache` as simple manager to connect PromptHandler and OpenAPIQueue to multithread executions (with cached user query-response pairs). |
