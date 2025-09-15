import sys

from rdl_ml_utils.handlers.openapi_handler import OpenAPIClient


def main():

    with OpenAPIClient(open_api_config="configs/ollama_config.json") as client:
        if not client.is_available():
            print("Ollama is not available!")
            return 1
        print("Ollama is available!")

        print("Uruchamianie: client.generate")
        print(
            client.generate(
                message="Wyjaśnij, czym jest regresja logistyczna.",
                system_prompt="Jesteś ekspertem w statystyce.",
                max_tokens=2048,
            )
        )

        print("Uruchamianie: client.chat")
        messages = [
            {"role": "user", "content": "Jakie są najważniejsze wyzwania w ML?"}
        ]
        print(
            client.chat(
                messages=messages,
                system_prompt="Rozmawiaj jak senior data scientist.",
                max_tokens=2048,
            )
        )
    return 0


sys.exit(main())
