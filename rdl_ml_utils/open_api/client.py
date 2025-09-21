import json
import requests

from pathlib import Path
from typing import List, Dict, Any, Optional, Union

#
# class OpenApiConfig:
#     """
#     Loads OpenAPI client configuration from a JSON file.
#     """
#
#     def __init__(
#         self,
#         base_url: Optional[str] = None,
#         model: Optional[str] = None,
#         api_key: Optional[str] = None,
#         system_prompt: Optional[str] = None,
#     ):
#         self.base_url = base_url.rstrip("/")
#         self.model = model
#         self.api_key = api_key
#         self.system_prompt = system_prompt if system_prompt else ""
#
#     @classmethod
#     def from_json(cls, json_path: str | Path) -> "OpenApiConfig":
#         """
#         Reads a JSON configuration file and returns an `OpenApiConfig` instance.
#         """
#         path = Path(json_path)
#         if not path.is_file():
#             raise FileNotFoundError(f"Config file not found: {path}")
#
#         with path.open("r", encoding="utf-8") as f:
#             data = json.load(f)
#
#         if "base_url" not in data or "model" not in data:
#             raise ValueError(
#                 "Config JSON must contain at least 'base_url' and 'model' fields."
#             )
#
#         return cls(
#             base_url=data["base_url"],
#             model=data["model"],
#             api_key=data.get("api_key"),
#             system_prompt=data.get("system_prompt"),
#         )


class OpenAPIClient:
    """
    Simple client for interacting with server with OpenAPI specification.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        system_prompt: Optional[str] = None,
        open_api_config: Optional[Union[str, Path]] = None,
    ):
        """
        Initialize the client.

        You can either:
        * Provide `base_url` and `model` directly, **or**
        * Pass a path to a JSON config file via `open_api_config` (the file must
          contain at least `base_url` and `model`).

        Parameters
        ----------
        base_url : str, optional (default=None)
            Base URL of the OpenAPI server.
        model : str, optional (default=None)
            Model name to use for generation.
        api_key : str, optional (default=None)
            Optional API key for authentication.
        system_prompt : str, optional (default: None)
            Optional system‑level prompt that will be prepended to every chat request.
        open_api_config : str | Path, optional
            Path to a JSON configuration file (see :class:`OpenApiConfig`).
        """
        self.model = model
        self.system_prompt = system_prompt
        self.base_url = base_url.rstrip("/") if base_url else None

        if open_api_config is not None:
            config = OpenApiConfig.from_json(open_api_config)
            self.model = config.model
            self.base_url = config.base_url
            self.system_prompt = config.system_prompt
            api_key = config.api_key

        if not self.base_url or not self.model:
            raise ValueError(
                "When 'open_api_config' is not given, "
                "both 'base_url' and 'model' must be provided."
            )

        # Header preparation (common for both paths)
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"

    def generate(
        self,
        message: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Generate a completion for a single prompt.

        If *system_prompt* is provided, it will be prefixed to the user prompt,
        overriding the client‑wide `self.system_prompt` (if any).
        """
        effective_system = (
            system_prompt if system_prompt is not None else self.system_prompt
        )
        full_prompt = (
            f"{effective_system}\n{message}" if effective_system else message
        )

        payload = {
            "model": self.model,
            "prompt": full_prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        result = self._post("/v1/completions", payload)
        return result.get("choices", [{}])[0].get("text", "")

    def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 256,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Chat completion using a list of messages.

        *system_prompt* – optional system‑level prompt that, when supplied,
        replaces the client‑wide `self.system_prompt` for this call.
        """
        effective_system = (
            system_prompt if system_prompt is not None else self.system_prompt
        )
        if effective_system:
            has_system = any(msg.get("role") == "system" for msg in messages)
            if not has_system:
                messages = [
                    {"role": "system", "content": effective_system}
                ] + messages

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        result = self._post("/v1/chat/completions", payload)
        return result.get("choices", [{}])[0].get("message", {}).get("content", "")

    def is_available(self, timeout: float = 2.0) -> bool:
        """
        Check whether the OpenAPI server is reachable.

        Performs a simple `GET` request to the base URL (or `/health` if the
        server implements such an endpoint).  Returns `True` if the request
        succeeds with a *2xx* status code, otherwise `False`.

        Parameters
        ----------
        timeout: float, optional
            Number of seconds to wait for a response before giving up.

        Returns
        -------
        bool
            `True` if the API is reachable, `False` otherwise.
        """
        try:
            # Most instances expose a simple health check at `/`.
            # If a dedicated `/health` endpoint exists, it will also return 200.
            url = f"{self.base_url}/"
            response = requests.get(url, headers=self.headers, timeout=timeout)
            return 200 <= response.status_code < 300
        except (requests.RequestException, requests.Timeout):
            return False

    def _post(self, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Internal helper to POST JSON payload to a given endpoint.
        """
        url = f"{self.base_url}{endpoint}"
        response = requests.post(url, json=payload, headers=self.headers)
        response.raise_for_status()
        return response.json()

    # Context‑manager protocol
    def __enter__(self) -> "OpenAPIClient":
        """
        Enter the runtime context related to this object.

        Returns
        -------
        OpenAPIClient
            The client instance itself.
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """
        Exit the runtime context and perform any cleanup.

        Parameters
        ----------
        exc_type, exc_val, exc_tb : optional
            Exception information (if any) propagated from the `with` block.

        Returns
        -------
        bool
            `False` to indicate that any exception should be propagated.
        """
        return False
