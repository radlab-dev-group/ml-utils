import json

from pathlib import Path
from typing import Optional


class OpenApiConfig:
    """
    Loads OpenAPI client configuration from a JSON file.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.system_prompt = system_prompt if system_prompt else ""

    @classmethod
    def from_json(cls, json_path: str | Path) -> "OpenApiConfig":
        """
        Reads a JSON configuration file and returns an `OpenApiConfig` instance.
        """
        path = Path(json_path)
        if not path.is_file():
            raise FileNotFoundError(f"Config file not found: {path}")

        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if "base_url" not in data or "model" not in data:
            raise ValueError(
                "Config JSON must contain at least 'base_url' and 'model' fields."
            )

        return cls(
            base_url=data["base_url"],
            model=data["model"],
            api_key=data.get("api_key"),
            system_prompt=data.get("system_prompt"),
        )
