from typing import Dict
from pathlib import Path


class PromptHandler:
    """
    Loads prompt files (*.prompt) from a given directory recursively.
    Prompts are accessible via a key representing the relative path
    from the base directory, using forward slashes.
    Example key: "subdir/example_prompt.prompt"
    """

    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir).resolve()
        self.prompts: Dict[str, str] = {}
        self._load_prompts()

    def get_prompt(self, key: str) -> str:
        """
        Retrieve the prompt content by its key.

        Args:
            key: Relative path to the prompt file (e.g., "dir/sample.prompt").

        Returns:
            The contents of the prompt file.

        Raises:
            KeyError: If the key does not exist.
        """
        return self.prompts[key]

    def list_prompts(self) -> Dict[str, str]:
        """
        Return a copy of the internal `prompts` dictionary.
        """
        return dict(self.prompts)

    def _load_prompts(self):
        """
        Recursively walk the base directory and read *.prompt files.
        """
        for file_path in self.base_dir.rglob("*.prompt"):
            rel_path_with_ext = file_path.relative_to(self.base_dir).as_posix()
            rel_path = rel_path_with_ext.rsplit(".", 1)[0]
            try:
                with file_path.open("r", encoding="utf-8") as f:
                    self.prompts[rel_path] = f.read()
            except OSError as e:
                raise RuntimeError(f"Failed to read prompt file {file_path}: {e}")

    def __enter__(self):
        """
        Enable usage as a context manager.
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Placeholder for cleanup; nothing special needed currently.
        """
        return False
