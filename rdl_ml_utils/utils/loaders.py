"""
Data loading utilities.

Provides a small, extensible framework for streaming data from various sources.
The current implementation includes a JSON Lines (JSONL) loader that can handle
very large files without loading the whole file into memory.
"""

from __future__ import annotations

import json

from pathlib import Path
from abc import ABC, abstractmethod
from typing import Any, Dict, Generator, Iterable, List, Optional


class LoadInterface(ABC):
    """Abstract base class for data loaders."""

    @abstractmethod
    def load(self) -> Iterable[Dict[str, Any]]:
        """
        Yield records from the data source.

        Implementations should return an **iterable** (often a generator) that
        yields one dictionary per record.
        """
        ...


class RawTextLoader(LoadInterface):
    """
    Loader for raw text files.

    Each line of the file is yielded as a dictionary. By default the
    line content is stored under the key ``"text"``, but you can customize
    the field name with ``text_field``. If ``include_line_number`` is
    ``True``, the line number (starting at 1) is added under the key
    ``line_number`` (or a custom ``line_number_field``).

    The loader streams the file, so memory usage stays low even for
    very large text sources.
    """

    def __init__(
        self,
        path: str | Path,
        accept_fields: Optional[List[str]] = None,
        *,
        encoding: str = "utf-8",
        strip: bool = True,
        include_line_number: bool = False,
        text_field: str = "text",
        line_number_field: str = "line_number",
    ):
        """
        Parameters
        ----------
        path : str | Path
            Path to the text file.
        encoding : str, optional
            File encoding (default ``'utf-8'``).
        strip : bool, optional
            Remove trailing newline characters from each line
            (default ``True``). Set to ``False`` to keep the exact
            line content.
        include_line_number : bool, optional
            Weather to add the line number to each yielded record
            (default ``False``).
        text_field : str, optional
            Dictionary key under which the line text is stored
            (default ``'text'``).
        line_number_field : str, optional
            Dictionary key for the line number when
            ``include_line_number`` is ``True``
            (default ``'line_number'``).
        """
        self.accept_fields = accept_fields

        self.path = Path(path)
        self.encoding = encoding
        self.strip = strip
        self.include_line_number = include_line_number
        self.text_field = text_field
        self.line_number_field = line_number_field

    def load(self) -> Generator[Dict[str, Any], None, None]:
        """
        Yield one record per line of the file.

        Returns
        -------
        Generator[Dict[str, Any], None, None]
            Each dict contains at least the ``text_field`` entry. If
            ``include_line_number`` is ``True``, a ``line_number_field``
            entry is also present.
        """
        with self.path.open("r", encoding=self.encoding) as f:
            for line_number, line in enumerate(f, start=1):
                if self.strip:
                    line = line.rstrip("\n\r")
                record: Dict[str, Any] = {self.text_field: line}
                if self.include_line_number:
                    record[self.line_number_field] = line_number
                yield record


class JSONLLoader(LoadInterface):
    """
    Loader for JSON Lines (JSONL) files.

    It reads the file line‑by‑line, parses each line as JSON and yields the
    resulting dictionaries. Because it streams the file, memory consumption
    stays low even for multi‑gigabyte inputs.
    """

    def __init__(
        self,
        path: str | Path,
        encoding: str = "utf-8",
        accept_fields: Optional[List[str]] = None,
    ):
        """
        Parameters
        ----------
        path: str | Path
            Path to the ``.jsonl`` file.
        encoding: str, optional
            File encoding (default ``'utf-8'``).
        accept_fields: list[str], optional
            List of field names to retain from each JSON object. Fields not in
            this list are discarded. If ``None`` (default), all fields are kept.
        """
        self.path = Path(path)
        self.encoding = encoding
        self.accept_fields = accept_fields

    def load(self) -> Generator[Dict[str, Any], None, None]:
        """
        Yield one JSON object per line.

        Empty lines are ignored. If a line cannot be parsed as JSON a
        ``ValueError`` with a helpful message is raised.
        """
        with self.path.open("r", encoding=self.encoding) as f:
            for line_number, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue  # skip blank lines
                try:
                    data = json.loads(line)
                    if type(data) not in [list]:
                        data = [data]

                    for _data in data:
                        if self.accept_fields is not None:
                            _data = {
                                k: v
                                for k, v in _data.items()
                                if k in self.accept_fields
                            }
                        yield _data
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON on line {line_number} of {self.path}: {exc}"
                    ) from exc


class JSONLoader(LoadInterface):
    """
    Loader for regular JSON files.

    The loader reads the entire JSON document into memory and yields
    dictionaries. If the top‑level JSON structure is a list, each element
    (expected to be a dict) is yielded separately. If it is a single dict,
    that dict is yielded as the sole record.

    An optional ``accept_fields`` argument works like in :class:`JSONLLoader`,
    allowing you to keep only a subset of keys.
    """

    def __init__(
        self,
        path: str | Path,
        encoding: str = "utf-8",
        accept_fields: Optional[List[str]] = None,
    ):
        """
        Parameters
        ----------
        path: str | Path
            Path to the ``.json`` file.
        encoding: str, optional
            File encoding (default ``'utf-8'``).
        accept_fields: list[str], optional
            List of field names to retain from each JSON object. If ``None``,
            all fields are kept.
        """
        self.path = Path(path)
        self.encoding = encoding
        self.accept_fields = accept_fields

    def load(self) -> Generator[Dict[str, Any], None, None]:
        """
        Yield dictionaries extracted from the JSON document.

        Raises
        ------
        ValueError
            If the file cannot be parsed as valid JSON.
        """
        with self.path.open("r", encoding=self.encoding) as f:
            try:
                content = json.load(f)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {self.path}: {exc}") from exc

        # Normalise to an iterable of dicts
        if isinstance(content, list):
            records = content
        else:
            records = [content]

        for record in records:
            if not isinstance(record, dict):
                # Skip non‑dict entries but keep the generator contract
                continue
            if self.accept_fields is not None:
                record = {k: v for k, v in record.items() if k in self.accept_fields}
            yield record
