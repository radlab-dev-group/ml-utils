"""
Utility module for processing JSON and JSONL datasets.

Provides the :class:`DatasetProcessor` which abstracts loading of one or
more dataset files, handling both JSON and JSONL formats transparently.
It also supports optional field filtering to reduce memory usage when
only a subset of the data is required.

Typical usage::

    processor = DatasetProcessor(
        dataset_paths=["data/train.jsonl", "data/valid.json"],
        accept_fields=["text", "label"]
    )
    records = processor.load_records()
"""

from pathlib import Path
from typing import List, Dict, Any

from rdl_ml_utils.utils.loaders import JSONLLoader, JSONLoader, RawTextLoader

# ----------------------------------------------------------------------
# Registry that maps a dataset type string to the appropriate loader class.
# This makes it easy to add new loaders in the future—just extend the dict.
# ----------------------------------------------------------------------
_DATASET_LOADER_REGISTRY = {
    "json": JSONLoader,
    "jsonl": JSONLLoader,
    "txt": RawTextLoader,
}


class DatasetProcessor:
    """
    Process JSON and JSONL dataset files.

    This class loads records from one or more dataset files that are
    either in JSON (single object) or JSONL (newline‑delimited JSON) format.
    It abstracts away the loader selection and provides a simple
    ``load_records`` method that returns a flat list of dictionaries.
    Optional ``accept_fields`` can be provided to filter the fields
    returned by the underlying loaders.
    """

    def __init__(
        self,
        dataset_paths: List[str],
        dataset_type: str | None = None,
        accept_fields: List[str] | None = None,
    ):
        """
        Initialize a new :class:`DatasetProcessor`.

        Parameters
        ----------
        dataset_paths: List[str]
            A list of file system paths (as strings) pointing to the dataset
            files to be processed. Each path may refer to a JSON or JSONL file.
        dataset_type: str | None, optional
            Explicitly specify the dataset type ('json' or 'jsonl'). If ``None``,
            the type is inferred from each file's extension.
        accept_fields: List[str] | None, optional
            If provided, only these top‑level keys will be retained in each
            loaded record. This is forwarded to the underlying loader classes.
        """
        self.dataset_paths = dataset_paths
        self.dataset_type = dataset_type
        self.accept_fields = accept_fields or None

    def load_records(self) -> List[Dict[str, Any]]:
        """
        Load and return all records from the configured dataset files.

        The method iterates over each path supplied at construction time,
        determines the appropriate loader based on the file extension (or
        the explicit ``dataset_type``), and aggregates the results into a
        single flat list.  Each element of the returned list is a dictionary
        representing a single record from the source files.

        Returns
        -------
        List[Dict[str, Any]]
            A list containing all loaded records, optionally filtered by
            ``accept_fields``.
        """
        records: List[Dict[str, Any]] = []
        for path_str in self.dataset_paths:
            path = Path(path_str)

            # Determine which loader to use (json / jsonl)
            dtype = self.dataset_type or self._infer_dataset_type(path)

            # Fetch the loader class from the registry; raise a clear error
            # if an unsupported type is encountered.
            loader_cls = _DATASET_LOADER_REGISTRY.get(dtype)
            if loader_cls is None:
                raise ValueError(
                    f"Unsupported dataset type '{dtype}' for file {path}"
                )

            loader = loader_cls(path, accept_fields=self.accept_fields)

            records.extend(list(loader.load()))
        return records

    @staticmethod
    def _infer_dataset_type(path: Path) -> str:
        """
        Infer the dataset type from a file's extension.

        Parameters
        ----------
        path: Path
            The path object pointing to a dataset file.

        Returns
        -------
        str
            Either ``'json'`` or ``'jsonl'`` depending on the file extension.

        Raises
        ------
        ValueError
            If the file extension is not recognised as a supported dataset
            type.
        """
        ext = path.suffix.lower()
        if ext == ".json":
            return "json"
        if ext == ".jsonl":
            return "jsonl"

        available_types = ", ".join(list(_DATASET_LOADER_REGISTRY.keys()))
        raise ValueError(
            f"Cannot infer dataset type from extension '{ext}' for file {path}. "
            f"Possible types to load with this Processor: [{available_types}]"
        )
