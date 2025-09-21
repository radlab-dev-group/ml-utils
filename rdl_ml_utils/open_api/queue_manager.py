"""
Queue‑aware wrapper for multiple OpenAPIClient instances.

Each configuration file (JSON) describes a single OpenAPIClient.  The
`OpenAPIQueue` class loads those configs, creates the clients and
maintains a pool of workers that execute `generate` and `chat`
requests in a FIFO order. Calls are synchronous – the caller blocks
until the underlying API returns a result – but the implementation
ensures that only one request is sent to a given client at a time.

Typical usage
-------------
    from pathlib import Path
    from openapi_queue_manager import OpenAPIQueue

    queue = OpenAPIQueue([
        Path("configs/ollama-config.json"),
        Path("configs/ollama-config_lab4_1.json"),
    ])

    # Direct call – will be processed by the first free client
    answer = queue.generate("Explain quantum entanglement.", max_tokens=128)

    # Chat style call
    reply = queue.chat("What is the capital of France?")
"""

import abc
import json
import queue
import threading

from pathlib import Path
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

from rdl_ml_utils.open_api.client import OpenAPIClient


@dataclass
class _Task:
    """
    Internal task representation A single unit of work for a worker thread.
    """

    client_idx: Optional[int]
    method_name: str
    args: Tuple[Any, ...]
    kwargs: dict
    result_event: threading.Event
    result: Optional[str] = None
    exception: Optional[BaseException] = None


class TaskHandle:
    """
    Public handle for an enqueued task. Allows waiting for completion and
    retrieving the result or exception without blocking submission of other tasks.
    """

    def __init__(self, task: _Task) -> None:
        self._task = task

    @property
    def client_idx(self) -> Optional[int]:
        return self._task.client_idx

    def done(self) -> bool:
        return self._task.result_event.is_set()

    def wait(self, timeout: Optional[float] = None) -> bool:
        self._task.result_event.wait(timeout)
        return self.done()

    def result(self, timeout: Optional[float] = None) -> str:
        self.wait(timeout)
        if self._task.exception:
            raise self._task.exception
        return self._task.result or ""

    def exception(self, timeout: Optional[float] = None) -> Optional[BaseException]:
        self.wait(timeout)
        return self._task.exception


class _OpenAPIQueueWorkerConfig(abc.ABC):
    def __init__(
        self,
        config_paths: List[Path] | List[str] | Path | str,
        worker_count: Optional[int] = None,
    ) -> None:
        self._clients: List[OpenAPIClient] = []
        self._client_locks: List[threading.Lock] = []

        self._workers = None
        self._worker_count = None
        self._shutdown_event = None

        self._task_queue: queue.Queue[_Task] = queue.Queue()

        self._load_clients(config_paths=self._resolve_config_paths(config_paths))
        self._prepare_workers(worker_count=worker_count)

    def _load_clients(self, config_paths: List[Path]) -> None:
        """
        Instantiate `OpenAPIClient` objects from JSON config files.
        """
        for cfg_path in config_paths:
            client = self._create_client_from_path(cfg_path)
            self._clients.append(client)
            self._client_locks.append(threading.Lock())

    def _prepare_workers(self, worker_count: Optional[int] = None):
        """
        Initialize and start background worker threads.

        Parameters
        ----------
        worker_count : Optional[int]
            Desired number of worker threads. If `None` (the default),
            the manager creates one worker per configured `OpenAPIClient`.
            Supplying a larger value can increase throughput when the
            underlying API supports parallel requests.

        Notes
        -----
        * Each worker runs `self._worker_loop` in daemon mode, so it will
          not prevent the interpreter from exiting.
        * The method also creates a fresh `threading.Event` used to signal
          a graceful shutdown via `close`.
        * The list `self._workers` holds references to the threads so that
          they can be joined later.
        """
        self._worker_count = worker_count or len(self._clients)
        self._workers: List[threading.Thread] = []
        self._shutdown_event = threading.Event()

        for i in range(self._worker_count):
            print(f"Starting OpenAPIQueue-Worker-{i}")
            print("...")

            t = threading.Thread(
                target=self._worker_loop,
                name=f"OpenAPIQueue-Worker-{i}",
                daemon=True,
            )
            t.start()
            self._workers.append(t)

    @staticmethod
    def _resolve_config_paths(
        configs: List[Path] | List[str] | Path | str,
    ) -> List[Path]:
        """
        Normalize input into a list of Path objects.
        - Accepts a single path (str|Path) or a list of them.
        - If a directory is given, loads all *.json files inside (non-recursive).
        """

        def expand_one(p: "Path | str") -> List[Path]:
            path = Path(p)
            if path.is_dir():
                return sorted(path.glob("*.json"))
            return [path]

        paths: List[Path] = []
        if isinstance(configs, (str, Path)):
            paths.extend(expand_one(configs))
        else:
            for item in configs:
                paths.extend(expand_one(item))

        if not paths:
            raise ValueError("No configuration files found.")
        return paths

    @staticmethod
    def _create_client_from_path(cfg_path: Path | str) -> OpenAPIClient:
        """
        Load a JSON configuration file and return an `OpenAPIClient` instance.
        This helper isolates file‑I/O and client construction logic.
        """
        path = Path(cfg_path)
        with path.open("r", encoding="utf-8") as f:
            cfg = json.load(f)

        return OpenAPIClient(
            base_url=cfg.get("base_url"),
            model=cfg.get("model"),
            api_key=cfg.get("api_key"),
            system_prompt=cfg.get("system_prompt"),
        )

    def _worker_loop(self) -> None:
        """
        Continuously pull tasks from the queue and execute them.
        """
        while not self._shutdown_event.is_set():
            task: Optional[_Task] = self._task_queue.get()
            if task is None:  # Sentinel for shutdown
                break

            try:
                # Select and reserve a client for this task
                client_idx = self._select_client()
                task.client_idx = client_idx
                client_lock = self._client_locks[client_idx]
                client = self._clients[client_idx]

                # Resolve the requested method on the chosen client
                method = getattr(client, task.method_name)
                try:
                    task.result = method(*task.args, **task.kwargs)
                except BaseException as exc:
                    task.exception = exc
                finally:
                    # Release the reserved client
                    client_lock.release()
            finally:
                task.result_event.set()
                self._task_queue.task_done()

    def _select_client(self) -> int:
        """
        Return the index of the first free client (i.e., whose lock is not held),
        and ACQUIRE its lock to reserve it for the enqueued task. If all are busy,
        block until one becomes available.
        """
        while not self._shutdown_event.is_set():
            for idx, lock in enumerate(self._client_locks):
                if lock.acquire(blocking=False):
                    # Do NOT release here — keep it held to reserve the client.
                    return idx
            # No client free – wait a tiny moment before retrying.
            self._shutdown_event.wait(0.05)
        raise RuntimeError("OpenAPIQueue is shutting down; cannot select client.")


class _OpenAPIQueue(_OpenAPIQueueWorkerConfig, abc.ABC):
    def __init__(
        self,
        config_paths: List[Path] | List[str] | Path | str,
        worker_count: Optional[int] = None,
    ) -> None:
        super().__init__(config_paths, worker_count)

    def _submit_task(
        self, method_name: str, args: Tuple[Any, ...], kwargs: dict
    ) -> _Task:
        """
        Create a task, push it to the queue and return immediately.
        """
        task = self._create_task(
            method_name=method_name,
            args=args,
            kwargs=kwargs,
        )
        self._task_queue.put(task)
        return task

    def _enqueue_task(
        self, method_name: str, args: Tuple[Any, ...], kwargs: dict
    ) -> str:
        """
        Create a task, push it to the queue and wait for the result.
        """
        task = self._submit_task(method_name=method_name, args=args, kwargs=kwargs)
        return self._wait_for_task(task)

    @staticmethod
    def _create_task(
        method_name: str,
        args: Tuple[Any, ...],
        kwargs: dict,
    ) -> _Task:
        """
        Assemble a `_Task` instance.
        Centralizing this logic makes it easier to adjust the task payload
        (e.g., add tracing IDs) in one place.
        """
        return _Task(
            client_idx=None,
            method_name=method_name,
            args=args,
            kwargs=kwargs,
            result_event=threading.Event(),
        )

    @staticmethod
    def _wait_for_task(task: _Task) -> str:
        """
        Block until the supplied task signals completion,
        then propagate any exception or return the result.
        """
        task.result_event.wait()  # Block until worker signals completion

        if task.exception:
            raise task.exception  # Propagate errors to the caller
        return task.result or ""


class OpenAPIQueue(_OpenAPIQueue):
    """
    Manages a pool of `OpenAPIClient` objects and processes requests
    through a thread‑safe queue.

    Parameters
    ----------
    config_paths :
        List of paths to JSON configuration files.  Each file must contain
        the keys required by `OpenAPIClient` (e.g. `base_url`, `model`,
        `api_key` and optionally `system_prompt`).
    worker_count :
        Number of background worker threads. Defaults to the number of
        clients (one worker per client) but can be increased for higher
        throughput when the underlying API supports parallelism.
    """

    def __init__(
        self,
        config_paths: List[Path] | List[str] | Path | str,
        worker_count: Optional[int] = None,
    ) -> None:
        super().__init__(config_paths, worker_count)

    def generate(
        self,
        message: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Generate a completion using the first available client.

        The call blocks until the request is processed.
        """
        return self._enqueue_task(
            method_name="generate",
            args=(message,),
            kwargs=dict(
                max_tokens=max_tokens,
                # temperature=temperature,
                system_prompt=system_prompt,
            ),
        )

    def chat(
        self,
        message: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Perform a chat‑style request using the first available client.

        The call blocks until the request is processed.
        """
        return self._enqueue_task(
            method_name="chat",
            args=(message,),
            kwargs=dict(
                max_tokens=max_tokens,
                # temperature=temperature,
                system_prompt=system_prompt,
            ),
        )

    def close(self, timeout: Optional[float] = None) -> None:
        """
        Gracefully shut down all worker threads.

        Parameters
        ----------
        timeout :
            Maximum number of seconds to wait for workers to finish.
            `None` means wait indefinitely.
        """
        self._shutdown_event.set()
        # Unblock workers waiting on an empty queue
        for _ in range(self._worker_count):
            self._task_queue.put_nowait(None)  # type: ignore
        for t in self._workers:
            t.join(timeout=timeout)
