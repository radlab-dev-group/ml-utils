import os
import json
import datetime
import threading

from typing import Optional, Dict, List
from concurrent.futures import ThreadPoolExecutor

from rdl_ml_utils.open_api.queue_manager import OpenAPIQueue
from rdl_ml_utils.handlers.prompt_handler import PromptHandler


class OpenApiHandlerWithCache:
    CACHE_RESULTS = 500

    def __init__(
        self,
        prompts_dir: str,
        prompt_name: str,
        workdir: str,
        openapi_configs_dir: str,
        max_workers: int | None = None,
        cache_results_size: Optional[int] = None,
    ):
        """
        Parameters
        ----------
        prompts_dir : str
            Directory containing prompt files; used by PromptHandler to load the prompt.
        prompt_name : str
            The key/name of the prompt to use as the system prompt for correction.
        max_workers : int, optional
            Maximum number of threads used for parallel correction
            (default Not set). The number of workers as default is equal
            to the number of available services.
        workdir: str, optional
            Working directory, if set, then each 1k corrected texts will be stored
            in the cache file. If a `workdir/translate.cache` file exists, then it will
            be loaded and these translations will not be corrected.

        Raises
        ------
        RuntimeError
            If the global `OPEN_API` is not initialized in the main process.
        KeyError
            If the specified `prompt_name` cannot be found by the PromptHandler.
        """
        self.workdir = workdir
        os.makedirs(self.workdir, exist_ok=True)

        self.prompts_dir = prompts_dir
        self.prompt_name = prompt_name
        self.cache_results_size = cache_results_size or self.CACHE_RESULTS

        config_paths = self._list_json_openapi_configs(openapi_configs_dir)
        self.open_api = OpenAPIQueue(config_paths=config_paths)
        self.max_workers = (
            len(config_paths)
            if max_workers is None or max_workers < 1
            else max_workers
        )

        with PromptHandler(base_dir=prompts_dir) as prompt_handler:
            self.prompt_str = prompt_handler.get_prompt(key=prompt_name)

        self.batch = {}
        self.batch_number = 0
        self._correct_texts: Dict[str, str] = {}
        self.pool = ThreadPoolExecutor(max_workers=max_workers)
        self.lock_map = threading.Lock()

        # Try to load _correct_texts from `workdir`
        loaded_cnt = self._load_cache_from_workdir()
        if loaded_cnt:
            print(f" ~> loaded {loaded_cnt} cached items from {self.workdir}")
            print(f" ~> current batch is set to number {self.batch_number}")

    def generate(self, text_str: str | List[str]) -> Optional[str | List[str]]:
        # with self.lock_map:
        if type(text_str) == list:
            raise NotImplementedError("Lists processing is not yet supported!")

        _ct = self._correct_texts.get(text_str)
        if _ct is not None:
            return _ct

        def _work(txt: str):
            _m_res = self._generate_with_retries(
                message=txt,
                system_prompt=self.prompt_str,
                max_tokens=9192,
            )
            if _m_res is None or not str(_m_res).strip():
                _m_res = txt

            with self.lock_map:
                self._correct_texts[text_str] = _m_res
                self.batch[text_str] = _m_res
                if len(self.batch) >= self.cache_results_size:
                    self._store_cache_batch(batch=self.batch)
                    self.batch.clear()

        _res = []
        res = self.pool.map(_work, [text_str])
        for s in res:
            _res.append(s)
        if len(_res) == 1:
            return _res[0]
        return _res

    def _store_cache_batch(self, batch):
        date_now = datetime.datetime.now()
        data_as_str = date_now.strftime("%Y%m%d_%H%M%S")
        data_as_str += f".{date_now.microsecond:06d}"

        out_f_path = os.path.join(
            self.workdir, f"{data_as_str}__{self.batch_number:05}.json"
        )
        print(f" ~> storing correct text to cache file: {out_f_path}")

        with open(out_f_path, "w") as f:
            json.dump(batch, f, indent=2, ensure_ascii=False)
            self.batch_number += 1

    def _load_cache_from_workdir(self) -> int:
        """
        Scan workdir for .json cache files and load their contents
        into self._correct_texts.

        Returns
        -------
        int
            Number of unique items loaded into the in-memory cache.
        """
        loaded = 0
        max_batch_seen = -1
        workdir_content = sorted(os.listdir(self.workdir))
        for f_name in workdir_content:
            if not f_name.lower().endswith(".json"):
                continue

            fpath = os.path.join(self.workdir, f_name)
            if not os.path.isfile(fpath):
                continue

            with open(fpath, "r") as f:
                data = json.load(f)
            if isinstance(data, dict):
                for k, v in data.items():
                    self._correct_texts[k] = v
                    loaded += 1

            base = os.path.basename(f_name).rstrip(".json")
            if "__" in base:
                batch_str = base.split("__")[-1]
                max_batch_seen = max(max_batch_seen, int(batch_str))

        if max_batch_seen >= 0:
            self.batch_number = max(self.batch_number, max_batch_seen + 1)
        return loaded

    def _generate_with_retries(
        self, message: str, system_prompt: str, max_tokens: int, retries: int = 3
    ) -> Optional[str]:
        """
        Call open_api.generate with basic retry logic.

        The function attempts to generate a response using the global OPEN_API
        client up to `retries` times. Any exceptions raised by the client are
        caught and ignored to allow subsequent attempts. If all attempts fail,
        the function returns None.

        Parameters
        ----------
        message : str
            The user message (input text) to send to the generator.
        system_prompt : str
            The system prompt guiding the generation.
        max_tokens : int
            The maximum number of tokens to generate.
        retries : int, optional
            Number of attempts before giving up (default: 3).

        Returns
        -------
        Optional[str]
            The generated string on success; None if all attempts failed.

        Notes
        -----
        - This function relies on the global `OPEN_API` object exposing a
          `generate(message, system_prompt, max_tokens)` method.
        - Exceptions from OPEN_API.generate are intentionally silenced.
        """
        for attempt in range(1, retries + 1):
            try:
                return self.open_api.generate(
                    message=message,
                    system_prompt=system_prompt,
                    max_tokens=max_tokens,
                )
            except Exception as e:
                pass
        return None

    def __del__(self):
        """
        Ensure any remaining cached batch is persisted when the handler
        is destroyed. Also attempts to shut down the thread pool gracefully.
        """
        try:
            if self.pool is not None:
                self.pool.shutdown(wait=True)
        except Exception:
            pass

        try:
            if self.batch is not None and len(self.batch):
                with self.lock_map:
                    if self.batch:
                        self._store_cache_batch(batch=self.batch)
                        self.batch.clear()
        except Exception:
            pass

    @staticmethod
    def _list_json_openapi_configs(look_dir: str) -> List[str]:
        """
        Return absolute paths of JSON OpenAPI configuration files in a directory.

        Parameters
        ----------
        look_dir : str
            Path to the directory to scan for JSON configuration files.

        Returns
        -------
        List[str]
            A list of absolute file paths for all entries whose names end with
            ".json" (case-insensitive) found directly in `look_dir`. The order
            reflects the underlying directory listing and is not guaranteed.

        Raises
        ------
        FileNotFoundError
            If `look_dir` does not exist or is not a directory.

        Notes
        -----
        Only files with names ending in ".json" (case-insensitive) are included.
        Paths in the returned list are absolute.
        """
        if not os.path.isdir(look_dir):
            raise FileNotFoundError(f"Directory not found: {look_dir}")

        json_paths = [
            os.path.abspath(os.path.join(look_dir, filename))
            for filename in os.listdir(look_dir)
            if filename.lower().endswith(".json")
        ]
        return json_paths
