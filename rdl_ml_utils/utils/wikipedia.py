import re
import os
import json
import requests
import datetime

from typing import Optional, Dict, Any
from urllib.parse import urlparse, parse_qs, unquote, quote

from rdl_ml_utils.utils.logger import prepare_logger


class WikipediaExtractor:
    """
    Extractor for fetching main description content from Wikipedia articles.
    """

    CACHE_RESULTS = 500

    def __init__(
        self,
        timeout: int = 10,
        max_sentences: int = 3,
        cache_dir: Optional[str] = None,
        cache_results_size: Optional[int] = None,
        logger_file: Optional[str] = None,
    ):
        """
        Initialize Wikipedia content extractor with optional content clear.

        Args:
            timeout: Request timeout in seconds
            max_sentences: Maximum number of sentences
                           to extract from the main description
            cache_dir: (Optional) Directory to cache results (extracted content)
            cache_results_size: (Optional) Size of cache results (extracted content)
        """
        self.timeout = timeout
        self.max_sentences = max_sentences
        self.logger = prepare_logger(
            logger_name=__name__,
            logger_file_name=logger_file or "wiki-extractor.log",
        )

        self.cache_dir = cache_dir or "__cache/wikipedia/raw"
        os.makedirs(self.cache_dir, exist_ok=True)

        self._cached_content = {}
        self._content_batch = {}
        self.cache_results_size = cache_results_size or self.CACHE_RESULTS

        # Session for connection reuse
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "PlWordnet-Handler/1.0 "
                "(https://github.com/radlab-dev-group/"
                "radlab-plwordnet; pawel@radlab.dev)"
            }
        )

        loaded_cnt = self._load_cache_from_workdir()
        if loaded_cnt:
            print(f" ~> loaded {loaded_cnt} cached items from {self.cache_dir}")

    def extract_main_description(
        self, wikipedia_url: str, elem_type: Optional[str] = ""
    ) -> Optional[str]:
        """
        Extract the main description from a Wikipedia article.

        Args:
            wikipedia_url: URL to Wikipedia article
            elem_type: Additional "debug" info to show

        Returns:
            Main description text or None if extraction failed
        """
        self.logger.debug(f"Processing article {wikipedia_url}")
        _cached = self._cached_content.get(wikipedia_url)
        if _cached is not None:
            return _cached

        try:
            article_title = self._extract_article_title(wikipedia_url=wikipedia_url)
            if not article_title:
                self.logger.error(
                    f"Could not extract article title from URL: {wikipedia_url}"
                )
                return None

            language = self._extract_language_from_url(wikipedia_url=wikipedia_url)
            if not language:
                self.logger.error(
                    f"Could not determine language from URL: {wikipedia_url}"
                )
                return None

            content = self._fetch_article_content(
                article_title=article_title, language=language, elem_type=elem_type
            )
            if not content:
                return None

            main_description = self._extract_and_clean_description(content=content)

            self._cached_content[wikipedia_url] = main_description
            self._content_batch[wikipedia_url] = main_description
            if len(self._content_batch) >= self.cache_results_size:
                self._store_cache_batch(batch=self._content_batch)
                self._content_batch.clear()

            return main_description

        except Exception as e:
            self.logger.error(
                f"Error extracting description from {wikipedia_url}: {e}"
            )
            return None

    def get_article_info(self, wikipedia_url: str) -> Optional[Dict[str, Any]]:
        """
        Get comprehensive information about a Wikipedia article.

        Args:
            wikipedia_url: Wikipedia URL

        Returns:
            Dictionary with article information or None if failed
        """
        try:
            article_title = self._extract_article_title(wikipedia_url=wikipedia_url)
            language = self._extract_language_from_url(wikipedia_url=wikipedia_url)
            if not article_title or not language:
                return None

            description = self.extract_main_description(wikipedia_url=wikipedia_url)
            return {
                "url": wikipedia_url,
                "title": article_title,
                "language": language,
                "description": description,
                "is_valid": description is not None,
            }

        except Exception as e:
            self.logger.error(f"Error getting article info for {wikipedia_url}: {e}")
            return None

    def close(self):
        """
        Close the session.
        """
        if self.session:
            self.session.close()

    @staticmethod
    def _split_into_sentences(text: str) -> list[str]:
        """
        Split text into sentences.

        Simple sentence splitting can be improved with more sophisticated methods.
        Handle common abbreviations that shouldn't trigger sentence breaks.

        Args:
            text: Input text

        Returns:
            List of sentences
        """
        text = re.sub(
            r"\b(np|tzn|tj|itp|itd|por|zob|ang|łac|gr|fr|niem|ros)\.\s*",
            r"\1._ABBREV_",
            text,
        )
        sentences = re.split(r"[.!?]+\s+", text)
        sentences = [s.replace("_ABBREV_", ".") for s in sentences]
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        return sentences

    @staticmethod
    def _clean_text(text: str) -> str:
        """
        Clean and normalize text.

        Args:
            text: Input text

        Returns:
            Cleaned text
        """
        if not text:
            return ""

        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text)
        # Remove parenthetical notes at the beginning (common in Wikipedia)
        text = re.sub(r"^\([^)]*\)\s*", "", text)
        # Clean up common Wikipedia artifacts
        #  - remove reference markers
        text = re.sub(r"\[.*?\]", "", text)
        #  - remove template markers
        text = re.sub(r"\{.*?\}", "", text)
        # Normalize punctuation
        text = re.sub(r"\s+([,.!?;:])", r"\1", text)
        return text.strip()

    @staticmethod
    def is_valid_wikipedia_url(url: str) -> bool:
        """
        Check if URL is a valid Wikipedia URL.

        Args:
            url: URL to validate

        Returns:
            True if valid Wikipedia URL, False otherwise
        """
        try:
            parsed_url = urlparse(url)
            return "wikipedia.org" in parsed_url.netloc and (
                parsed_url.path.startswith("/wiki/") or "title=" in parsed_url.query
            )
        except Exception:
            return False

    def _extract_article_title(self, wikipedia_url: str) -> Optional[str]:
        """
        Extract article title from Wikipedia URL.

        Args:
            wikipedia_url: Wikipedia URL

        Returns:
            Article title or None if extraction failed
        """
        try:
            parsed_url = urlparse(wikipedia_url)

            if "/wiki/" in parsed_url.path:
                # Standard format:
                #  https://pl.wikipedia.org/wiki/Article_Title
                title = parsed_url.path.split("/wiki/")[-1]
                return unquote(title)
            elif "title=" in parsed_url.query:
                # Query format:
                #  https://pl.wikipedia.org/w/index.php?title=Article_Title
                query_params = parse_qs(parsed_url.query)
                if "title" in query_params:
                    return query_params["title"][0]
            return None
        except Exception as e:
            self.logger.error(
                f"Error extracting title from URL {wikipedia_url}: {e}"
            )
            return None

    def _extract_language_from_url(self, wikipedia_url: str) -> Optional[str]:
        """
        Extract language code from Wikipedia URL.

        Args:
            wikipedia_url: Wikipedia URL

        Returns:
            Language code (e.g., 'pl', 'en') or None if extraction failed
        """
        try:
            parsed_url = urlparse(wikipedia_url)
            if "wikipedia.org" in parsed_url.netloc:
                parts = parsed_url.netloc.split(".")
                if len(parts) >= 3 and parts[1] == "wikipedia":
                    return parts[0]
            return None
        except Exception as e:
            self.logger.error(
                f"Error extracting language from URL {wikipedia_url}: {e}"
            )
            return None

    def _fetch_article_content(
        self, article_title: str, language: str, elem_type: str
    ) -> Optional[str]:
        """
        Fetch article content using Wikipedia API.

        Args:
            article_title: Title of the Wikipedia article
            language: Language code
            elem_type: Additional info to show in debug

        Returns:
            Article content or None if fetch failed
        """
        try:
            api_url = self._build_api_url(language)
            params = self._build_query_params(article_title)

            data = self._request_json(api_url, params=params)
            if data is None:
                self.logger.warning(
                    f"[{elem_type}] Empty response for: {article_title}"
                )
                return None

            extract, redirected_to, missing = self._parse_extract_from_pages(
                data=data, elem_type=elem_type, article_title=article_title
            )
            if missing:
                return None

            # Retry with redirected title if extract is empty
            # and we have a redirect target
            if not extract and redirected_to:
                extract = self._retry_extract_with_redirect(
                    api_url=api_url,
                    base_params=params,
                    redirected_to=redirected_to,
                    elem_type=elem_type,
                    original_title=article_title,
                )

            # Fallback to REST Summary API
            if not extract:
                extract = self._fetch_summary_rest(
                    language=language,
                    title=redirected_to or article_title,
                    elem_type=elem_type,
                )

            if not extract:
                self.logger.warning(
                    f"[{elem_type}] No extract found for: "
                    f"{redirected_to or article_title}"
                )
                return None

            return extract

        except requests.RequestException as e:
            self.logger.error(
                f"[{elem_type}] Network error fetching article {article_title}: {e}"
            )
            return None
        except json.JSONDecodeError as e:
            self.logger.error(
                f"[{elem_type}] JSON decode error for article {article_title}: {e}"
            )
            return None
        except Exception as e:
            self.logger.error(
                f"[{elem_type}] Unexpected error fetching article "
                f"{article_title}: {e}"
            )
            return None

    @staticmethod
    def _build_api_url(language: str) -> str:
        return f"https://{language}.wikipedia.org/w/api.php"

    @staticmethod
    def _build_query_params(article_title: str) -> Dict[str, Any]:
        """
        API parameters to get article content
        """
        return {
            "action": "query",
            "format": "json",
            "titles": article_title,
            "prop": "extracts",
            "exintro": True,
            "explaintext": True,
            "exsectionformat": "plain",
            "redirects": 1,
        }

    def _request_json(
        self,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Optional[Dict[str, Any]]:
        resp = self.session.get(
            url, params=params, headers=headers, timeout=self.timeout
        )
        resp.raise_for_status()
        return resp.json()

    def _parse_extract_from_pages(
        self,
        data: Dict[str, Any],
        elem_type: str,
        article_title: str,
    ) -> tuple[Optional[str], Optional[str], bool]:
        """
        Returns:
            (extract, redirected_to, missing)
            missing=True when page explicitly marked as missing
        """
        pages = data.get("query", {}).get("pages", {})
        if not pages:
            self.logger.warning(
                f"[{elem_type}] No pages found for title: {article_title}"
            )
            return None, None, True

        # If API informs about redirects,
        # remember the target title (for retry/fallback)
        redirects_info = data.get("query", {}).get("redirects", [])
        redirected_to = redirects_info[0].get("to") if redirects_info else None
        if redirected_to and redirected_to != article_title:
            self.logger.debug(
                f"[{elem_type}] Title redirected: "
                f"'{article_title}' -> '{redirected_to}'"
            )

        page_id = next(iter(pages))
        page_data = pages[page_id]
        if "missing" in page_data:
            self.logger.warning(f"[{elem_type}] Page not found: {article_title}")
            return None, redirected_to, True

        extract = page_data.get("extract", "")
        return extract or None, redirected_to, False

    def _retry_extract_with_redirect(
        self,
        api_url: str,
        base_params: Dict[str, Any],
        redirected_to: str,
        elem_type: str,
        original_title: str,
    ) -> Optional[str]:
        self.logger.debug(
            f"[{elem_type}] Empty extract for '{original_title}', "
            f"retrying with redirected title '{redirected_to}'"
        )
        retry_params = dict(base_params)
        retry_params["titles"] = redirected_to

        retry_data = self._request_json(api_url, params=retry_params)
        if not retry_data:
            return None

        retry_pages = retry_data.get("query", {}).get("pages", {})
        if not retry_pages:
            return None

        retry_page_id = next(iter(retry_pages))
        retry_page_data = retry_pages[retry_page_id]
        if "missing" in retry_page_data:
            return None
        return retry_page_data.get("extract") or None

    def _fetch_summary_rest(
        self, language: str, title: str, elem_type: str
    ) -> Optional[str]:
        rest_url = (
            f"https://{language}.wikipedia.org/api/"
            f"rest_v1/page/summary/{quote(title, safe='')}"
        )
        try:
            rest_data = self._request_json(
                rest_url,
                params={"redirect": "true"},
                headers={"accept": "application/json"},
            )
            extract = (rest_data or {}).get("extract") or ""
            if extract:
                self.logger.debug(
                    f"[{elem_type}] Extract obtained via REST Summary API for '{title}'"
                )
            return extract or None
        except requests.RequestException as rexc:
            self.logger.debug(
                f"[{elem_type}] REST Summary fallback failed for '{title}': {rexc}"
            )
            return None

    def _extract_and_clean_description(self, content: str) -> str:
        """
        Extract and clean the main description from article content.

        Args:
            content: Raw article content

        Returns:
            Cleaned main description
        """
        if not content:
            return ""

        sentences = self._split_into_sentences(text=content)
        main_sentences = sentences[: self.max_sentences]
        description = " ".join(main_sentences)
        description = self._clean_text(text=description)

        return description

    def _store_cache_batch(self, batch):
        """
        Persist the current in-memory batch to a timestamped JSON file.

        This method serializes the provided key-value pairs (input text -> corrected
        text) to a file located in the configured working directory. The output
        filename encodes a high-resolution timestamp:
          {YYYYMMDD_HHMMSS}.{microseconds}.json

        Notes
        -----
        - The working directory is expected to exist (created during initialization).
        - JSON is written with indentation (2 spaces), and ensure_ascii=False to
          preserve non-ASCII characters.
        - Thread-safety: callers should guard concurrent invocations (e.g., via
          `self.lock_map`) to avoid interleaved writes.

        Parameters
        ----------
        batch : dict
            Mapping of source strings to their corrected results to be persisted.

        Returns
        -------
        None
        """
        date_now = datetime.datetime.now()
        data_as_str = date_now.strftime("%Y%m%d_%H%M%S")
        data_as_str += f".{date_now.microsecond:06d}"
        out_f_path = os.path.join(self.cache_dir, f"{data_as_str}.json")

        with open(out_f_path, "w") as f:
            print(f" ~> storing content to cache file: {out_f_path}")
            json.dump(batch, f, indent=2, ensure_ascii=False)

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
        workdir_content = sorted(os.listdir(self.cache_dir))
        for f_name in workdir_content:
            if not f_name.lower().endswith(".json"):
                continue

            fpath = os.path.join(self.cache_dir, f_name)
            if not os.path.isfile(fpath):
                continue

            with open(fpath, "r") as f:
                data = json.load(f)
            if isinstance(data, dict):
                for k, v in data.items():
                    self._cached_content[k] = v
                    loaded += 1
        return loaded

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def __del__(self):
        if self._content_batch is None or not len(self._content_batch):
            return
        self._store_cache_batch(batch=self._content_batch)
        self._content_batch.clear()


def is_wikipedia_url(url: str) -> bool:
    """
    Utility function to check if URL is a Wikipedia URL.

    Args:
        url: URL to check

    Returns:
        True if Wikipedia URL, False otherwise
    """
    return WikipediaExtractor().is_valid_wikipedia_url(url)


def extract_wikipedia_description(url: str, max_sentences: int = 3) -> Optional[str]:
    """
    Utility function to extract Wikipedia description.

    Args:
        url: Wikipedia URL
        max_sentences: Maximum number of sentences to extract

    Returns:
        Main description or None if extraction failed
    """
    if not is_wikipedia_url(url=url):
        return None

    with WikipediaExtractor(max_sentences=max_sentences) as extractor:
        return extractor.extract_main_description(url)
