"""llms-txt-mcp: Lean Documentation MCP via llms.txt.

Implements three focused tools:
- docs_query: Primary unified search + retrieval interface with auto-retrieve
- docs_sources: List indexed documentation sources
- docs_refresh: Force refresh cached documentation

Parses both AI SDK YAML-frontmatter llms.txt and official llms.txt headings.
Embeds with BAAI/bge-small-en-v1.5 into a unified Chroma collection with host metadata.
Enforces TTL + ETag/Last-Modified for freshness. Ephemeral by default; optional disk store.

Follows latest FastMCP patterns from the MCP Python SDK.
"""

from __future__ import annotations

# /// script
# dependencies = [
#     "mcp>=1.0.0",
#     "httpx>=0.27.0",
#     "sentence-transformers>=3.0.0",
#     "chromadb>=0.5.0",
#     "pyyaml>=6.0.0"
# ]
# ///
import argparse
import asyncio
import dataclasses
import hashlib
import logging
import os
import re
import signal
import sys
import time
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast
from urllib.parse import urlparse

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Mapping

try:
    from importlib.metadata import version

    __version__ = version("llms-txt-mcp")
except ImportError:
    # Fallback for development/editable installs when importlib.metadata fails
    try:
        from ._version import __version__
    except ImportError:
        # Final fallback if version file doesn't exist (development mode)
        __version__ = "0.1.0-dev"

import chromadb
import httpx
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.server import Context
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

from .parser import ParsedDoc, parse_llms_txt

try:
    import anthropic

    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    anthropic = None  # type: ignore

try:
    # Chroma telemetry settings (0.5+)
    from chromadb.config import Settings as ChromaSettings  # type: ignore
except Exception:  # pragma: no cover - fallback if import path changes
    ChromaSettings = None  # type: ignore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],
)
logger = logging.getLogger("llms-txt-mcp")


# -------------------------
# Type definitions and aliases
# -------------------------

# Type aliases for ChromaDB to improve type safety
ChromaWhere = dict[str, Any]  # ChromaDB where clause
ChromaIncludeParam = Literal["documents", "embeddings", "metadatas", "distances", "uris", "data"]
ChromaInclude = list[ChromaIncludeParam]  # ChromaDB include parameter
ChromaIds = list[str]  # ChromaDB document IDs


class ChromaMetadata(BaseModel):
    """Metadata stored in Chroma collection."""

    id: str
    source: str
    host: str
    title: str
    description: str = ""
    content: str
    requested_url: str = ""
    content_hash: str = ""
    section_index: int = 0
    indexed_at: float = 0


# -------------------------
# Data models
# -------------------------


@dataclass
class Config:
    """Server configuration."""

    allowed_urls: set[str]
    ttl_seconds: int
    timeout: int
    embed_model_name: str
    store_mode: str
    store_path: str | None
    max_get_bytes: int
    auto_retrieve_threshold: float
    auto_retrieve_limit: int
    include_snippets: bool
    preindex: bool
    background_preindex: bool
    max_response_tokens: int


@dataclass
class SourceState:
    source_url: str
    host: str
    etag: str | None
    last_modified: str | None
    last_indexed: float
    doc_count: int
    actual_url: str | None = None  # Track if auto-upgraded to llms-full.txt


# -------------------------
# Pydantic models for API responses
# -------------------------


class SourceInfo(BaseModel):
    """Information about an indexed documentation source."""

    source_url: str
    host: str
    lastIndexed: int
    docCount: int


class SearchResult(BaseModel):
    """A single search result from semantic search."""

    id: str
    source: str
    title: str
    description: str = ""
    score: float
    auto_retrieved: bool = False
    snippet: str = ""


class DocContent(BaseModel):
    """Retrieved document content."""

    id: str
    source: str
    host: str
    title: str
    content: str


class RefreshResult(BaseModel):
    """Result of refreshing documentation sources."""

    refreshed: list[str]
    counts: dict[str, int]


class QueryResult(BaseModel):
    """Combined search and retrieval result."""

    search_results: list[SearchResult]
    retrieved_content: dict[str, DocContent] = {}
    merged_content: str = ""
    auto_retrieved_count: int = 0
    total_results: int = 0
    total_tokens: int | None = None
    truncated_due_to_tokens: bool = False


# -------------------------
# Resource Manager (replaces global state)
# -------------------------


class ResourceManager:
    """Manages server resources with async initialization."""

    def __init__(self, http_client: httpx.AsyncClient, config: Config):
        self.http_client = http_client
        self.config = config

        # Resources (initialized asynchronously)
        self.embedding_model: SentenceTransformer | None = None
        self.chroma_client: chromadb.ClientAPI | None = None
        self.chroma_collection: chromadb.Collection | None = None
        self.index_manager: IndexManager | None = None

        # Async coordination
        self._model_ready = asyncio.Event()
        self._db_ready = asyncio.Event()
        self._index_ready = asyncio.Event()
        self._all_ready = asyncio.Event()
        self._init_error: Exception | None = None

    async def initialize_heavy_resources(self) -> None:
        """Initialize embedding model, Chroma DB, and preindex concurrently."""
        logger.info("Starting background resource initialization...")

        try:
            # Initialize model and DB concurrently
            await asyncio.gather(
                self._load_embedding_model(), self._init_chroma_db(), return_exceptions=False
            )

            # Then preindex after both are ready
            await self._preindex_sources()

        except Exception as e:
            self._init_error = e
            logger.error(f"Resource initialization failed: {e}")
        finally:
            self._all_ready.set()

    async def _load_embedding_model(self) -> None:
        """Load the SentenceTransformer model."""
        try:
            logger.info(f"Loading embedding model: {self.config.embed_model_name}")
            self.embedding_model = SentenceTransformer(self.config.embed_model_name)
            logger.info("Embedding model loaded successfully")
            self._model_ready.set()
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise

    async def _init_chroma_db(self) -> None:
        """Initialize Chroma database client."""
        try:
            if self.config.store_mode == "disk":
                assert self.config.store_path is not None
                if ChromaSettings is not None:
                    self.chroma_client = chromadb.PersistentClient(
                        path=self.config.store_path,
                        settings=ChromaSettings(anonymized_telemetry=False),
                    )
                else:
                    self.chroma_client = chromadb.PersistentClient(path=self.config.store_path)
                logger.info(f"ChromaDB PersistentClient initialized at {self.config.store_path}")
            else:
                if ChromaSettings is not None:
                    self.chroma_client = chromadb.Client(
                        settings=ChromaSettings(anonymized_telemetry=False)
                    )
                else:
                    self.chroma_client = chromadb.Client()
                logger.info("ChromaDB ephemeral client initialized")

            self._db_ready.set()
        except Exception as e:
            logger.error(f"Failed to initialize Chroma DB: {e}")
            raise

    async def _preindex_sources(self) -> None:
        """Initialize IndexManager and preindex sources if configured."""
        try:
            # Wait for both model and DB to be ready
            await self._model_ready.wait()
            await self._db_ready.wait()

            if self.embedding_model is None or self.chroma_client is None:
                raise RuntimeError("Model or DB not properly initialized")

            # Initialize index manager
            self.index_manager = IndexManager(
                ttl_seconds=self.config.ttl_seconds,
                max_get_bytes=self.config.max_get_bytes,
                embedding_model=self.embedding_model,
                chroma_client=self.chroma_client,
                config=self.config,
            )

            # Clean up expired documents if using disk storage
            if self.config.store_mode == "disk":
                try:
                    cleaned_up = await self.index_manager.cleanup_expired_documents()
                    if cleaned_up > 0:
                        logger.info(f"Startup cleanup: removed {cleaned_up} expired documents")
                except Exception as e:
                    logger.debug(f"Startup cleanup skipped: {e}")

            # Preindex if configured
            if self.config.preindex:
                await self._run_preindexing()

            self._index_ready.set()
            logger.info("Resource initialization complete")

        except Exception as e:
            logger.error(f"Failed to initialize IndexManager or preindex: {e}")
            raise

    async def _run_preindexing(self) -> None:
        """Run preindexing of configured sources."""
        if self.index_manager is None:
            raise RuntimeError("IndexManager not initialized")

        total = len(self.config.allowed_urls)
        start = time.time()
        logger.info("Preindexing %d source(s)...", total)

        for i, url in enumerate(self.config.allowed_urls, 1):
            logger.info(f"Fetching {url} ({i}/{total})...")
            await self.index_manager.maybe_refresh(url, force=True, http_client=self.http_client)

        total_docs = sum(st.doc_count for st in self.index_manager.sources.values())
        indexed_count = len([st for st in self.index_manager.sources.values() if st.doc_count > 0])

        logger.info(
            "Indexing complete: %d sections from %d/%d sources (%.1fs)",
            total_docs,
            indexed_count,
            total,
            time.time() - start,
        )

        # Display summary table
        self._display_indexing_summary()

    def _display_indexing_summary(self) -> None:
        """Display a summary table of indexed sources."""
        if not self.index_manager or not self.index_manager.sources:
            return

        logger.info(SUMMARY_SEPARATOR)
        logger.info("Indexing Summary:")
        logger.info(SUMMARY_SEPARATOR)

        for source_url, state in self.index_manager.sources.items():
            display_url = source_url
            if state.actual_url and state.actual_url != source_url:
                file_type = (
                    "llms-full.txt"
                    if state.actual_url.endswith("/llms-full.txt")
                    else state.actual_url.split("/")[-1]
                )
                display_url = f"{source_url} → {file_type}"

            logger.info(f"{display_url} | {state.doc_count} sections")

        logger.info(SUMMARY_SEPARATOR)

    async def ensure_ready(self, timeout: float = 1.0) -> None:
        """Wait for all resources to be ready within timeout."""
        try:
            await asyncio.wait_for(self._all_ready.wait(), timeout=timeout)
            if self._init_error:
                raise self._init_error
        except TimeoutError:
            if not self._all_ready.is_set():
                raise TimeoutError("Resources still initializing") from None
            raise

    def is_ready(self) -> bool:
        """Check if all resources are ready without waiting."""
        return self._all_ready.is_set() and self._init_error is None


# -------------------------
# Constants
# -------------------------

# Timeout values (in seconds)
RESOURCE_INIT_TIMEOUT = 30.0  # Maximum time to wait for resource initialization
RESOURCE_WAIT_TIMEOUT = 1.0  # Default timeout when waiting for resources in tools
HTTP_CLOSE_TIMEOUT = 2.0  # Timeout for closing HTTP client

# Limits and thresholds
DEFAULT_MAX_GET_BYTES = 75000  # Default byte cap for document retrieval
DEFAULT_AUTO_RETRIEVE_THRESHOLD = 0.1  # Default score threshold for auto-retrieval
DEFAULT_AUTO_RETRIEVE_LIMIT = 5  # Default max number of docs to auto-retrieve
DEFAULT_TTL_HOURS = 24  # Default TTL for cached documents
DEFAULT_HTTP_TIMEOUT = 30  # Default HTTP request timeout in seconds
DEFAULT_MAX_RESPONSE_TOKENS = 25000  # Default max tokens in MCP response
# Reserve 1000 tokens for JSON structure, metadata fields, and response formatting
# This ensures the actual content + JSON wrapper stays under Claude's context limit
TOKEN_LIMIT_BUFFER = 1000  # Buffer to account for JSON structure overhead

# Display constants
SUMMARY_SEPARATOR = "=" * 60  # Separator for summary displays


# -------------------------
# FastMCP Server
# -------------------------


mcp = FastMCP(
    "llms-txt-mcp",
    dependencies=[
        "httpx>=0.27.0",
        "sentence-transformers>=3.0.0",
        "chromadb>=0.5.0",
        "pyyaml>=6.0.0",
    ],
)

# Global resource manager for MCP tool access
# This is initialized once during server startup and shared across all tool calls.
# This pattern is necessary because FastMCP tools are module-level functions
# that cannot receive custom dependency injection beyond the MCP Context.
# The resource manager lifecycle matches the server lifecycle exactly.
resource_manager: ResourceManager | None = None


def _ensure_resource_manager() -> ResourceManager:
    """Ensure resource manager is available, with clear error message.

    Returns:
        ResourceManager: The initialized resource manager

    Raises:
        RuntimeError: If the resource manager is not initialized
    """
    if resource_manager is None:
        raise RuntimeError(
            "Server resources not initialized. This typically means the server "
            "is still starting up or there was an initialization error."
        )
    return resource_manager


# -------------------------
# Token Counting
# -------------------------


class TokenCounter:
    """Utility class for counting tokens using Anthropic's API."""

    def __init__(self, model: str = "claude-3-5-sonnet-20241022"):
        """Initialize the token counter.

        Args:
            model: The Claude model to use for tokenization.
        """
        self.model = model
        self.client: anthropic.Anthropic | None = None
        self.enabled = ANTHROPIC_AVAILABLE

        if self.enabled:
            try:
                # API key is optional for token counting
                # Will use ANTHROPIC_API_KEY env var if set
                api_key = os.environ.get("ANTHROPIC_API_KEY", "dummy-key-for-counting")
                self.client = anthropic.Anthropic(api_key=api_key)
            except Exception as e:
                logger.warning(f"Failed to initialize Anthropic client for token counting: {e}")
                self.enabled = False

    def count_tokens(self, text: str) -> int:
        """Count tokens in the given text.

        Args:
            text: The text to count tokens for.

        Returns:
            Number of tokens, or 0 if counting fails.
        """
        if not self.enabled or not self.client:
            # Fallback: rough estimate based on character count
            # Approximation: ~4 chars per token for English text
            return len(text) // 4

        try:
            response = self.client.messages.count_tokens(
                model=self.model, messages=[{"role": "user", "content": text}]
            )
            # The response contains input_tokens field
            return response.input_tokens
        except Exception as e:
            logger.debug(f"Token counting failed, using estimate: {e}")
            # Fallback to rough estimate
            return len(text) // 4


# -------------------------
# Utilities
# -------------------------


def parse_duration_to_seconds(duration: str) -> int:
    match = re.fullmatch(r"(\d+)([smhd])", duration.strip(), re.IGNORECASE)
    if not match:
        raise ValueError("Invalid duration. Use formats like '30s', '15m', '24h', '7d'")
    value, unit = match.groups()
    value_int = int(value)
    unit = unit.lower()
    if unit == "s":
        return value_int
    if unit == "m":
        return value_int * 60
    if unit == "h":
        return value_int * 3600
    if unit == "d":
        return value_int * 86400
    raise ValueError("Unsupported duration unit")


def slugify(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-")
    return text or "section"


def canonical_id(source_url: str, title: str) -> str:
    return f"{source_url}#{slugify(title)}"


def get_content_hash(content: str, max_bytes: int = 1024) -> str:
    """Create a hash of content for change detection.

    Uses MD5 hash of first max_bytes for performance.
    This is NOT for security, just for change detection.
    """
    sample = content[:max_bytes].encode("utf-8")
    return hashlib.md5(sample).hexdigest()[:12]  # 12 chars is enough for our use case


def host_of(url: str) -> str:
    return urlparse(url).netloc


def extract_snippet(content: str, query_terms: list[str], max_length: int = 200) -> str:
    """Extract a relevant snippet from content based on query terms."""
    if not content or not query_terms:
        return content[:max_length] + "..." if len(content) > max_length else content

    content_lower = content.lower()
    query_lower = [term.lower() for term in query_terms]

    # Find the first occurrence of any query term
    best_pos = len(content)
    for term in query_lower:
        pos = content_lower.find(term)
        if pos != -1 and pos < best_pos:
            best_pos = pos

    if best_pos == len(content):
        # No query terms found, return beginning
        return content[:max_length] + "..." if len(content) > max_length else content

    # Extract snippet around the found term
    start = max(0, best_pos - 50)
    end = min(len(content), start + max_length)
    snippet = content[start:end]

    # Add ellipsis if truncated
    if start > 0:
        snippet = "..." + snippet
    if end < len(content):
        snippet = snippet + "..."

    return snippet


# -------------------------
# Index Manager
# -------------------------


class IndexManager:
    def __init__(
        self,
        ttl_seconds: int,
        max_get_bytes: int,
        embedding_model: SentenceTransformer,
        chroma_client: chromadb.ClientAPI,
        config: Config,
    ) -> None:
        self.ttl_seconds = ttl_seconds
        self.max_get_bytes = max_get_bytes
        self.embedding_model = embedding_model
        self.chroma_client = chroma_client
        self.config = config
        self.chroma_collection: chromadb.Collection | None = None
        self.sources: dict[str, SourceState] = {}
        self.token_counter = TokenCounter()

    def ensure_collection(self) -> chromadb.Collection:
        if self.chroma_collection is None:
            self.chroma_collection = self.chroma_client.get_or_create_collection(
                name="docs",
                metadata={"purpose": "llms-txt-mcp"},
                embedding_function=None,
            )
        return self.chroma_collection

    async def maybe_refresh(
        self, source_url: str, force: bool = False, http_client: httpx.AsyncClient | None = None
    ) -> None:
        if http_client is None:
            raise RuntimeError("HTTP client is required")
        now = time.time()
        st = self.sources.get(source_url)
        if st and not force and (now - st.last_indexed) < self.ttl_seconds:
            return
        await self._index_source(source_url, st, http_client)

    async def _stream_lines(
        self, url: str, headers: dict[str, str], http_client: httpx.AsyncClient
    ) -> tuple[AsyncIterator[str], dict[str, str]]:
        # Stream bytes and decode into lines incrementally
        async def line_iter() -> AsyncIterator[str]:
            decoder = re.compile("\\r?\\n")
            buffer = ""
            async with http_client.stream("GET", url, headers=headers) as resp:
                if resp.status_code == 304:
                    # No body to stream
                    return
                resp.raise_for_status()
                nonlocal_headers.update(
                    {
                        "ETag": resp.headers.get("ETag", ""),
                        "Last-Modified": resp.headers.get("Last-Modified", ""),
                    }
                )
                async for chunk in resp.aiter_bytes():
                    if not chunk:
                        continue
                    buffer += chunk.decode("utf-8", errors="ignore")
                    # Split into lines while keeping remainder in buffer
                    parts = decoder.split(buffer)
                    # If buffer ends with newline, no remainder; else keep last part
                    if buffer.endswith(("\n", "\r")):
                        buffer = ""
                        lines_to_yield = parts
                        if lines_to_yield and lines_to_yield[-1] == "":
                            lines_to_yield = lines_to_yield[:-1]
                    else:
                        buffer = parts[-1]
                        lines_to_yield = parts[:-1]
                    for ln in lines_to_yield:
                        yield ln
                if buffer:
                    # Flush remaining buffered text as a final line
                    yield buffer

        nonlocal_headers: dict[str, str] = {}
        # Return iterator and headers dict (to be populated during iteration)
        return line_iter(), nonlocal_headers

    async def _fetch_and_parse_sections(
        self, url: str, etag: str | None, last_modified: str | None, http_client: httpx.AsyncClient
    ) -> tuple[int, list[ParsedDoc], str | None, str | None, str]:
        """Fetch and parse llms.txt with auto-discovery for llms-full.txt.

        Returns: (status_code, sections, etag, last_modified, actual_url_used)
        """
        headers: dict[str, str] = {}
        if etag:
            headers["If-None-Match"] = etag
        if last_modified:
            headers["If-Modified-Since"] = last_modified

        # Try llms-full.txt first if URL ends with llms.txt
        urls_to_try: list[str] = []
        if url.endswith("/llms.txt"):
            base_url = url[:-9]  # Remove /llms.txt
            urls_to_try = [f"{base_url}/llms-full.txt", url]
        else:
            urls_to_try = [url]

        for try_url in urls_to_try:
            try:
                lines_iter, hdrs = await self._stream_lines(
                    try_url, headers if try_url == url else {}, http_client
                )

                # Collect and parse content
                all_lines = [line async for line in lines_iter]
                if not all_lines:
                    continue

                result = parse_llms_txt("\n".join(all_lines))
                sections = result.docs
                format_type = result.format

                # Create short format name for logging
                format_name = format_type.replace("-llms-txt", "").replace("-full", "")

                # Log format: original_url → actual_file [format] sections
                file_type = "llms-full.txt" if try_url.endswith("/llms-full.txt") else "llms.txt"
                if try_url != url:
                    # Auto-upgrade happened
                    logger.info(f"{url} → {file_type} [{format_name}] {len(sections)} sections")
                else:
                    logger.info(f"{url} [{format_name}] {len(sections)} sections")

                return (
                    200,
                    sections,
                    hdrs.get("ETag"),
                    hdrs.get("Last-Modified"),
                    try_url,
                )

            except Exception as e:
                if try_url == urls_to_try[-1]:  # Last URL, re-raise
                    raise
                logger.debug(f"Failed to fetch {try_url}: {e}")
                continue

        # Should never reach here
        raise Exception(f"Failed to fetch from any URL: {urls_to_try}")

    async def _index_source(
        self, source_url: str, prior: SourceState | None, http_client: httpx.AsyncClient
    ) -> None:
        try:
            (
                code,
                sections,
                etag,
                last_mod,
                actual_url,
            ) = await self._fetch_and_parse_sections(
                source_url,
                prior.etag if prior else None,
                prior.last_modified if prior else None,
                http_client,
            )
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                logger.warning(f"Source not found (404): {source_url}")
                return
            logger.error(f"HTTP error fetching {source_url}: {e}")
            raise
        except Exception as e:
            logger.error(f"Error indexing {source_url}: {e}")
            raise
        if code == 304 and prior:
            self.sources[source_url] = dataclasses.replace(prior, last_indexed=time.time())
            return

        host = host_of(source_url)
        collection = self.ensure_collection()

        ids: list[str] = []
        docs: list[str] = []
        metadatas: list[dict[str, str | int | float | bool | None]] = []

        for idx, sec in enumerate(sections):
            sec_title = sec.title or "Untitled"
            sec_desc = sec.description
            sec_content = sec.content
            # Ensure ID uniqueness even when titles repeat by appending a stable ordinal suffix
            cid = f"{canonical_id(source_url, sec_title)}-{idx:03d}"
            ids.append(cid)
            # Emphasize description in embedding by repeating it
            embedding_text = f"{sec_title}\n{sec_desc}\n{sec_desc}\n{sec_content}"
            docs.append(embedding_text)
            metadatas.append(
                ChromaMetadata(
                    id=cid,
                    source=actual_url,  # Use the actual URL that was fetched
                    requested_url=source_url,  # Original URL from config
                    host=host,
                    title=sec_title,
                    description=sec_desc,  # Add description to metadata
                    content=sec_content,
                    content_hash=get_content_hash(
                        sec_content
                    ),  # Add content hash for change detection
                    section_index=idx,  # Add section index for ordering
                    indexed_at=time.time(),  # Timestamp for TTL-based cleanup
                ).model_dump()
            )

        # delete previous docs for this source (check both original and actual URLs)
        try:
            # Try to delete docs with both the original URL and actual URL
            all_ids_to_delete = []

            # Check original URL
            where_clause: ChromaWhere = {"source": source_url}
            existing = collection.get(where=where_clause, include=["ids"])  # type: ignore[list-item]
            if existing and existing.get("ids"):
                all_ids_to_delete.extend(existing["ids"])

            # Check actual URL if different
            if actual_url != source_url:
                where_actual: ChromaWhere = {"source": actual_url}
                existing_actual = collection.get(where=where_actual, include=["ids"])  # type: ignore[list-item]
                if existing_actual and existing_actual.get("ids"):
                    all_ids_to_delete.extend(existing_actual["ids"])

            if all_ids_to_delete:
                try:
                    ids_to_delete_typed: ChromaIds = all_ids_to_delete
                    collection.delete(ids=ids_to_delete_typed)
                    logger.info(f"Deleted {len(all_ids_to_delete)} old documents from {source_url}")
                except Exception as e:
                    logger.warning(f"Failed to delete old documents from {source_url}: {e}")
                    # Continue with adding new documents anyway
        except Exception as e:
            logger.debug(f"Could not check for existing documents from {source_url}: {e}")
            # This might happen on first indexing, which is fine

        if ids:
            embeddings = self.embedding_model.encode(docs)
            # Cast for ChromaDB which expects Mapping instead of dict
            metadata_mappings = cast(
                "list[Mapping[str, str | int | float | bool | None]]", metadatas
            )
            collection.add(
                ids=ids, documents=docs, embeddings=embeddings.tolist(), metadatas=metadata_mappings
            )

        self.sources[source_url] = SourceState(
            source_url=source_url,
            host=host,
            etag=etag,
            last_modified=last_mod,
            last_indexed=time.time(),
            doc_count=len(ids),
            actual_url=actual_url,
        )

        # Log indexing results with format hint
        if len(ids) > 0:
            # Infer format from sections structure
            format_hint = ""
            if sections and hasattr(sections[0], "url") and sections[0].url:
                format_hint = " (standard-llms-txt: links)"
            elif sections and sections[0].description:
                format_hint = " (yaml-frontmatter)"
            else:
                format_hint = " (standard-full)"

            # Show actual URL if different from requested
            url_info = actual_url if actual_url != source_url else source_url
            logger.info(f"Indexed {len(ids)} sections from {url_info}{format_hint}")
        else:
            logger.warning(f"No sections found in {source_url}")

    def search(self, query: str, limit: int, include_snippets: bool = True) -> list[SearchResult]:
        collection = self.ensure_collection()

        # Build list of valid source URLs (including actual URLs from redirects)
        allowed_urls = self.config.allowed_urls

        query_embedding = self.embedding_model.encode([query]).tolist()

        # Query with filter for configured URLs
        # Documents can match either by requested_url (original) or source (actual)
        # ChromaDB Where clause - the type hints don't properly support $or with $in
        where_clause: Any = {
            "$or": [
                {"requested_url": {"$in": list(allowed_urls)}},
                {"source": {"$in": list(allowed_urls)}},
            ]
        }

        res = collection.query(
            query_embeddings=query_embedding,
            n_results=min(max(limit, 1), 20),
            where=where_clause,
            include=["metadatas", "distances"],
        )
        items: list[SearchResult] = []
        metas_result = res.get("metadatas", [[]])
        dists_result = res.get("distances", [[]])
        metas = metas_result[0] if metas_result else []
        dists = dists_result[0] if dists_result else []

        # Parse query terms for snippet extraction
        query_terms = query.split() if include_snippets else []

        for meta, dist in zip(metas, dists, strict=False):
            score = max(0.0, 1.0 - float(dist))

            # Extract snippet if requested
            snippet = ""
            if include_snippets and query_terms:
                content = str(meta.get("content", ""))
                snippet = extract_snippet(content, query_terms)

            items.append(
                SearchResult(
                    id=str(meta.get("id", "")),
                    source=str(meta.get("source", "")),
                    title=str(meta.get("title", "")),
                    description=str(meta.get("description", "")),
                    score=round(score, 3),
                    auto_retrieved=False,  # Will be set by caller
                    snippet=snippet,
                )
            )
            if len(items) >= limit:
                break
        return items

    def get(
        self, ids: list[str], max_bytes: int | None, merge: bool, max_tokens: int | None = None
    ) -> dict[str, Any]:
        collection = self.ensure_collection()
        max_budget = int(max_bytes) if max_bytes is not None else self.max_get_bytes

        # Use token limit if specified, with buffer for JSON overhead
        effective_token_limit = None
        if max_tokens:
            effective_token_limit = max_tokens - TOKEN_LIMIT_BUFFER
            logger.debug(f"Token limit: {effective_token_limit} (buffer: {TOKEN_LIMIT_BUFFER})")

        results: list[DocContent] = []
        total_bytes = 0
        total_tokens = 0
        merged_content_parts: list[str] = []
        truncated_due_to_tokens = False

        for cid in ids:
            include_meta: ChromaInclude = ["metadatas"]
            res = collection.get(ids=[cid], include=include_meta)
            metas = res.get("metadatas") or []
            if not metas:
                continue
            meta = metas[0]
            title = str(meta.get("title", ""))
            content = str(meta.get("content", ""))
            source = str(meta.get("source", ""))
            host = str(meta.get("host", ""))
            header = f"# {title}\n"
            contribution = header + content

            # Check token limit if enabled (explicitly check for not None)
            if effective_token_limit is not None:
                contribution_tokens = self.token_counter.count_tokens(contribution)
                if total_tokens + contribution_tokens > effective_token_limit:
                    # Try to fit partial content
                    remaining_tokens = effective_token_limit - total_tokens
                    if remaining_tokens <= 0:
                        truncated_due_to_tokens = True
                        logger.info(f"Reached token limit ({max_tokens}), truncating response")
                        break

                    # Truncate content to fit within token limit (rough approximation)
                    # Estimate chars to keep based on token ratio
                    char_per_token_ratio = (
                        len(contribution) / contribution_tokens if contribution_tokens > 0 else 4
                    )
                    chars_to_keep = int(
                        remaining_tokens * char_per_token_ratio * 0.9
                    )  # 90% to be safe
                    if chars_to_keep > 0:
                        contribution = contribution[:chars_to_keep]
                        contribution_tokens = self.token_counter.count_tokens(contribution)
                        truncated_due_to_tokens = True
                    else:
                        break

                total_tokens += contribution_tokens

            # Also check byte limit
            contribution_bytes = contribution.encode("utf-8")
            if total_bytes + len(contribution_bytes) > max_budget:
                remaining = max_budget - total_bytes
                if remaining <= 0:
                    break
                truncated = contribution_bytes[:remaining].decode("utf-8", errors="ignore")
                contribution = truncated
                contribution_bytes = truncated.encode("utf-8")

            total_bytes += len(contribution_bytes)

            if merge:
                merged_content_parts.append(contribution)
            else:
                results.append(
                    DocContent(
                        id=cid,
                        source=source,
                        host=host,
                        title=title,
                        content=contribution,
                    )
                )

            if total_bytes >= max_budget:
                break

        response: dict[str, Any] = {
            "merged": merge,
            "total_tokens": total_tokens if effective_token_limit is not None else None,
            "truncated_due_to_tokens": truncated_due_to_tokens,
        }

        if merge:
            response["content"] = "\n\n".join(merged_content_parts)
        else:
            response["items"] = results

        return response

    async def cleanup_expired_documents(self) -> int:
        """Remove documents older than TTL from unconfigured sources.

        Returns the number of documents cleaned up.
        """
        collection = self.ensure_collection()
        allowed_urls = self.config.allowed_urls
        ttl_seconds = self.ttl_seconds
        now = time.time()

        # Get all documents to check their metadata
        try:
            include_meta: ChromaInclude = ["metadatas"]
            all_docs = collection.get(include=include_meta)
            all_metas = all_docs.get("metadatas", [])
        except Exception as e:
            logger.debug(f"No documents to clean up: {e}")
            return 0

        if not all_metas:
            return 0

        # Group documents by their requested_url
        docs_by_source: dict[str, list[tuple[str, float]]] = {}
        for meta in all_metas:
            doc_id = meta.get("id")
            requested_url = meta.get("requested_url") or meta.get("source")
            indexed_at = meta.get("indexed_at", 0)

            if doc_id and requested_url:
                # Ensure types are correct
                doc_id_str = str(doc_id)
                requested_url_str = str(requested_url)
                indexed_at_float = float(indexed_at) if indexed_at else 0.0

                if requested_url_str not in docs_by_source:
                    docs_by_source[requested_url_str] = []
                docs_by_source[requested_url_str].append((doc_id_str, indexed_at_float))

        # Find expired documents from unconfigured sources
        ids_to_delete = []
        for source_url, doc_infos in docs_by_source.items():
            if source_url not in allowed_urls:
                # Check if all docs from this source are expired
                for doc_id, indexed_at in doc_infos:
                    if (now - indexed_at) > ttl_seconds:
                        ids_to_delete.append(doc_id)

        # Delete expired documents
        if ids_to_delete:
            try:
                ids_typed: ChromaIds = ids_to_delete
                collection.delete(ids=ids_typed)
                logger.info(
                    f"Cleaned up {len(ids_to_delete)} expired documents from unconfigured sources"
                )
            except Exception as e:
                logger.warning(f"Failed to clean up expired documents: {e}")
                return 0

        return len(ids_to_delete)


# -------------------------
# Resources
# -------------------------


@mcp.resource("resource://sources")
async def get_sources() -> list[SourceInfo]:
    """Get list of all indexed documentation sources."""
    try:
        rm = _ensure_resource_manager()
        if rm.index_manager is None:
            return []

        return [
            SourceInfo(
                source_url=st.source_url,
                host=st.host,
                lastIndexed=int(st.last_indexed),
                docCount=st.doc_count,
            )
            for st in rm.index_manager.sources.values()
        ]
    except RuntimeError:
        # Server not initialized yet
        return []


# -------------------------
# Field constants (avoid B008 rule violations)
# -------------------------

_RETRIEVE_IDS_FIELD = Field(default=None)
_MAX_BYTES_FIELD = Field(default=None)
_MERGE_FIELD = Field(default=False)

# -------------------------
# Tools
# -------------------------


@mcp.tool()
async def docs_sources() -> list[SourceInfo]:
    """List indexed documentation sources."""
    try:
        rm = _ensure_resource_manager()
        if rm.index_manager is None:
            return []

        return [
            SourceInfo(
                source_url=st.source_url,
                host=st.host,
                lastIndexed=int(st.last_indexed),
                docCount=st.doc_count,
            )
            for st in rm.index_manager.sources.values()
        ]
    except RuntimeError:
        # Server not initialized yet
        return []


@mcp.tool()
async def docs_refresh(
    source: str | None = None,
    ctx: Context | None = None,
) -> RefreshResult:
    """Force refresh cached documentation."""
    rm = _ensure_resource_manager()
    if rm.index_manager is None:
        raise RuntimeError("Index manager not initialized")

    # Wait for resources to be ready
    try:
        await rm.ensure_ready(timeout=RESOURCE_INIT_TIMEOUT)
    except TimeoutError:
        raise RuntimeError("Server resources still initializing, please try again") from None

    refreshed: list[str] = []
    allowed_urls = rm.config.allowed_urls

    if source:
        if source not in allowed_urls:
            raise ValueError("Source not allowed")
        if ctx:
            await ctx.report_progress(0.5, 1.0, f"Refreshing {source}...")
        await rm.index_manager.maybe_refresh(source, force=True, http_client=rm.http_client)
        refreshed.append(source)
    else:
        total = len(allowed_urls)
        for i, url in enumerate(list(allowed_urls), 1):
            if ctx:
                await ctx.report_progress(
                    float(i - 1) / total, float(total), f"Refreshing source {i}/{total}: {url}"
                )
            await rm.index_manager.maybe_refresh(url, force=True, http_client=rm.http_client)
            refreshed.append(url)

    if ctx:
        await ctx.report_progress(1.0, 1.0, "Refresh complete")

    return RefreshResult(
        refreshed=refreshed,
        counts={
            u: rm.index_manager.sources[u].doc_count
            for u in refreshed
            if u in rm.index_manager.sources
        },
    )


@mcp.tool()
async def docs_query(
    query: str = Field(description="Search query text"),
    limit: int = 10,
    auto_retrieve: bool = True,
    auto_retrieve_threshold: float | None = None,
    auto_retrieve_limit: int | None = None,
    retrieve_ids: list[str] | None = _RETRIEVE_IDS_FIELD,
    max_bytes: int | None = _MAX_BYTES_FIELD,
    merge: bool = _MERGE_FIELD,
) -> QueryResult:
    """Search documentation with optional auto-retrieval. Combines search + get functionality."""
    rm = _ensure_resource_manager()
    if rm.index_manager is None:
        raise RuntimeError("Index manager not initialized")

    # Wait briefly for resources to be ready
    try:
        await rm.ensure_ready(timeout=RESOURCE_WAIT_TIMEOUT)
    except TimeoutError:
        # Return friendly message if resources not ready yet
        return QueryResult(
            search_results=[],
            retrieved_content={},
            merged_content="⏳ Server is initializing resources, please try again in a moment",
            auto_retrieved_count=0,
            total_results=0,
        )

    # Use defaults from config if not provided
    threshold = (
        auto_retrieve_threshold
        if auto_retrieve_threshold is not None
        else rm.config.auto_retrieve_threshold
    )
    retrieve_limit = (
        auto_retrieve_limit if auto_retrieve_limit is not None else rm.config.auto_retrieve_limit
    )
    include_snippets = rm.config.include_snippets

    # Refresh stale sources
    for url in list(rm.config.allowed_urls):
        await rm.index_manager.maybe_refresh(url, http_client=rm.http_client)

    # Perform search
    search_results = rm.index_manager.search(
        query=query, limit=limit, include_snippets=include_snippets
    )

    # Determine which IDs to retrieve
    ids_to_retrieve: list[str] = []
    auto_retrieved_count = 0

    if retrieve_ids:
        # Explicit retrieval requested
        ids_to_retrieve.extend(retrieve_ids)

    if auto_retrieve:
        # Auto-retrieve based on score threshold
        for result in search_results[:retrieve_limit]:
            if result.score >= threshold:
                if result.id not in ids_to_retrieve:
                    ids_to_retrieve.append(result.id)
                    auto_retrieved_count += 1
                # Mark as auto-retrieved
                result.auto_retrieved = True

    # Retrieve content
    retrieved_content: dict[str, DocContent] = {}
    merged_content = ""

    total_tokens = None
    truncated_due_to_tokens = False

    if ids_to_retrieve:
        # Pass max_tokens from config
        get_result = rm.index_manager.get(
            ids=ids_to_retrieve,
            max_bytes=max_bytes,
            merge=merge,
            max_tokens=rm.config.max_response_tokens,
        )

        # Extract token information
        total_tokens = get_result.get("total_tokens")
        truncated_due_to_tokens = get_result.get("truncated_due_to_tokens", False)

        if merge and get_result.get("merged"):
            merged_content = get_result["content"]
        else:
            for item in get_result.get("items", []):
                # item is already a DocContent object
                retrieved_content[item.id] = item

    # Create initial result
    result = QueryResult(
        search_results=search_results,
        retrieved_content=retrieved_content,
        merged_content=merged_content,
        auto_retrieved_count=auto_retrieved_count,
        total_results=len(search_results),
        total_tokens=total_tokens,
        truncated_due_to_tokens=truncated_due_to_tokens,
    )
    
    # Count tokens in the complete response and truncate if needed
    response_json = result.model_dump_json()
    response_tokens = rm.index_manager.token_counter.count_tokens(response_json)
    
    # If response exceeds limit, reduce search results
    if response_tokens > rm.config.max_response_tokens:
        logger.info(f"Response ({response_tokens} tokens) exceeds limit ({rm.config.max_response_tokens}), truncating search results")
        
        # Reduce search results until we're under the limit
        max_results = len(search_results)
        for i in range(max_results - 1, 0, -1):  # Start from end, keep at least 1 result
            truncated_result = QueryResult(
                search_results=search_results[:i],
                retrieved_content=retrieved_content,
                merged_content=merged_content,
                auto_retrieved_count=auto_retrieved_count,
                total_results=i,
                total_tokens=total_tokens,
                truncated_due_to_tokens=True,
            )
            
            truncated_json = truncated_result.model_dump_json()
            truncated_tokens = rm.index_manager.token_counter.count_tokens(truncated_json)
            
            if truncated_tokens <= rm.config.max_response_tokens:
                logger.info(f"Truncated to {i} search results ({truncated_tokens} tokens)")
                return truncated_result
        
        # If even 1 result is too big, return minimal response
        logger.warning("Even single result exceeds token limit, returning minimal response")
        return QueryResult(
            search_results=[],
            retrieved_content={},
            merged_content="Response too large - please use more specific query terms or filters",
            auto_retrieved_count=0,
            total_results=0,
            total_tokens=None,
            truncated_due_to_tokens=True,
        )
    
    return result


# -------------------------
# Initialization / CLI
# -------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="llms-txt-mcp", description="Lean Documentation MCP via llms.txt"
    )
    parser.add_argument("--version", "-v", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("sources", nargs="*", help="llms.txt URLs to index (positional)")
    parser.add_argument("--sources", dest="sources_flag", nargs="*", help="llms.txt URLs to index")
    parser.add_argument("--ttl", default="24h", help="Refresh cadence (e.g., 30m, 24h)")
    parser.add_argument("--timeout", type=int, default=30, help="HTTP timeout seconds")
    parser.add_argument(
        "--embed-model", default="BAAI/bge-small-en-v1.5", help="SentenceTransformers model id"
    )
    parser.add_argument(
        "--no-preindex", action="store_true", help="Disable automatic pre-indexing on launch"
    )
    parser.add_argument(
        "--no-background-preindex",
        action="store_true",
        help="Disable background preindexing (by default runs in background)",
    )
    parser.add_argument(
        "--store",
        choices=["memory", "disk"],
        default=None,
        help="Override auto-detected storage mode (auto-detects based on --store-path)",
    )
    parser.add_argument(
        "--store-path",
        default=None,
        help="Store path for disk persistence (if provided, enables disk mode)",
    )
    parser.add_argument(
        "--max-get-bytes", type=int, default=75000, help="Default byte cap for document retrieval"
    )
    parser.add_argument(
        "--auto-retrieve-threshold",
        type=float,
        default=0.1,
        help="Default score threshold for auto-retrieval (0-1, default: 0.1)",
    )
    parser.add_argument(
        "--auto-retrieve-limit",
        type=int,
        default=5,
        help="Default max number of docs to auto-retrieve (default: 5)",
    )
    parser.add_argument(
        "--no-snippets", action="store_true", help="Disable content snippets in search results"
    )
    parser.add_argument(
        "--max-response-tokens",
        type=int,
        default=DEFAULT_MAX_RESPONSE_TOKENS,
        help=f"Maximum tokens in MCP response (default: {DEFAULT_MAX_RESPONSE_TOKENS})",
    )
    return parser.parse_args()


@asynccontextmanager
async def managed_resources(cfg: Config) -> AsyncIterator[ResourceManager]:
    """FastMCP-style resource management with async initialization."""

    # Validate URLs
    for url in cfg.allowed_urls:
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            raise ValueError(f"Invalid URL: {url}")
        # Support both llms.txt and llms-full.txt
        if not (url.endswith(("/llms.txt", "/llms-full.txt"))):
            raise ValueError(f"URL must end with /llms.txt or /llms-full.txt: {url}")

    # Initialize HTTP client immediately (lightweight)
    http_client = httpx.AsyncClient(
        timeout=cfg.timeout,
        follow_redirects=True,
        headers={"User-Agent": f"llms-txt-mcp/{__version__}"},
    )

    logger.info(
        "Starting llms-txt-mcp with %d source(s): %s",
        len(cfg.allowed_urls),
        ", ".join(sorted(cfg.allowed_urls)),
    )

    # Create resource manager
    resource_manager = ResourceManager(http_client, cfg)

    # Start background initialization (non-blocking)
    init_task = asyncio.create_task(resource_manager.initialize_heavy_resources())

    try:
        yield resource_manager
    finally:
        # Cleanup resources
        logger.debug("Cleaning up resources...")

        # Cancel initialization task if still running
        if not init_task.done():
            init_task.cancel()
            with suppress(asyncio.CancelledError):
                await init_task

        # Close HTTP client
        try:
            await asyncio.wait_for(http_client.aclose(), timeout=HTTP_CLOSE_TIMEOUT)
        except TimeoutError:
            logger.warning("HTTP client close timed out")
        except Exception as e:
            logger.error(f"Error closing HTTP client: {e}")


def main() -> None:
    args = parse_args()

    # Merge positional and flag sources
    urls: list[str] = []
    if args.sources:
        urls.extend(args.sources)
    if args.sources_flag:
        urls.extend(args.sources_flag)
    if not urls:
        raise SystemExit("Provide at least one llms.txt URL via positional args or --sources")

    # Create config
    cfg = Config(
        allowed_urls=set(urls),
        ttl_seconds=parse_duration_to_seconds(args.ttl),
        timeout=args.timeout,
        embed_model_name=args.embed_model,
        store_mode=args.store if args.store else ("disk" if args.store_path else "memory"),
        store_path=args.store_path,
        max_get_bytes=args.max_get_bytes,
        auto_retrieve_threshold=args.auto_retrieve_threshold,
        auto_retrieve_limit=args.auto_retrieve_limit,
        include_snippets=not args.no_snippets,
        preindex=not args.no_preindex,
        background_preindex=not args.no_background_preindex,
        max_response_tokens=args.max_response_tokens,
    )

    async def run_server() -> None:
        """Run the server with managed resources."""
        global resource_manager
        shutdown_event = asyncio.Event()

        # Set up async signal handlers
        loop = asyncio.get_running_loop()

        def shutdown_handler() -> None:
            logger.info("Received shutdown signal, shutting down...")
            shutdown_event.set()

        loop.add_signal_handler(signal.SIGINT, shutdown_handler)
        loop.add_signal_handler(signal.SIGTERM, shutdown_handler)

        async with managed_resources(cfg) as rm:
            # Set global resource manager for tools to access
            resource_manager = rm

            logger.info("llms-txt-mcp ready. Server accepting connections...")
            logger.info("Resources initializing in background...")

            try:
                # Run server immediately - resources initialize in background
                server_task = asyncio.create_task(mcp.run_stdio_async())
                shutdown_task = asyncio.create_task(shutdown_event.wait())

                done, _ = await asyncio.wait(
                    {server_task, shutdown_task}, return_when=asyncio.FIRST_COMPLETED
                )

                if shutdown_task in done:
                    logger.info("Shutting down...")
                    server_task.cancel()
                    with suppress(TimeoutError, asyncio.CancelledError):
                        await asyncio.wait_for(server_task, timeout=HTTP_CLOSE_TIMEOUT)
            except Exception as e:
                logger.error(f"Server error: {e}")
            finally:
                # Clear global resource manager
                resource_manager = None

    # Run the async server
    try:
        asyncio.run(run_server())
    except KeyboardInterrupt:
        pass
    finally:
        logger.info("Server shutdown complete")


if __name__ == "__main__":
    main()
