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

# /// script
# dependencies = [
#     "mcp>=1.0.0",
#     "httpx>=0.27.0",
#     "sentence-transformers>=3.0.0",
#     "chromadb>=0.5.0",
#     "pyyaml>=6.0.0"
# ]
# ///

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import hashlib
import logging
import re
import signal
import sys
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

try:
    from importlib.metadata import version

    __version__ = version("llms-txt-mcp")
except ImportError:
    # Fallback for development/editable installs
    __version__ = "0.1.0-dev"

import chromadb
import httpx
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.server import Context
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

from .parsers import parse_llms_txt

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
# Data models
# -------------------------


@dataclass
class SourceState:
    source_url: str
    host: str
    etag: str | None
    last_modified: str | None
    last_indexed: float
    doc_count: int
    format_type: str | None = None  # Track detected format
    actual_url: str | None = None  # Track if auto-upgraded to llms-full.txt


# -------------------------
# Pydantic models for API responses
# -------------------------


class SourceInfo(BaseModel):
    """Information about an indexed documentation source."""

    source_url: str = Field(description="URL of the llms.txt source")
    host: str = Field(description="Host domain of the source")
    lastIndexed: int = Field(description="Unix timestamp of last indexing")
    docCount: int = Field(description="Number of documents from this source")


class SearchResult(BaseModel):
    """A single search result from semantic search."""

    id: str = Field(description="Unique document identifier")
    source: str = Field(description="Source URL of the document")
    title: str = Field(description="Document title")
    description: str = Field(default="", description="Document description")
    score: float = Field(description="Similarity score (0-1)")
    auto_retrieved: bool = Field(default=False, description="Whether content was auto-retrieved")
    snippet: str = Field(default="", description="Relevant content snippet")


class DocContent(BaseModel):
    """Retrieved document content."""

    id: str = Field(description="Unique document identifier")
    source: str = Field(description="Source URL")
    host: str = Field(description="Host domain")
    title: str = Field(description="Document title")
    content: str = Field(description="Full or truncated content")


class RefreshResult(BaseModel):
    """Result of refreshing documentation sources."""

    refreshed: list[str] = Field(description="URLs that were refreshed")
    counts: dict[str, int] = Field(description="Document counts per source")


class QueryResult(BaseModel):
    """Combined search and retrieval result with metadata."""

    search_results: list[SearchResult] = Field(description="Search results with scores")
    retrieved_content: dict[str, DocContent] = Field(
        default_factory=dict, description="Auto-retrieved document contents"
    )
    merged_content: str = Field(default="", description="Merged content if merge=true")
    # Metadata fields merged directly
    auto_retrieved_count: int = Field(default=0, description="Number of documents auto-retrieved")
    total_results: int = Field(default=0, description="Total search results found")


# -------------------------
# Lazy Loading Support
# -------------------------


class LazyEmbeddingModel:
    """Loads embedding model only when first needed."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model: SentenceTransformer | None = None

    def get(self) -> SentenceTransformer:
        if self._model is None:
            logger.info(f"Loading embedding model on first use: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode(self, texts):
        """Forward encode calls to the actual model."""
        return self.get().encode(texts)


# -------------------------
# Context variables for state management
# -------------------------


# Initialize context variables
allowed_urls_var: ContextVar[set[str]] = ContextVar("allowed_urls", default=set())
http_client_var: ContextVar[httpx.AsyncClient | None] = ContextVar("http_client", default=None)
embedding_model_var: ContextVar[LazyEmbeddingModel | SentenceTransformer | None] = ContextVar(
    "embedding_model", default=None
)
chroma_client_var: ContextVar[chromadb.Client | None] = ContextVar("chroma_client", default=None)
chroma_collection_var: ContextVar[chromadb.Collection | None] = ContextVar(
    "chroma_collection", default=None
)
ttl_seconds_var: ContextVar[int] = ContextVar("ttl_seconds", default=24 * 3600)
default_max_get_bytes_var: ContextVar[int] = ContextVar("default_max_get_bytes", default=60000)
store_mode_var: ContextVar[str] = ContextVar("store_mode", default="memory")
store_path_var: ContextVar[str | None] = ContextVar("store_path", default=None)
index_manager_var: ContextVar[IndexManager | None] = ContextVar("index_manager", default=None)
auto_retrieve_threshold_var: ContextVar[float] = ContextVar("auto_retrieve_threshold", default=0.1)
auto_retrieve_limit_var: ContextVar[int] = ContextVar("auto_retrieve_limit", default=5)
include_snippets_var: ContextVar[bool] = ContextVar("include_snippets", default=True)


# No getter functions needed - use var.get() directly


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
    def __init__(self, ttl_seconds: int, max_get_bytes: int) -> None:
        self.ttl_seconds = ttl_seconds
        self.max_get_bytes = max_get_bytes
        self.sources: dict[str, SourceState] = {}

    def ensure_collection(self) -> chromadb.Collection:
        chroma_client = chroma_client_var.get()
        chroma_collection = chroma_collection_var.get()

        if chroma_client is None:
            raise RuntimeError("Chroma client not initialized")
        if chroma_collection is None:
            chroma_collection = chroma_client.get_or_create_collection(
                name="docs",
                metadata={"purpose": "llms-txt-mcp"},
                embedding_function=None,
            )
            chroma_collection_var.set(chroma_collection)
        return chroma_collection

    async def maybe_refresh(self, source_url: str, force: bool = False) -> None:
        now = time.time()
        st = self.sources.get(source_url)
        if st and not force and (now - st.last_indexed) < self.ttl_seconds:
            return
        await self._index_source(source_url, st)

    async def _stream_lines(
        self, url: str, headers: dict[str, str]
    ) -> tuple[AsyncIterator[str], dict[str, str]]:
        http_client = http_client_var.get()
        if http_client is None:
            raise RuntimeError("HTTP client not initialized")

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
                    if buffer.endswith("\n") or buffer.endswith("\r"):
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
        self, url: str, etag: str | None, last_modified: str | None
    ) -> tuple[int, list[dict[str, Any]], str | None, str | None, str]:
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
                    try_url, headers if try_url == url else {}
                )

                # Collect and parse content
                all_lines = [line async for line in lines_iter]
                if not all_lines:
                    continue

                result = parse_llms_txt("\n".join(all_lines))
                sections = result["docs"]
                format_type = result.get("format", "unknown")

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
                    format_type,
                )

            except Exception as e:
                if try_url == urls_to_try[-1]:  # Last URL, re-raise
                    raise
                logger.debug(f"Failed to fetch {try_url}: {e}")
                continue

        # Should never reach here
        raise Exception(f"Failed to fetch from any URL: {urls_to_try}")

    async def _index_source(self, source_url: str, prior: SourceState | None) -> None:
        try:
            (
                code,
                sections,
                etag,
                last_mod,
                actual_url,
                format_type,
            ) = await self._fetch_and_parse_sections(
                source_url, prior.etag if prior else None, prior.last_modified if prior else None
            )
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                logger.warning(f"Source not found (404): {source_url}")
                return
            else:
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

        embedding_model = embedding_model_var.get()
        if embedding_model is None:
            raise RuntimeError("Embedding model not initialized")

        ids: list[str] = []
        docs: list[str] = []
        metadatas: list[dict[str, Any]] = []

        for idx, sec in enumerate(sections):
            sec_title = sec.get("title") or "Untitled"
            sec_desc = sec.get("description") or ""
            sec_content = sec.get("content") or ""
            # Ensure ID uniqueness even when titles repeat by appending a stable ordinal suffix
            cid = f"{canonical_id(source_url, sec_title)}-{idx:03d}"
            ids.append(cid)
            # Emphasize description in embedding by repeating it
            embedding_text = f"{sec_title}\n{sec_desc}\n{sec_desc}\n{sec_content}"
            docs.append(embedding_text)
            metadatas.append(
                {
                    "id": cid,
                    "source": actual_url,  # Use the actual URL that was fetched
                    "requested_url": source_url,  # Original URL from config
                    "host": host,
                    "title": sec_title,
                    "description": sec_desc,  # Add description to metadata
                    "content": sec_content,
                    "content_hash": get_content_hash(
                        sec_content
                    ),  # Add content hash for change detection
                    "section_index": idx,  # Add section index for ordering
                    "indexed_at": time.time(),  # Timestamp for TTL-based cleanup
                }
            )

        # delete previous docs for this source (check both original and actual URLs)
        try:
            # Try to delete docs with both the original URL and actual URL
            all_ids_to_delete = []

            # Check original URL
            existing = collection.get(where={"source": source_url}, include=["ids"])  # type: ignore[arg-type]
            if existing and existing.get("ids"):
                all_ids_to_delete.extend(existing["ids"])

            # Check actual URL if different
            if actual_url != source_url:
                existing_actual = collection.get(where={"source": actual_url}, include=["ids"])  # type: ignore[arg-type]
                if existing_actual and existing_actual.get("ids"):
                    all_ids_to_delete.extend(existing_actual["ids"])

            if all_ids_to_delete:
                try:
                    collection.delete(ids=all_ids_to_delete)  # type: ignore[arg-type]
                    logger.info(f"Deleted {len(all_ids_to_delete)} old documents from {source_url}")
                except Exception as e:
                    logger.warning(f"Failed to delete old documents from {source_url}: {e}")
                    # Continue with adding new documents anyway
        except Exception as e:
            logger.debug(f"Could not check for existing documents from {source_url}: {e}")
            # This might happen on first indexing, which is fine

        if ids:
            embeddings = embedding_model.encode(docs)
            collection.add(
                ids=ids, documents=docs, embeddings=embeddings.tolist(), metadatas=metadatas
            )

        self.sources[source_url] = SourceState(
            source_url=source_url,
            host=host,
            etag=etag,
            last_modified=last_mod,
            last_indexed=time.time(),
            doc_count=len(ids),
            format_type=format_type,
            actual_url=actual_url,
        )

        # Log indexing results with format hint
        if len(ids) > 0:
            # Infer format from sections structure
            format_hint = ""
            if sections and "url" in sections[0]:
                format_hint = " (standard-llms-txt: links)"
            elif sections and sections[0].get("description"):
                format_hint = " (yaml-frontmatter)"
            else:
                format_hint = " (standard-full)"

            # Show actual URL if different from requested
            url_info = actual_url if actual_url != source_url else source_url
            logger.info(f"Indexed {len(ids)} sections from {url_info}{format_hint}")
        else:
            logger.warning(f"No sections found in {source_url}")

    def search(self, query: str, limit: int, include_snippets: bool = True) -> list[dict[str, Any]]:
        collection = self.ensure_collection()
        embedding_model = embedding_model_var.get()
        if embedding_model is None:
            raise RuntimeError("Embedding model not initialized")

        # Build list of valid source URLs (including actual URLs from redirects)
        allowed_urls = allowed_urls_var.get()

        query_embedding = embedding_model.encode([query]).tolist()

        # Query with filter for configured URLs
        # Documents can match either by requested_url (original) or source (actual)
        res = collection.query(
            query_embeddings=query_embedding,
            n_results=min(max(limit, 1), 20),
            where={
                "$or": [
                    {"requested_url": {"$in": list(allowed_urls)}},
                    {"source": {"$in": list(allowed_urls)}},
                ]
            },
            include=["metadatas", "distances"],
        )  # type: ignore[arg-type]
        items: list[dict[str, Any]] = []
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]

        # Parse query terms for snippet extraction
        query_terms = query.split() if include_snippets else []

        for meta, dist in zip(metas, dists):
            score = max(0.0, 1.0 - float(dist))

            # Extract snippet if requested
            snippet = ""
            if include_snippets and query_terms:
                content = str(meta.get("content", ""))
                snippet = extract_snippet(content, query_terms)

            items.append(
                {
                    "id": meta.get("id"),
                    "source": meta.get("source"),
                    "title": meta.get("title"),
                    "description": meta.get("description", ""),
                    "score": round(score, 3),
                    "auto_retrieved": False,  # Will be set by caller
                    "snippet": snippet,
                }
            )
            if len(items) >= limit:
                break
        return items

    def get(self, ids: list[str], max_bytes: int | None, merge: bool) -> dict[str, Any]:
        collection = self.ensure_collection()
        max_budget = int(max_bytes) if max_bytes is not None else self.max_get_bytes
        results: list[dict[str, Any]] = []
        total = 0
        merged_content_parts: list[str] = []

        for cid in ids:
            res = collection.get(ids=[cid], include=["metadatas"])  # type: ignore[arg-type]
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
            contribution_bytes = contribution.encode("utf-8")
            if total + len(contribution_bytes) > max_budget:
                remaining = max_budget - total
                if remaining <= 0:
                    break
                truncated = contribution_bytes[:remaining].decode("utf-8", errors="ignore")
                contribution = truncated
                contribution_bytes = truncated.encode("utf-8")
            total += len(contribution_bytes)
            if merge:
                merged_content_parts.append(contribution)
            else:
                results.append(
                    {
                        "id": cid,
                        "source": source,
                        "host": host,
                        "title": title,
                        "content": contribution,
                    }
                )
            if total >= max_budget:
                break

        if merge:
            return {"merged": True, "content": "\n\n".join(merged_content_parts)}
        return {"merged": False, "items": results}

    async def cleanup_expired_documents(self) -> int:
        """Remove documents older than TTL from unconfigured sources.

        Returns the number of documents cleaned up.
        """
        collection = self.ensure_collection()
        allowed_urls = allowed_urls_var.get()
        ttl_seconds = self.ttl_seconds
        now = time.time()

        # Get all documents to check their metadata
        try:
            all_docs = collection.get(include=["metadatas"])  # type: ignore[arg-type]
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
                if requested_url not in docs_by_source:
                    docs_by_source[requested_url] = []
                docs_by_source[requested_url].append((doc_id, indexed_at))

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
                collection.delete(ids=ids_to_delete)  # type: ignore[arg-type]
                logger.info(
                    f"Cleaned up {len(ids_to_delete)} expired documents from unconfigured sources"
                )
            except Exception as e:
                logger.warning(f"Failed to clean up expired documents: {e}")
                return 0

        return len(ids_to_delete)


# -------------------------
# Tools
# -------------------------


@mcp.tool()
async def docs_sources() -> list[SourceInfo]:
    """List indexed documentation sources."""
    index = index_manager_var.get()
    if index is None:
        return []

    return [
        SourceInfo(
            source_url=st.source_url,
            host=st.host,
            lastIndexed=int(st.last_indexed),
            docCount=st.doc_count,
        )
        for st in index.sources.values()
    ]


@mcp.tool()
async def docs_refresh(
    source: str | None = Field(
        default=None, description="Specific source URL to refresh, or None for all"
    ),
    ctx: Context | None = None,
) -> RefreshResult:
    """Force refresh cached documentation."""
    index = index_manager_var.get()
    if index is None:
        raise RuntimeError("Server not initialized")

    refreshed: list[str] = []
    allowed_urls = allowed_urls_var.get()

    if source:
        if source not in allowed_urls:
            raise ValueError("Source not allowed")
        if ctx:
            await ctx.report_progress(f"Refreshing {source}...")
        await index.maybe_refresh(source, force=True)
        refreshed.append(source)
    else:
        total = len(allowed_urls)
        for i, url in enumerate(list(allowed_urls), 1):
            if ctx:
                await ctx.report_progress(f"Refreshing source {i}/{total}: {url}")
            await index.maybe_refresh(url, force=True)
            refreshed.append(url)

    if ctx:
        await ctx.report_progress("Refresh complete")

    return RefreshResult(
        refreshed=refreshed,
        counts={u: index.sources[u].doc_count for u in refreshed if u in index.sources},
    )


@mcp.tool()
async def docs_query(
    query: str = Field(description="Search query text"),
    limit: int = Field(default=10, description="Maximum number of search results"),
    auto_retrieve: bool = Field(default=True, description="Auto-retrieve top relevant results"),
    auto_retrieve_threshold: float | None = Field(
        default=None, description="Min score for auto-retrieve (default: 0.1)"
    ),
    auto_retrieve_limit: int | None = Field(
        default=None, description="Max docs to auto-retrieve (default: 5)"
    ),
    retrieve_ids: list[str] | None = Field(
        default=None, description="Specific document IDs to retrieve"
    ),
    max_bytes: int | None = Field(default=None, description="Byte limit per retrieved document"),
    merge: bool = Field(
        default=False, description="Merge all retrieved content into single response"
    ),
) -> QueryResult:
    """Search documentation with optional auto-retrieval. Combines search + get functionality."""
    index = index_manager_var.get()
    if index is None:
        raise RuntimeError("Server not initialized")

    # Use defaults from config if not provided
    threshold = (
        auto_retrieve_threshold
        if auto_retrieve_threshold is not None
        else auto_retrieve_threshold_var.get()
    )
    retrieve_limit = (
        auto_retrieve_limit if auto_retrieve_limit is not None else auto_retrieve_limit_var.get()
    )
    include_snippets = include_snippets_var.get()

    # Refresh stale sources
    for url in list(allowed_urls_var.get()):
        await index.maybe_refresh(url)

    # Perform search
    search_results = index.search(query=query, limit=limit, include_snippets=include_snippets)

    # Determine which IDs to retrieve
    ids_to_retrieve: list[str] = []
    auto_retrieved_count = 0

    if retrieve_ids:
        # Explicit retrieval requested
        ids_to_retrieve.extend(retrieve_ids)

    if auto_retrieve:
        # Auto-retrieve based on score threshold
        for result in search_results[:retrieve_limit]:
            if result["score"] >= threshold:
                if result["id"] not in ids_to_retrieve:
                    ids_to_retrieve.append(result["id"])
                    auto_retrieved_count += 1
                # Mark as auto-retrieved
                result["auto_retrieved"] = True

    # Retrieve content
    retrieved_content: dict[str, DocContent] = {}
    merged_content = ""

    if ids_to_retrieve:
        result = index.get(ids=ids_to_retrieve, max_bytes=max_bytes, merge=merge)
        if merge and result.get("merged"):
            merged_content = result["content"]
        else:
            for item in result.get("items", []):
                retrieved_content[item["id"]] = DocContent(**item)

    return QueryResult(
        search_results=[SearchResult(**result) for result in search_results],
        retrieved_content=retrieved_content,
        merged_content=merged_content,
        auto_retrieved_count=auto_retrieved_count,
        total_results=len(search_results),
    )


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
        "--no-smart-preindex",
        action="store_true",
        help="Disable smart preindexing (by default only stale sources are indexed)",
    )
    parser.add_argument(
        "--parallel-preindex",
        type=int,
        default=3,
        help="Number of parallel indexing tasks (default: 3)",
    )
    parser.add_argument(
        "--no-background-preindex",
        action="store_true",
        help="Disable background preindexing (by default runs in background)",
    )
    parser.add_argument(
        "--no-lazy-embed",
        action="store_true",
        help="Load embedding model immediately instead of on first use",
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
    return parser.parse_args()


@asynccontextmanager
async def managed_resources(
    urls: list[str],
    ttl: int,
    timeout: int,
    embed_model: str,
    store: str,
    store_path: str | None,
    max_get_bytes: int,
    lazy_embed: bool = False,
    auto_retrieve_threshold: float = 0.1,
    auto_retrieve_limit: int = 5,
    include_snippets: bool = True,
):
    """Async context manager for managing all server resources."""
    # Validate and store URLs
    allowed: set[str] = set()
    for url in urls:
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            raise ValueError(f"Invalid URL: {url}")
        # Support both llms.txt and llms-full.txt
        if not (url.endswith("/llms.txt") or url.endswith("/llms-full.txt")):
            raise ValueError(f"URL must end with /llms.txt or /llms-full.txt: {url}")
        allowed.add(url)

    # Set context variables
    allowed_urls_var.set(allowed)
    ttl_seconds_var.set(ttl)
    default_max_get_bytes_var.set(max_get_bytes)
    store_mode_var.set(store)
    store_path_var.set(store_path)
    auto_retrieve_threshold_var.set(auto_retrieve_threshold)
    auto_retrieve_limit_var.set(auto_retrieve_limit)
    include_snippets_var.set(include_snippets)

    # Initialize HTTP client
    http_client = httpx.AsyncClient(
        timeout=timeout,
        follow_redirects=True,
        headers={"User-Agent": f"llms-txt-mcp/{__version__}"},
    )
    http_client_var.set(http_client)

    # Initialize embedding model
    logger.info(
        "Starting llms-txt-mcp with %d source(s): %s",
        len(allowed),
        ", ".join(sorted(allowed)),
    )

    if lazy_embed:
        logger.info(f"Embedding model ({embed_model}) will load on first use")
        embedding_model = LazyEmbeddingModel(embed_model)
    else:
        logger.info(f"Loading embedding model: {embed_model}")
        embedding_model = SentenceTransformer(embed_model)
    embedding_model_var.set(embedding_model)

    # Initialize Chroma
    if store == "disk":
        # store_path is guaranteed to exist when store="disk" due to auto-detection logic
        if ChromaSettings is not None:
            chroma_client = chromadb.PersistentClient(
                path=store_path, settings=ChromaSettings(anonymized_telemetry=False)
            )
        else:
            chroma_client = chromadb.PersistentClient(path=store_path)
        logger.info(f"ChromaDB PersistentClient at {store_path}")
    else:
        if ChromaSettings is not None:
            chroma_client = chromadb.Client(settings=ChromaSettings(anonymized_telemetry=False))
        else:
            chroma_client = chromadb.Client()
        logger.info("ChromaDB ephemeral client initialized (telemetry disabled)")

    chroma_client_var.set(chroma_client)

    # Initialize index manager
    index_manager = IndexManager(ttl_seconds=ttl, max_get_bytes=max_get_bytes)
    index_manager_var.set(index_manager)

    # Clean up expired documents from unconfigured sources
    if store == "disk":
        # Only cleanup when using persistent storage
        try:
            cleaned_up = await index_manager.cleanup_expired_documents()
            if cleaned_up > 0:
                logger.info(f"Startup cleanup: removed {cleaned_up} expired documents")
        except Exception as e:
            logger.debug(f"Startup cleanup skipped: {e}")

    try:
        yield
    finally:
        # Cleanup resources
        logger.debug("Cleaning up resources...")
        try:
            # Add timeout to prevent hanging
            await asyncio.wait_for(http_client.aclose(), timeout=2.0)
        except TimeoutError:
            logger.warning("HTTP client close timed out")
        except Exception as e:
            logger.error(f"Error closing HTTP client: {e}")

        # Clear context variables
        allowed_urls_var.set(set())
        http_client_var.set(None)
        embedding_model_var.set(None)
        chroma_client_var.set(None)
        chroma_collection_var.set(None)
        index_manager_var.set(None)


def _display_indexing_summary(index: IndexManager) -> None:
    """Display a summary table of indexed sources."""
    if not index.sources:
        return

    logger.info("=" * 60)
    logger.info("Indexing Summary:")
    logger.info("=" * 60)

    for source_url, state in index.sources.items():
        # Format the display based on what we know
        format_name = "unknown"
        if state.format_type:
            format_name = state.format_type.replace("-llms-txt", "").replace("-full", "")

        # Check if auto-upgrade happened
        display_url = source_url
        if state.actual_url and state.actual_url != source_url:
            file_type = (
                "llms-full.txt"
                if state.actual_url.endswith("/llms-full.txt")
                else state.actual_url.split("/")[-1]
            )
            display_url = f"{source_url} → {file_type}"

        # Display the summary line
        logger.info(f"{display_url} | {format_name} | {state.doc_count} sections")

    logger.info("=" * 60)


async def preindex_sources(ctx: Context | None = None, parallel: int = 1) -> None:
    """Pre-index all configured sources."""
    index = index_manager_var.get()
    if index is None:
        raise RuntimeError("Index manager not initialized")

    allowed_urls = allowed_urls_var.get()
    total = len(allowed_urls)
    start = time.time()
    logger.info("Preindexing %d source(s)...", total)

    if parallel > 1:
        # Parallel indexing
        semaphore = asyncio.Semaphore(parallel)

        async def index_one(url: str, idx: int):
            async with semaphore:
                logger.info(f"Fetching {url}...")
                await index.maybe_refresh(url, force=True)
                if ctx:
                    await ctx.report_progress(f"Indexed {idx}/{total}: {url}")

        tasks = [index_one(url, i) for i, url in enumerate(allowed_urls, 1)]
        await asyncio.gather(*tasks)
    else:
        # Sequential indexing
        for i, url in enumerate(list(allowed_urls), 1):
            if ctx:
                await ctx.report_progress(f"Pre-indexing source {i}/{total}: {url}")
            logger.info(f"Fetching {url}...")
            await index.maybe_refresh(url, force=True)

    # Calculate total documents indexed
    total_docs = sum(st.doc_count for st in index.sources.values())
    indexed_count = len([st for st in index.sources.values() if st.doc_count > 0])

    if ctx:
        await ctx.report_progress("Pre-indexing complete")

    logger.info(
        "Indexing complete: %d sections from %d/%d sources (%.1fs)",
        total_docs,
        indexed_count,
        total,
        time.time() - start,
    )

    # Display summary table
    _display_indexing_summary(index)


async def smart_preindex_sources(ctx: Context | None = None, parallel: int = 1) -> None:
    """Only preindex sources that are stale or missing."""
    index = index_manager_var.get()
    if index is None:
        raise RuntimeError("Index manager not initialized")

    allowed_urls = allowed_urls_var.get()
    now = time.time()
    to_index = []

    # Check which sources need indexing
    for url in allowed_urls:
        st = index.sources.get(url)
        if not st or (now - st.last_indexed) >= index.ttl_seconds:
            to_index.append(url)

    if not to_index:
        logger.info("All sources are fresh, skipping preindex")
        return

    logger.info(f"Smart preindex: {len(to_index)}/{len(allowed_urls)} sources need updating")
    start = time.time()

    if parallel > 1:
        # Parallel indexing
        semaphore = asyncio.Semaphore(parallel)

        async def index_one(url: str, idx: int):
            async with semaphore:
                logger.info(f"Fetching {url}...")
                await index.maybe_refresh(url, force=True)
                if ctx:
                    await ctx.report_progress(f"Updated {idx}/{len(to_index)}: {url}")

        tasks = [index_one(url, i) for i, url in enumerate(to_index, 1)]
        await asyncio.gather(*tasks)
    else:
        # Sequential indexing
        for i, url in enumerate(to_index, 1):
            if ctx:
                await ctx.report_progress(f"Updating {i}/{len(to_index)}: {url}")
            logger.info(f"Fetching {url}...")
            await index.maybe_refresh(url, force=True)

    # Calculate total documents indexed
    total_docs = sum(st.doc_count for st in index.sources.values())

    if ctx:
        await ctx.report_progress("Smart preindex complete")

    logger.info(
        "Indexing complete: %d sections from %d/%d sources updated (%.1fs)",
        total_docs,
        len(to_index),
        len(allowed_urls),
        time.time() - start,
    )

    # Display summary table
    _display_indexing_summary(index)


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
    ttl_seconds = parse_duration_to_seconds(args.ttl)

    async def run_server():
        """Run the server with managed resources."""
        # Invert the no-* flags to get the actual settings
        lazy_embed = not args.no_lazy_embed
        preindex = not args.no_preindex  # Preindex by default
        smart_preindex = not args.no_smart_preindex
        background_preindex = not args.no_background_preindex

        # Auto-detect storage mode based on store_path, unless explicitly overridden
        if args.store is not None:
            store = args.store
        else:
            store = "disk" if args.store_path else "memory"

        # Track background tasks and shutdown state
        background_tasks = set()
        shutdown_event = asyncio.Event()

        async def shutdown_handler(sig):
            """Handle shutdown signals gracefully."""
            logger.info(f"Received {sig.name}, initiating graceful shutdown...")
            shutdown_event.set()

        def signal_handler_sync(signum, frame):
            """Synchronous signal handler fallback."""
            sig = signal.Signals(signum)
            logger.info(f"Received {sig.name}, initiating graceful shutdown...")
            shutdown_event.set()

        async with managed_resources(
            urls=urls,
            ttl=ttl_seconds,
            timeout=args.timeout,
            embed_model=args.embed_model,
            store=store,
            store_path=args.store_path,
            max_get_bytes=args.max_get_bytes,
            lazy_embed=lazy_embed,
            auto_retrieve_threshold=args.auto_retrieve_threshold,
            auto_retrieve_limit=args.auto_retrieve_limit,
            include_snippets=not args.no_snippets,
        ):
            # Set up signal handlers on the event loop
            loop = asyncio.get_running_loop()
            for sig in (signal.SIGTERM, signal.SIGINT):
                try:
                    loop.add_signal_handler(
                        sig, lambda s=sig: asyncio.create_task(shutdown_handler(s))
                    )
                    logger.debug(f"Registered handler for {sig.name}")
                except (NotImplementedError, ValueError) as e:
                    logger.warning(f"Could not register signal handler for {sig.name}: {e}")
                    # Fall back to basic signal handling
                    signal.signal(sig, signal_handler_sync)

            # Log that server is ready
            logger.info(
                "llms-txt-mcp ready. Waiting for MCP client on stdio. Press Ctrl+C to exit."
            )

            # Handle preindexing based on flags
            preindex_task = None
            if background_preindex and preindex:
                # Start preindexing in background
                if smart_preindex:
                    logger.info("Starting automatic indexing in background...")
                    preindex_task = asyncio.create_task(
                        smart_preindex_sources(parallel=args.parallel_preindex)
                    )
                else:
                    logger.info("Starting full indexing in background...")
                    preindex_task = asyncio.create_task(
                        preindex_sources(parallel=args.parallel_preindex)
                    )
                background_tasks.add(preindex_task)
            elif preindex:
                # Preindex synchronously
                if smart_preindex:
                    await smart_preindex_sources(parallel=args.parallel_preindex)
                else:
                    await preindex_sources(parallel=args.parallel_preindex)

            try:
                # Create server task
                logger.debug("Starting MCP server with stdio transport...")
                server_task = asyncio.create_task(mcp.run_stdio_async())

                # Create shutdown wait task
                shutdown_wait_task = asyncio.create_task(shutdown_event.wait())

                # Wait for either server completion or shutdown signal
                done, pending = await asyncio.wait(
                    {server_task, shutdown_wait_task}, return_when=asyncio.FIRST_COMPLETED
                )

                # Check which task completed
                if shutdown_wait_task in done:
                    logger.info("Shutdown signal received, stopping MCP server...")
                    server_task.cancel()
                    try:
                        await asyncio.wait_for(server_task, timeout=2.0)
                    except (TimeoutError, asyncio.CancelledError):
                        pass
                elif server_task in done:
                    logger.info("MCP server stopped normally")

            except Exception as e:
                logger.error(f"Server error: {e}")

            finally:
                # Remove signal handlers
                for sig in (signal.SIGTERM, signal.SIGINT):
                    try:
                        loop.remove_signal_handler(sig)
                    except (ValueError, OSError):
                        pass

                # Cancel all background tasks
                for task in background_tasks:
                    if not task.done():
                        task.cancel()

                # Wait for background tasks to complete
                if background_tasks:
                    await asyncio.gather(*background_tasks, return_exceptions=True)

    # Run the async server
    try:
        asyncio.run(run_server())
    except KeyboardInterrupt:
        # This shouldn't happen with proper signal handlers, but just in case
        pass
    finally:
        logger.info("Server shutdown complete")


if __name__ == "__main__":
    main()
