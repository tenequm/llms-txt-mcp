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
import re
import signal
import sys
import time
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from typing import TYPE_CHECKING
from urllib.parse import urlparse

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from types import FrameType

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
from mcp.server.fastmcp.server import Context  # noqa: TC002
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

from .parser import ParsedDoc, parse_llms_txt

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
# Type definitions
# -------------------------


class ChromaMetadata(BaseModel):
    """Metadata stored in Chroma collection."""

    id: str = Field(description="Unique document identifier")
    source: str = Field(description="Source URL of the document")
    host: str = Field(description="Host domain of the source")
    title: str = Field(description="Document title")
    description: str = Field(default="", description="Document description")
    content: str = Field(description="Document content")
    requested_url: str = Field(default="", description="Original requested URL")
    content_hash: str = Field(default="", description="Content hash for change detection")
    section_index: int = Field(default=0, description="Section index for ordering")
    indexed_at: float = Field(default=0, description="Timestamp when indexed")


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
    """Combined search and retrieval result."""

    search_results: list[SearchResult] = Field(description="Search results with scores")
    retrieved_content: dict[str, DocContent] = Field(
        default_factory=dict, description="Auto-retrieved document contents"
    )
    merged_content: str = Field(default="", description="Merged content if merge=true")
    auto_retrieved_count: int = Field(default=0, description="Number of documents auto-retrieved")
    total_results: int = Field(default=0, description="Total search results found")


# -------------------------
# Global state management
# -------------------------

# Global state variables - initialized during startup
config: Config | None = None
http_client: httpx.AsyncClient | None = None
embedding_model: SentenceTransformer | None = None
chroma_client: chromadb.Client | None = None
chroma_collection: chromadb.Collection | None = None
index_manager: IndexManager | None = None


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
        global chroma_client, chroma_collection

        if chroma_client is None:
            raise RuntimeError("Chroma client not initialized")
        if chroma_collection is None:
            chroma_collection = chroma_client.get_or_create_collection(
                name="docs",
                metadata={"purpose": "llms-txt-mcp"},
                embedding_function=None,
            )
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
        global http_client
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
        self, url: str, etag: str | None, last_modified: str | None
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
                    try_url, headers if try_url == url else {}
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

    async def _index_source(self, source_url: str, prior: SourceState | None) -> None:
        try:
            (
                code,
                sections,
                etag,
                last_mod,
                actual_url,
            ) = await self._fetch_and_parse_sections(
                source_url, prior.etag if prior else None, prior.last_modified if prior else None
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

        global embedding_model
        if embedding_model is None:
            raise RuntimeError("Embedding model not initialized")

        ids: list[str] = []
        docs: list[str] = []
        metadatas: list[ChromaMetadata] = []

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
                ).model_dump()  # type: ignore[arg-type]
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
        global embedding_model, config
        if embedding_model is None:
            raise RuntimeError("Embedding model not initialized")
        if config is None:
            raise RuntimeError("Config not initialized")

        # Build list of valid source URLs (including actual URLs from redirects)
        allowed_urls = config.allowed_urls

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
        items: list[SearchResult] = []
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]

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
                    id=meta.get("id", ""),
                    source=meta.get("source", ""),
                    title=meta.get("title", ""),
                    description=meta.get("description", ""),
                    score=round(score, 3),
                    auto_retrieved=False,  # Will be set by caller
                    snippet=snippet,
                )
            )
            if len(items) >= limit:
                break
        return items

    def get(
        self, ids: list[str], max_bytes: int | None, merge: bool
    ) -> dict[str, str | list[DocContent]] | list[DocContent]:
        collection = self.ensure_collection()
        max_budget = int(max_bytes) if max_bytes is not None else self.max_get_bytes
        results: list[DocContent] = []
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
        global config
        if config is None:
            raise RuntimeError("Config not initialized")
        allowed_urls = config.allowed_urls
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
# Resources
# -------------------------


@mcp.resource("resource://sources")
async def get_sources() -> list[SourceInfo]:
    """Get list of all indexed documentation sources."""
    global index_manager
    if index_manager is None:
        return []

    return [
        SourceInfo(
            source_url=st.source_url,
            host=st.host,
            lastIndexed=int(st.last_indexed),
            docCount=st.doc_count,
        )
        for st in index_manager.sources.values()
    ]


# -------------------------
# Field constants (avoid B008 rule violations)
# -------------------------

_RETRIEVE_IDS_FIELD = Field(default=None, description="Specific document IDs to retrieve")
_MAX_BYTES_FIELD = Field(default=None, description="Byte limit per retrieved document")
_MERGE_FIELD = Field(default=False, description="Merge all retrieved content into single response")

# -------------------------
# Tools
# -------------------------


@mcp.tool()
async def docs_sources() -> list[SourceInfo]:
    """List indexed documentation sources."""
    global index_manager
    if index_manager is None:
        return []

    return [
        SourceInfo(
            source_url=st.source_url,
            host=st.host,
            lastIndexed=int(st.last_indexed),
            docCount=st.doc_count,
        )
        for st in index_manager.sources.values()
    ]


@mcp.tool()
async def docs_refresh(
    source: str | None = Field(
        default=None, description="Specific source URL to refresh, or None for all"
    ),
    ctx: Context | None = None,
) -> RefreshResult:
    """Force refresh cached documentation."""
    global index_manager, config
    if index_manager is None or config is None:
        raise RuntimeError("Server not initialized")

    refreshed: list[str] = []
    allowed_urls = config.allowed_urls

    if source:
        if source not in allowed_urls:
            raise ValueError("Source not allowed")
        if ctx:
            await ctx.report_progress(f"Refreshing {source}...")
        await index_manager.maybe_refresh(source, force=True)
        refreshed.append(source)
    else:
        total = len(allowed_urls)
        for i, url in enumerate(list(allowed_urls), 1):
            if ctx:
                await ctx.report_progress(f"Refreshing source {i}/{total}: {url}")
            await index_manager.maybe_refresh(url, force=True)
            refreshed.append(url)

    if ctx:
        await ctx.report_progress("Refresh complete")

    return RefreshResult(
        refreshed=refreshed,
        counts={
            u: index_manager.sources[u].doc_count for u in refreshed if u in index_manager.sources
        },
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
    retrieve_ids: list[str] | None = _RETRIEVE_IDS_FIELD,
    max_bytes: int | None = _MAX_BYTES_FIELD,
    merge: bool = _MERGE_FIELD,
) -> QueryResult:
    """Search documentation with optional auto-retrieval. Combines search + get functionality."""
    global index_manager, config
    if index_manager is None or config is None:
        raise RuntimeError("Server not initialized")

    # Use defaults from config if not provided
    threshold = (
        auto_retrieve_threshold
        if auto_retrieve_threshold is not None
        else config.auto_retrieve_threshold
    )
    retrieve_limit = (
        auto_retrieve_limit if auto_retrieve_limit is not None else config.auto_retrieve_limit
    )
    include_snippets = config.include_snippets

    # Refresh stale sources
    for url in list(config.allowed_urls):
        await index_manager.maybe_refresh(url)

    # Perform search
    search_results = index_manager.search(
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

    if ids_to_retrieve:
        result = index_manager.get(ids=ids_to_retrieve, max_bytes=max_bytes, merge=merge)
        if merge and result.get("merged"):
            merged_content = result["content"]
        else:
            for item in result.get("items", []):
                retrieved_content[item["id"]] = DocContent(
                    id=item["id"],
                    source=item["source"],
                    host=item["host"],
                    title=item["title"],
                    content=item["content"],
                )

    return QueryResult(
        search_results=search_results,
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
    return parser.parse_args()


@asynccontextmanager
async def managed_resources(cfg: Config) -> AsyncIterator[None]:
    """Async context manager for managing all server resources."""
    global config, http_client, embedding_model, chroma_client, chroma_collection, index_manager

    # Validate URLs
    for url in cfg.allowed_urls:
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            raise ValueError(f"Invalid URL: {url}")
        # Support both llms.txt and llms-full.txt
        if not (url.endswith(("/llms.txt", "/llms-full.txt"))):
            raise ValueError(f"URL must end with /llms.txt or /llms-full.txt: {url}")

    # Set global config
    config = cfg

    # Initialize HTTP client
    http_client = httpx.AsyncClient(
        timeout=cfg.timeout,
        follow_redirects=True,
        headers={"User-Agent": f"llms-txt-mcp/{__version__}"},
    )

    # Initialize embedding model
    logger.info(
        "Starting llms-txt-mcp with %d source(s): %s",
        len(cfg.allowed_urls),
        ", ".join(sorted(cfg.allowed_urls)),
    )

    logger.info(f"Loading embedding model: {cfg.embed_model_name}")
    embedding_model = SentenceTransformer(cfg.embed_model_name)

    # Initialize Chroma
    if cfg.store_mode == "disk":
        # store_path is guaranteed to exist when store="disk" due to auto-detection logic
        if ChromaSettings is not None:
            chroma_client = chromadb.PersistentClient(
                path=cfg.store_path, settings=ChromaSettings(anonymized_telemetry=False)
            )
        else:
            chroma_client = chromadb.PersistentClient(path=cfg.store_path)
        logger.info(f"ChromaDB PersistentClient at {cfg.store_path}")
    else:
        if ChromaSettings is not None:
            chroma_client = chromadb.Client(settings=ChromaSettings(anonymized_telemetry=False))
        else:
            chroma_client = chromadb.Client()
        logger.info("ChromaDB ephemeral client initialized (telemetry disabled)")

    # Initialize index manager
    index_manager = IndexManager(ttl_seconds=cfg.ttl_seconds, max_get_bytes=cfg.max_get_bytes)

    # Clean up expired documents from unconfigured sources
    if cfg.store_mode == "disk":
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

        # Clear global state
        config = None
        http_client = None
        embedding_model = None
        chroma_client = None
        chroma_collection = None
        index_manager = None


def _display_indexing_summary(index: IndexManager) -> None:
    """Display a summary table of indexed sources."""
    if not index.sources:
        return

    logger.info("=" * 60)
    logger.info("Indexing Summary:")
    logger.info("=" * 60)

    for source_url, state in index.sources.items():
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
        logger.info(f"{display_url} | {state.doc_count} sections")

    logger.info("=" * 60)


async def preindex_sources() -> None:
    """Pre-index all configured sources."""
    global index_manager, config
    if index_manager is None or config is None:
        raise RuntimeError("Server not initialized")

    total = len(config.allowed_urls)
    start = time.time()
    logger.info("Preindexing %d source(s)...", total)

    # Simple sequential indexing
    for i, url in enumerate(config.allowed_urls, 1):
        logger.info(f"Fetching {url} ({i}/{total})...")
        await index_manager.maybe_refresh(url, force=True)

    # Calculate total documents indexed
    total_docs = sum(st.doc_count for st in index_manager.sources.values())
    indexed_count = len([st for st in index_manager.sources.values() if st.doc_count > 0])

    logger.info(
        "Indexing complete: %d sections from %d/%d sources (%.1fs)",
        total_docs,
        indexed_count,
        total,
        time.time() - start,
    )

    # Display summary table
    _display_indexing_summary(index_manager)


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
    )

    async def run_server() -> None:
        """Run the server with managed resources."""
        shutdown_event = asyncio.Event()

        def signal_handler(signum: int, _frame: FrameType | None) -> None:
            """Simple signal handler."""
            logger.info(f"Received signal {signum}, shutting down...")
            shutdown_event.set()

        # Set up signal handlers
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        async with managed_resources(cfg):
            logger.info(
                "llms-txt-mcp ready. Waiting for MCP client on stdio. Press Ctrl+C to exit."
            )

            # Handle preindexing
            preindex_task = None
            if cfg.preindex:
                if cfg.background_preindex:
                    logger.info("Starting indexing in background...")
                    preindex_task = asyncio.create_task(preindex_sources())
                else:
                    await preindex_sources()

            try:
                # Run server
                server_task = asyncio.create_task(mcp.run_stdio_async())
                shutdown_task = asyncio.create_task(shutdown_event.wait())

                done, _ = await asyncio.wait(
                    {server_task, shutdown_task}, return_when=asyncio.FIRST_COMPLETED
                )

                if shutdown_task in done:
                    logger.info("Shutting down...")
                    server_task.cancel()
                    with suppress(TimeoutError, asyncio.CancelledError):
                        await asyncio.wait_for(server_task, timeout=2.0)
            except Exception as e:
                logger.error(f"Server error: {e}")
            finally:
                # Cancel preindex task if still running
                if preindex_task and not preindex_task.done():
                    preindex_task.cancel()
                    with suppress(asyncio.CancelledError):
                        await preindex_task

    # Run the async server
    try:
        asyncio.run(run_server())
    except KeyboardInterrupt:
        pass
    finally:
        logger.info("Server shutdown complete")


if __name__ == "__main__":
    main()
