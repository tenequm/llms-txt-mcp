"""llms-txt-mcp: Lean Documentation MCP via llms.txt.

Implements four tools as per plan:
- docs_sources
- docs_search
- docs_get
- docs_refresh

Parses both AI SDK YAML-frontmatter llms.txt and official llms.txt headings.
Embeds with thenlper/gte-small into a unified Chroma collection with host metadata.
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
import logging
import re
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

import chromadb
import httpx
import yaml
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.server import Context
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

try:
    # Chroma telemetry settings (0.5+)
    from chromadb.config import Settings as ChromaSettings  # type: ignore
except Exception:  # pragma: no cover - fallback if import path changes
    ChromaSettings = None  # type: ignore

logging.basicConfig(level=logging.INFO)
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


class DocContent(BaseModel):
    """Retrieved document content."""

    id: str = Field(description="Unique document identifier")
    source: str = Field(description="Source URL")
    host: str = Field(description="Host domain")
    title: str = Field(description="Document title")
    content: str = Field(description="Full or truncated content")


class MergedContent(BaseModel):
    """Merged document content response."""

    merged: bool = Field(default=True, description="Whether content was merged")
    content: str = Field(description="Combined content from multiple documents")


class SeparateContent(BaseModel):
    """Separate document content response."""

    merged: bool = Field(default=False, description="Whether content was merged")
    items: list[DocContent] = Field(description="Individual document contents")


class RefreshResult(BaseModel):
    """Result of refreshing documentation sources."""

    refreshed: list[str] = Field(description="URLs that were refreshed")
    counts: dict[str, int] = Field(description="Document counts per source")


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


# Getter functions for context variables
def get_allowed_urls() -> set[str]:
    return allowed_urls_var.get()


def get_http_client() -> httpx.AsyncClient | None:
    return http_client_var.get()


def get_embedding_model() -> LazyEmbeddingModel | SentenceTransformer | None:
    return embedding_model_var.get()


def get_chroma_client() -> chromadb.Client | None:
    return chroma_client_var.get()


def get_chroma_collection() -> chromadb.Collection | None:
    return chroma_collection_var.get()


def get_ttl_seconds() -> int:
    return ttl_seconds_var.get()


def get_default_max_get_bytes() -> int:
    return default_max_get_bytes_var.get()


def get_index_manager() -> IndexManager | None:
    return index_manager_var.get()


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
    import hashlib
    
    sample = content[:max_bytes].encode('utf-8')
    return hashlib.md5(sample).hexdigest()[:12]  # 12 chars is enough for our use case


def host_of(url: str) -> str:
    return urlparse(url).netloc


# -------------------------
# Parsing
# -------------------------


def parse_yaml_blocks(content: str) -> list[dict[str, Any]]:
    """Parse AI SDK style repeated YAML-frontmatter blocks.

    Returns a list of dicts: {title, description, tags, content}
    """
    sections: list[dict[str, Any]] = []
    lines = content.splitlines()
    i = 0
    n = len(lines)
    while i < n:
        if lines[i].strip() == "---":
            i += 1
            yaml_lines: list[str] = []
            while i < n and lines[i].strip() != "---":
                yaml_lines.append(lines[i])
                i += 1
            if i < n and lines[i].strip() == "---":
                i += 1  # consume closing ---
            else:
                continue
            try:
                meta = yaml.safe_load("\n".join(yaml_lines)) or {}
            except yaml.YAMLError:
                meta = {}
            if not isinstance(meta, dict) or "title" not in meta:
                continue
            title = str(meta.get("title", "Untitled"))
            description = str(meta.get("description", ""))
            tags = meta.get("tags") or []

            content_lines: list[str] = []
            # Collect content until we hit another YAML frontmatter block or EOF
            while i < n:
                # Check if we're at the start of a new YAML block
                if lines[i].strip() == "---":
                    # Look ahead to see if this is a real YAML frontmatter
                    j = i + 1
                    found_title = False
                    # Look for title within next 20 lines (reasonable YAML header size)
                    while j < n and j < i + 20 and lines[j].strip() != "---":
                        if lines[j].startswith("title:"):
                            found_title = True
                            break
                        j += 1
                    # If we found a title and a closing ---, this is a new section
                    if found_title and j < n and j < i + 20:
                        # Check if there's a closing --- after the title
                        while j < n and j < i + 20:
                            if lines[j].strip() == "---":
                                # This is definitely a new YAML block, stop collecting content
                                break
                            j += 1
                        else:
                            # No closing ---, treat as content
                            content_lines.append(lines[i])
                            i += 1
                            continue
                        # Found a real YAML block, stop here
                        break
                    else:
                        # Not a YAML block, just content with ---
                        content_lines.append(lines[i])
                        i += 1
                else:
                    content_lines.append(lines[i])
                    i += 1

            sections.append(
                {
                    "title": title,
                    "description": description,
                    "tags": tags if isinstance(tags, list) else [str(tags)],
                    "content": "\n".join(content_lines).strip(),
                }
            )
        else:
            i += 1
    return sections


_H1 = re.compile(r"^#\s+(.+)$")
_H2 = re.compile(r"^##\s+(.+)$")


def parse_official_headings(content: str) -> list[dict[str, Any]]:
    """Parse official llms.txt headings: prefer H2 sections; H1 fallback."""
    lines = content.splitlines()
    sections: list[dict[str, Any]] = []
    current_title: str | None = None
    current_desc: str = ""
    current_lines: list[str] = []
    saw_h2 = False

    for line in lines:
        m2 = _H2.match(line)
        m1 = None if m2 else _H1.match(line)
        if m2:
            saw_h2 = True
            if current_title is not None:
                sections.append(
                    {
                        "title": current_title,
                        "description": current_desc,
                        "tags": [],
                        "content": "\n".join(current_lines).strip(),
                    }
                )
            current_title = m2.group(1).strip()
            current_desc = ""
            current_lines = []
        elif m1 and not saw_h2:
            if current_title is not None:
                sections.append(
                    {
                        "title": current_title,
                        "description": current_desc,
                        "tags": [],
                        "content": "\n".join(current_lines).strip(),
                    }
                )
            current_title = m1.group(1).strip()
            current_desc = ""
            current_lines = []
        else:
            current_lines.append(line)

    if current_title is not None:
        sections.append(
            {
                "title": current_title,
                "description": current_desc,
                "tags": [],
                "content": "\n".join(current_lines).strip(),
            }
        )

    return sections


def parse_llms_text(content: str) -> list[dict[str, Any]]:
    sections = parse_yaml_blocks(content)
    if sections:
        return sections
    return parse_official_headings(content)


# -------------------------
# Index Manager
# -------------------------


class IndexManager:
    def __init__(self, ttl_seconds: int, max_get_bytes: int) -> None:
        self.ttl_seconds = ttl_seconds
        self.max_get_bytes = max_get_bytes
        self.sources: dict[str, SourceState] = {}

    def ensure_collection(self) -> chromadb.Collection:
        chroma_client = get_chroma_client()
        chroma_collection = get_chroma_collection()

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
        http_client = get_http_client()
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
    ) -> tuple[int, list[dict[str, Any]], str | None, str | None]:
        headers: dict[str, str] = {}
        if etag:
            headers["If-None-Match"] = etag
        if last_modified:
            headers["If-Modified-Since"] = last_modified

        lines_iter, hdrs = await self._stream_lines(url, headers)

        # Collect all lines from the stream
        all_lines: list[str] = []
        async for line in lines_iter:
            all_lines.append(line)

        # Parse the complete content using the already-fixed parse_llms_text
        full_content = "\n".join(all_lines)
        sections = parse_llms_text(full_content)

        return 200, sections, hdrs.get("ETag"), hdrs.get("Last-Modified")

    async def _index_source(self, source_url: str, prior: SourceState | None) -> None:
        code, sections, etag, last_mod = await self._fetch_and_parse_sections(
            source_url, prior.etag if prior else None, prior.last_modified if prior else None
        )
        if code == 304 and prior:
            self.sources[source_url] = dataclasses.replace(prior, last_indexed=time.time())
            return

        host = host_of(source_url)
        collection = self.ensure_collection()

        embedding_model = get_embedding_model()
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
            embedding_text = f"{sec_title}\n{sec_desc}\n{sec_desc}\n{sec_content[:500]}"
            docs.append(embedding_text)
            metadatas.append(
                {
                    "id": cid,
                    "source": source_url,
                    "host": host,
                    "title": sec_title,
                    "description": sec_desc,  # Add description to metadata
                    "content": sec_content,
                    "content_hash": get_content_hash(sec_content),  # Add content hash for change detection
                    "section_index": idx,  # Add section index for ordering
                }
            )

        # delete previous docs for this source (if supported)
        try:
            existing = collection.get(where={"source": source_url}, include=["ids"])  # type: ignore[arg-type]
            if existing and existing.get("ids"):
                existing_ids = existing["ids"]
                try:
                    collection.delete(ids=existing_ids)  # type: ignore[arg-type]
                    logger.info(f"Deleted {len(existing_ids)} old documents from {source_url}")
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
        )

    def search(self, query: str, hosts: list[str] | None, limit: int) -> list[dict[str, Any]]:
        collection = self.ensure_collection()
        embedding_model = get_embedding_model()
        if embedding_model is None:
            raise RuntimeError("Embedding model not initialized")
        query_embedding = embedding_model.encode([query]).tolist()
        res = collection.query(
            query_embeddings=query_embedding,
            n_results=min(max(limit, 1), 20),
            include=["metadatas", "distances"],
        )  # type: ignore[arg-type]
        items: list[dict[str, Any]] = []
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]
        for meta, dist in zip(metas, dists):
            if hosts and meta.get("host") not in hosts:
                continue
            score = max(0.0, 1.0 - float(dist))
            items.append(
                {
                    "id": meta.get("id"),
                    "source": meta.get("source"),
                    "title": meta.get("title"),
                    "description": meta.get("description", ""),  # Include description in results
                    "score": round(score, 3),
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


# -------------------------
# Tools
# -------------------------


@mcp.tool()
async def docs_sources() -> list[SourceInfo]:
    """List indexed documentation sources."""
    index = get_index_manager()
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
async def docs_search(
    query: str = Field(description="Search query text"),
    hosts: list[str] | None = Field(default=None, description="Filter results by host domains"),
    limit: int = Field(default=10, description="Maximum number of results to return"),
) -> list[SearchResult]:
    """Search docs by semantic similarity. Returns id, title, snippet, score."""
    index = get_index_manager()
    if index is None:
        raise RuntimeError("Server not initialized")

    for url in list(get_allowed_urls()):
        await index.maybe_refresh(url)

    results = index.search(query=query, hosts=hosts, limit=limit)
    return [SearchResult(**item) for item in results]


@mcp.tool()
async def docs_get(
    ids: list[str] = Field(description="Document IDs to retrieve"),
    max_bytes: int | None = Field(default=None, description="Maximum bytes to return per document"),
    merge: bool = Field(
        default=False, description="Whether to merge documents into a single response"
    ),
) -> MergedContent | SeparateContent:
    """Fetch full content by IDs from search. Merge=true combines sections."""
    index = get_index_manager()
    if index is None:
        raise RuntimeError("Server not initialized")

    implicated_sources = {cid.split("#", 1)[0] for cid in ids if "#" in cid}
    allowed_urls = get_allowed_urls()
    for src in implicated_sources:
        if src in allowed_urls:
            await index.maybe_refresh(src)

    result = index.get(ids=ids, max_bytes=max_bytes, merge=merge)
    if result.get("merged"):
        return MergedContent(content=result["content"])
    else:
        return SeparateContent(items=[DocContent(**item) for item in result["items"]])


@mcp.tool()
async def docs_refresh(
    source: str | None = Field(
        default=None, description="Specific source URL to refresh, or None for all"
    ),
    ctx: Context | None = None,
) -> RefreshResult:
    """Force refresh cached documentation."""
    index = get_index_manager()
    if index is None:
        raise RuntimeError("Server not initialized")

    refreshed: list[str] = []
    allowed_urls = get_allowed_urls()

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


# -------------------------
# Initialization / CLI
# -------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="llms-txt-mcp", description="Lean Documentation MCP via llms.txt"
    )
    parser.add_argument("sources", nargs="*", help="llms.txt URLs to index (positional)")
    parser.add_argument("--sources", dest="sources_flag", nargs="*", help="llms.txt URLs to index")
    parser.add_argument("--ttl", default="24h", help="Refresh cadence (e.g., 30m, 24h)")
    parser.add_argument("--timeout", type=int, default=30, help="HTTP timeout seconds")
    parser.add_argument(
        "--embed-model", default="all-MiniLM-L6-v2", help="SentenceTransformers model id"
    )
    parser.add_argument("--preindex", action="store_true", help="Pre-index on launch")
    parser.add_argument(
        "--no-smart-preindex", 
        action="store_true",
        help="Disable smart preindexing (by default only stale sources are indexed)"
    )
    parser.add_argument(
        "--parallel-preindex",
        type=int,
        default=3,
        help="Number of parallel indexing tasks (default: 3)"
    )
    parser.add_argument(
        "--no-background-preindex",
        action="store_true",
        help="Disable background preindexing (by default runs in background)"
    )
    parser.add_argument(
        "--no-lazy-embed",
        action="store_true",
        help="Load embedding model immediately instead of on first use"
    )
    parser.add_argument(
        "--store", choices=["memory", "disk"], default="memory", help="Index store mode"
    )
    parser.add_argument("--store-path", default=None, help="Store path (required for --store=disk)")
    parser.add_argument(
        "--max-get-bytes", type=int, default=80000, help="Default byte cap for docs_get"
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
):
    """Async context manager for managing all server resources."""
    # Validate and store URLs
    allowed: set[str] = set()
    for url in urls:
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            raise ValueError(f"Invalid URL: {url}")
        if not url.endswith("/llms.txt"):
            raise ValueError(f"URL must end with /llms.txt: {url}")
        allowed.add(url)

    # Set context variables
    allowed_urls_var.set(allowed)
    ttl_seconds_var.set(ttl)
    default_max_get_bytes_var.set(max_get_bytes)
    store_mode_var.set(store)
    store_path_var.set(store_path)

    # Initialize HTTP client
    http_client = httpx.AsyncClient(
        timeout=timeout, follow_redirects=True, headers={"User-Agent": "llms-txt-mcp/0.1.0"}
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
        if not store_path:
            raise ValueError("--store-path is required when --store=disk")
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

    try:
        yield
    finally:
        # Cleanup resources
        logger.info("Cleaning up resources...")
        try:
            await http_client.aclose()
        except Exception as e:
            logger.error(f"Error closing HTTP client: {e}")

        # Clear context variables
        allowed_urls_var.set(set())
        http_client_var.set(None)
        embedding_model_var.set(None)
        chroma_client_var.set(None)
        chroma_collection_var.set(None)
        index_manager_var.set(None)


async def preindex_sources(ctx: Context | None = None, parallel: int = 1) -> None:
    """Pre-index all configured sources."""
    index = get_index_manager()
    if index is None:
        raise RuntimeError("Index manager not initialized")

    allowed_urls = get_allowed_urls()
    total = len(allowed_urls)
    start = time.time()
    logger.info("Preindexing %d source(s)...", total)

    if parallel > 1:
        # Parallel indexing
        semaphore = asyncio.Semaphore(parallel)
        
        async def index_one(url: str, idx: int):
            async with semaphore:
                logger.info(f"Pre-indexing {url}")
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
            logger.info(f"Pre-indexing {url}")
            await index.maybe_refresh(url, force=True)

    if ctx:
        await ctx.report_progress("Pre-indexing complete")
    logger.info("Preindex complete in %.2fs", time.time() - start)


async def smart_preindex_sources(ctx: Context | None = None, parallel: int = 1) -> None:
    """Only preindex sources that are stale or missing."""
    index = get_index_manager()
    if index is None:
        raise RuntimeError("Index manager not initialized")
    
    allowed_urls = get_allowed_urls()
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
                logger.info(f"Indexing stale source: {url}")
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
            logger.info(f"Indexing stale source: {url}")
            await index.maybe_refresh(url, force=True)
    
    if ctx:
        await ctx.report_progress("Smart preindex complete")
    logger.info("Smart preindex complete in %.2fs", time.time() - start)


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
        smart_preindex = not args.no_smart_preindex
        background_preindex = not args.no_background_preindex
        
        async with managed_resources(
            urls=urls,
            ttl=ttl_seconds,
            timeout=args.timeout,
            embed_model=args.embed_model,
            store=args.store,
            store_path=args.store_path,
            max_get_bytes=args.max_get_bytes,
            lazy_embed=lazy_embed,
        ):
            # Handle preindexing based on flags
            preindex_task = None
            
            if background_preindex and args.preindex:
                # Start preindexing in background
                if smart_preindex:
                    logger.info("Starting smart preindex in background...")
                    preindex_task = asyncio.create_task(
                        smart_preindex_sources(parallel=args.parallel_preindex)
                    )
                else:
                    logger.info("Starting preindex in background...")
                    preindex_task = asyncio.create_task(
                        preindex_sources(parallel=args.parallel_preindex)
                    )
            elif args.preindex:
                # Preindex synchronously
                if smart_preindex:
                    await smart_preindex_sources(parallel=args.parallel_preindex)
                else:
                    await preindex_sources(parallel=args.parallel_preindex)

            try:
                logger.info(
                    "llms-txt-mcp ready. Waiting for MCP client on stdio. Press Ctrl+C to exit."
                )
                # Run the MCP server in a thread to avoid blocking the event loop
                await asyncio.to_thread(mcp.run)
            except KeyboardInterrupt:
                logger.info("Shutting down (Ctrl+C)")
            finally:
                # Cancel background task if still running
                if preindex_task and not preindex_task.done():
                    preindex_task.cancel()
                    try:
                        await preindex_task
                    except asyncio.CancelledError:
                        pass

    # Run the async server
    asyncio.run(run_server())


if __name__ == "__main__":
    main()
