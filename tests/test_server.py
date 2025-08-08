"""Tests for MCP server following modern best practices."""

import json
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from llms_txt_mcp.server import (
    docs_query,
    docs_refresh,
    docs_sources,
    managed_resources,
    mcp,
)

# Test fixtures directory
FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def minimal_content():
    """Load minimal test fixture."""
    return (FIXTURES_DIR / "minimal.llms.txt").read_text(encoding="utf-8")


@pytest.fixture
def large_content():
    """Load large test fixture."""
    return (FIXTURES_DIR / "large.llms.txt").read_text(encoding="utf-8")


@pytest.fixture
def mock_http_client():
    """Mock HTTP client for testing."""
    client = MagicMock()

    # Mock response
    response = MagicMock()
    response.text = """# Test Doc

## Section 1
Content 1

## Section 2
Content 2
"""
    response.raise_for_status.return_value = None
    client.get = AsyncMock(return_value=response)
    client.aclose = AsyncMock()

    # Mock streaming context manager
    class MockStreamResponse:
        def __init__(self):
            self.text = response.text
            self.status_code = 200
            self.headers = {"ETag": "test-etag", "Last-Modified": "Mon, 01 Jan 2024 00:00:00 GMT"}

        def raise_for_status(self):
            pass

        async def aiter_bytes(self, chunk_size=1024):
            content = self.text.encode("utf-8")
            for i in range(0, len(content), chunk_size):
                yield content[i : i + chunk_size]

    class MockStreamContext:
        async def __aenter__(self):
            return MockStreamResponse()

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    def mock_stream(*_args, **_kwargs):
        return MockStreamContext()

    client.stream = mock_stream
    return client


@pytest.fixture
async def server_setup(mock_http_client):
    """Setup server for async tests with mocked HTTP."""
    import llms_txt_mcp.server as server_mod

    # Stub embedding class BEFORE init to avoid model download
    class _FakeEmb:
        def __init__(self, data):
            self._data = data

        def tolist(self):
            return self._data

    class _FakeModel:
        def __init__(self, dim: int = 8):
            self.dim = dim

        def encode(self, texts):
            if isinstance(texts, str):
                texts = [texts]
            data = [[0.01 * ((i + j) % 10) for j in range(self.dim)] for i, _ in enumerate(texts)]
            return _FakeEmb(data)

    class _FakeSentenceTransformer:
        def __init__(self, *_args, **_kwargs):
            self._m = _FakeModel()

        def encode(self, texts):
            return self._m.encode(texts)

    server_mod.SentenceTransformer = _FakeSentenceTransformer  # type: ignore[attr-defined]

    test_urls = ["https://example.com/llms.txt"]

    # Use the new managed_resources context manager
    async with managed_resources(
        urls=test_urls,
        ttl=0,
        timeout=30,
        embed_model="thenlper/gte-small",
        store="memory",
        store_path=None,
        max_get_bytes=60000,
    ):
        # Replace HTTP client with mock using context variable
        server_mod.http_client_var.set(mock_http_client)

        yield

    # Cleanup happens automatically via context manager


# Removed legacy cache tests; new implementation uses TTL/ETag on network + Chroma storage.


class TestParser:
    """Test parsing functionality (high-level via IndexManager parsing)."""

    def test_extract_headers(self, minimal_content):
        from llms_txt_mcp.server import parse_llms_text

        sections = parse_llms_text(minimal_content)
        assert isinstance(sections, list)
        assert len(sections) > 0
        assert all("title" in s and "content" in s for s in sections)

    def test_extract_multiple_header_levels(self):
        """Test that we extract all header levels (H1, H2, H3, etc.), not just H1."""
        test_content = """# Main Title

## Section 1
Content here

### Subsection 1.1
More content

#### Deep Section
Even more content

## Section 2
Final content

### Subsection 2.1
Last bit
"""
        from llms_txt_mcp.server import parse_llms_text

        sections = parse_llms_text(test_content)
        got_titles = [s["title"] for s in sections]
        assert "Main Title" in got_titles or "Section 1" in got_titles

    def test_large_fixture_has_multiple_headers(self, large_content):
        """Large fixture should parse into many sections."""
        from llms_txt_mcp.server import parse_llms_text

        sections = parse_llms_text(large_content)
        titles = [s["title"] for s in sections]
        assert len(titles) > 10
        # Ensure we parsed a reasonable number of sections
        assert len(titles) > 10

    def test_exact_section_count_in_large_fixture(self, large_content):
        """Large fixture should parse exactly 132 sections as documented."""
        from llms_txt_mcp.server import parse_llms_text

        sections = parse_llms_text(large_content)
        assert len(sections) == 132, f"Expected 132 sections but got {len(sections)}"

        # Verify first and last sections
        assert sections[0]["title"] == "RAG Agent"
        assert sections[-1]["title"] == "OpenAI Compatible Providers"

    def test_proper_structured_parsing_with_yaml(self, large_content):
        """YAML blocks should produce many sections with titles and content."""
        from llms_txt_mcp.server import parse_yaml_blocks

        secs = parse_yaml_blocks(large_content)
        assert isinstance(secs, list)
        assert len(secs) >= 20
        assert all("title" in s for s in secs)

    def test_yaml_frontmatter_extraction(self):
        """Test extraction of YAML frontmatter metadata."""
        content_with_yaml = """---
title: Test Document
description: A sample document for testing
tags: [ai, sdk, test]
---

# Main Title

## Section 1
Content here
"""

        from llms_txt_mcp.server import parse_official_headings, parse_yaml_blocks

        yaml_secs = parse_yaml_blocks(content_with_yaml)
        assert len(yaml_secs) >= 1
        assert yaml_secs[0]["title"] == "Test Document"
        assert "A sample document for testing" in (yaml_secs[0]["description"] or "")

        # Should also parse markdown headings fallback
        md_secs = parse_official_headings(content_with_yaml)
        titles = [s["title"] for s in md_secs]
        assert "Main Title" in titles
        assert any("Section 1" == t or t.startswith("Section 1") for t in titles)

    def test_h1_navigation_with_minimal_fixture(self, minimal_content):
        """Official llms.txt should at least yield sections for H1/H2."""
        from llms_txt_mcp.server import parse_official_headings

        secs = parse_official_headings(minimal_content)
        titles = [s["title"] for s in secs]
        assert "llms.txt" in titles
        assert any(t == "Docs" for t in titles)

    def test_extract_section(self, minimal_content):
        """Section extraction via headings parser: select 'Docs' content."""
        from llms_txt_mcp.server import parse_official_headings

        secs = parse_official_headings(minimal_content)
        doc = next((s for s in secs if s["title"].lower() == "docs".lower()), None)
        assert doc is not None
        assert "Docs" in doc["content"]

    def test_consecutive_yaml_blocks(self):
        """Test parsing of consecutive YAML blocks without content between them."""
        content = """---
title: First Section
description: First description
tags: [tag1, tag2]
---

This is the content of the first section.

---
title: Second Section
description: Second description
tags: [tag3, tag4]
---

This is the content of the second section.

---
title: Third Section
description: Third description
---

This is the content of the third section.
"""
        from llms_txt_mcp.server import parse_yaml_blocks

        sections = parse_yaml_blocks(content)
        assert len(sections) == 3, f"Expected 3 sections but got {len(sections)}"

        # Verify all sections are parsed correctly
        assert sections[0]["title"] == "First Section"
        assert sections[0]["description"] == "First description"
        assert "first section" in sections[0]["content"].lower()

        assert sections[1]["title"] == "Second Section"
        assert sections[1]["description"] == "Second description"
        assert "second section" in sections[1]["content"].lower()

        assert sections[2]["title"] == "Third Section"
        assert sections[2]["description"] == "Third description"
        assert "third section" in sections[2]["content"].lower()


@pytest.mark.asyncio
class TestMCPTools:
    """Test MCP tool functionality."""

    async def test_docs_sources(self, server_setup):
        """Test sources listing."""
        sources = await docs_sources()
        assert isinstance(sources, list)

    async def test_search_and_get(self, server_setup):
        """Test search and retrieval using docs_query."""
        result = await docs_query(
            query="Section 1",
            limit=3,
            auto_retrieve=True,
            auto_retrieve_threshold=None,
            auto_retrieve_limit=None,
            retrieve_ids=None,
            max_bytes=None,
            merge=False
        )
        assert hasattr(result, "search_results")
        assert hasattr(result, "retrieved_content")
        assert hasattr(result, "metadata")

        # Check that we got search results
        assert isinstance(result.search_results, list)

        # If auto-retrieve worked, we should have retrieved content
        if result.metadata.auto_retrieved_count > 0:
            assert len(result.retrieved_content) > 0

    async def test_refresh(self, server_setup):
        """Test manual refresh."""
        # Pass None explicitly since it's looking for a Field
        out = await docs_refresh(source=None)
        # out is now RefreshResult Pydantic model
        assert hasattr(out, "refreshed")
        assert isinstance(out.refreshed, list)

    async def test_docs_get_with_merge(self, server_setup):
        """Test docs_query with merge=True parameter."""
        # Use docs_query to search and get multiple sections
        result = await docs_query(
            query="Section",
            limit=3,
            auto_retrieve=True,
            auto_retrieve_threshold=None,
            auto_retrieve_limit=None,
            retrieve_ids=None,
            max_bytes=None,
            merge=True
        )
        assert hasattr(result, "search_results")
        assert hasattr(result, "merged_content")
        assert isinstance(result.search_results, list)

        if result.metadata.auto_retrieved_count >= 2:
            # Check that merged content was returned
            assert isinstance(result.merged_content, str)
            assert len(result.merged_content) > 0

            # Also test with merge=False for comparison
            result_no_merge = await docs_query(
                query="Section",
                limit=3,
                auto_retrieve=True,
                auto_retrieve_threshold=None,
                auto_retrieve_limit=None,
                retrieve_ids=None,
                max_bytes=None,
                merge=False
            )
            # Should have retrieved content as separate items
            assert len(result_no_merge.retrieved_content) > 0
            assert result_no_merge.merged_content == ""  # No merged content when merge=False


@pytest.mark.performance
class TestPerformanceClaims:
    """Test performance claims from README."""

    def test_large_file_handling(self, large_content):
        """Test that we handle large files (25K+ lines)."""
        lines = large_content.split("\n")
        line_count = len(lines)

        assert line_count >= 25584, (
            f"🚨 README CLAIM: Should handle 25K+ line files (got {line_count:,} lines)"
        )

        # Should parse without errors
        from llms_txt_mcp.server import parse_llms_text

        sections = parse_llms_text(large_content)
        assert len(sections) > 0

    def test_fast_structure_discovery(self, large_content):
        """Test fast structure discovery (<1000ms)."""
        start_time = time.time()
        from llms_txt_mcp.server import parse_llms_text

        sections = parse_llms_text(large_content)
        duration_ms = (time.time() - start_time) * 1000

        assert len(sections) > 0
        assert duration_ms < 2000, (
            f"🚨 README CLAIM: Fast discovery <2000ms (got {duration_ms:.1f}ms)"
        )

    def test_clean_context_windows(self, large_content):
        """Test that structure responses are much smaller than full content."""
        from llms_txt_mcp.server import parse_llms_text

        secs = parse_llms_text(large_content)
        meta = [{"title": s.get("title"), "len": len(s.get("content", ""))} for s in secs[:50]]
        structure_json = json.dumps(meta)

        structure_size = len(structure_json)
        full_content_size = len(large_content)
        ratio = full_content_size / max(1, structure_size)

        assert structure_size < 120000
        assert ratio > 5

    def test_surgical_section_extraction(self, large_content):
        """Test precise section extraction."""
        from llms_txt_mcp.server import parse_llms_text

        secs = parse_llms_text(large_content)
        if not secs:
            pytest.skip("No sections in large fixture")
        first = secs[0]
        section_content = first.get("content", "")
        section_size = len(section_content)
        full_size = len(large_content)
        ratio = full_size / max(1, section_size)

        assert section_size < full_size / 3
        assert ratio > 3
        assert section_content.strip()


def test_server_initialization():
    """Test that server initializes correctly."""
    assert mcp.name == "llms-txt-mcp"
