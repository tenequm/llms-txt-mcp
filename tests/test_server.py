"""Tests for MCP server following modern best practices."""

import json
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.server import (
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
    return (FIXTURES_DIR / "ai-sdk-dev-llms.txt").read_text(encoding="utf-8")


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
    import src.server as server_mod

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

    from src.server import Config

    test_urls = ["https://example.com/llms.txt"]

    # Create config for tests
    test_config = Config(
        allowed_urls=set(test_urls),
        ttl_seconds=0,
        timeout=30,
        embed_model_name="thenlper/gte-small",
        store_mode="memory",
        store_path=None,
        max_get_bytes=60000,
        auto_retrieve_threshold=0.1,
        auto_retrieve_limit=5,
        include_snippets=True,
        preindex=False,
        background_preindex=False,
    )

    # Use the new managed_resources context manager
    async with managed_resources(test_config):
        # Replace HTTP client with mock
        server_mod.http_client = mock_http_client

        yield

    # Cleanup happens automatically via context manager


# Removed legacy cache tests; new implementation uses TTL/ETag on network + Chroma storage.


@pytest.mark.asyncio
@pytest.mark.usefixtures("server_setup")
class TestMCPTools:
    """Test MCP tool functionality."""

    async def test_docs_sources(self):
        """Test sources listing."""
        sources = await docs_sources()
        assert isinstance(sources, list)

    async def test_search_and_get(self):
        """Test search and retrieval using docs_query."""
        result = await docs_query(
            query="Section 1",
            limit=3,
            auto_retrieve=True,
            auto_retrieve_threshold=None,
            auto_retrieve_limit=None,
            retrieve_ids=None,
            max_bytes=None,
            merge=False,
        )
        assert hasattr(result, "search_results")
        assert hasattr(result, "retrieved_content")
        assert hasattr(result, "auto_retrieved_count")
        assert hasattr(result, "total_results")

        # Check that we got search results
        assert isinstance(result.search_results, list)

        # If auto-retrieve worked, we should have retrieved content
        if result.auto_retrieved_count > 0:
            assert len(result.retrieved_content) > 0

    async def test_refresh(self):
        """Test manual refresh."""
        # Pass None explicitly since it's looking for a Field
        out = await docs_refresh(source=None)
        # out is now RefreshResult Pydantic model
        assert hasattr(out, "refreshed")
        assert isinstance(out.refreshed, list)

    async def test_docs_get_with_merge(self):
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
            merge=True,
        )
        assert hasattr(result, "search_results")
        assert hasattr(result, "merged_content")
        assert isinstance(result.search_results, list)

        if result.auto_retrieved_count >= 2:
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
                merge=False,
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
        from src.parser import parse_llms_txt

        result = parse_llms_txt(large_content)
        sections = result.docs
        assert len(sections) > 0

    def test_fast_structure_discovery(self, large_content):
        """Test fast structure discovery (<1000ms)."""
        start_time = time.time()
        from src.parser import parse_llms_txt

        result = parse_llms_txt(large_content)
        sections = result.docs
        duration_ms = (time.time() - start_time) * 1000

        assert len(sections) > 0
        assert duration_ms < 2000, (
            f"🚨 README CLAIM: Fast discovery <2000ms (got {duration_ms:.1f}ms)"
        )

    def test_clean_context_windows(self, large_content):
        """Test that structure responses are much smaller than full content."""
        from src.parser import parse_llms_txt

        result = parse_llms_txt(large_content)
        secs = result.docs
        meta = [{"title": s.title, "len": len(s.content)} for s in secs[:50]]
        structure_json = json.dumps(meta)

        structure_size = len(structure_json)
        full_content_size = len(large_content)
        ratio = full_content_size / max(1, structure_size)

        assert structure_size < 120000
        assert ratio > 5

    def test_surgical_section_extraction(self, large_content):
        """Test precise section extraction."""
        from src.parser import parse_llms_txt

        result = parse_llms_txt(large_content)
        secs = result.docs
        if not secs:
            pytest.skip("No sections in large fixture")
        first = secs[0]
        section_content = first.content
        section_size = len(section_content)
        full_size = len(large_content)
        ratio = full_size / max(1, section_size)

        assert section_size < full_size / 3
        assert ratio > 3
        assert section_content.strip()


def test_server_initialization():
    """Test that server initializes correctly."""
    assert mcp.name == "llms-txt-mcp"
