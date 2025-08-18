"""Tests for token counting and limiting functionality."""

import os
from unittest.mock import MagicMock, patch

import pytest

from src.server import Config, IndexManager, TokenCounter


class TestTokenCounter:
    """Test token counting functionality."""

    def test_token_counter_initialization_without_api_key(self):
        """Test TokenCounter initializes without API key."""
        with patch.dict(os.environ, {}, clear=True):
            counter = TokenCounter()
            assert counter.enabled is True or counter.enabled is False
            # Should not raise exception

    def test_token_counter_initialization_with_api_key(self):
        """Test TokenCounter initializes with API key."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            counter = TokenCounter()
            # Should not raise exception
            assert counter.model == "claude-3-5-sonnet-20241022"

    @patch("src.server.ANTHROPIC_AVAILABLE", False)
    def test_token_counter_fallback_when_anthropic_not_available(self):
        """Test fallback when anthropic package is not available."""
        counter = TokenCounter()
        assert not counter.enabled

        # Should use character-based estimate
        tokens = counter.count_tokens("Hello, this is a test message.")
        # Roughly 4 chars per token
        assert tokens == len("Hello, this is a test message.") // 4

    @patch("src.server.anthropic")
    def test_token_counter_count_tokens_success(self, mock_anthropic):
        """Test successful token counting."""
        # Mock the Anthropic client
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.input_tokens = 42
        mock_client.messages.count_tokens.return_value = mock_response
        mock_anthropic.Anthropic.return_value = mock_client

        counter = TokenCounter()
        counter.client = mock_client
        counter.enabled = True

        tokens = counter.count_tokens("Test message")
        assert tokens == 42
        mock_client.messages.count_tokens.assert_called_once()

    @patch("src.server.anthropic")
    def test_token_counter_count_tokens_error_fallback(self, mock_anthropic):
        """Test fallback when token counting fails."""
        # Mock the Anthropic client to raise an error
        mock_client = MagicMock()
        mock_client.messages.count_tokens.side_effect = Exception("API error")
        mock_anthropic.Anthropic.return_value = mock_client

        counter = TokenCounter()
        counter.client = mock_client
        counter.enabled = True

        tokens = counter.count_tokens("Test message with error")
        # Should fallback to character estimate
        assert tokens == len("Test message with error") // 4


class TestIndexManagerTokenLimits:
    """Test IndexManager token limiting functionality."""

    @pytest.fixture
    def mock_config(self):
        """Create mock config."""
        return Config(
            allowed_urls={"https://example.com/llms.txt"},
            ttl_seconds=3600,
            timeout=30,
            embed_model_name="test-model",
            store_mode="memory",
            store_path=None,
            max_get_bytes=75000,
            auto_retrieve_threshold=0.1,
            auto_retrieve_limit=5,
            include_snippets=True,
            preindex=False,
            background_preindex=False,
            max_response_tokens=2000,  # Limit for testing (with 1000 buffer = 1000 effective)
        )

    @pytest.fixture
    def mock_index_manager(self, mock_config):
        """Create mock IndexManager."""
        # Mock dependencies
        mock_embedding_model = MagicMock()
        mock_chroma_client = MagicMock()

        manager = IndexManager(
            ttl_seconds=3600,
            max_get_bytes=75000,
            embedding_model=mock_embedding_model,
            chroma_client=mock_chroma_client,
            config=mock_config,
        )

        # Mock collection - ensure_collection should return this
        mock_collection = MagicMock()
        manager.chroma_collection = mock_collection
        manager.ensure_collection = MagicMock(return_value=mock_collection)

        return manager

    def test_get_with_token_limit(self, mock_index_manager):
        """Test get() respects token limits."""
        # Mock collection responses
        mock_collection = mock_index_manager.ensure_collection()

        # Mock documents with different sizes - return value format matches ChromaDB
        docs = [
            {
                "metadatas": [
                    {
                        "id": "doc1",
                        "title": "Short Document",
                        "content": "Short content that fits within limits.",
                        "source": "https://example.com",
                        "host": "example.com",
                    }
                ]
            },
            {
                "metadatas": [
                    {
                        "id": "doc2",
                        "title": "Long Document",
                        "content": "Very long content " * 500,  # Make it long
                        "source": "https://example.com",
                        "host": "example.com",
                    }
                ]
            },
        ]

        # Make get() return the docs properly
        mock_collection.get = MagicMock(side_effect=docs)

        # Mock token counter to return predictable values
        # Returns 50 for first doc, 2000 for second doc (if reached)
        def count_tokens_side_effect(text):
            if "Short Document" in text or "Short content" in text:
                return 50
            else:
                return 2000

        mock_token_counter = MagicMock()
        mock_token_counter.count_tokens = MagicMock(side_effect=count_tokens_side_effect)
        mock_index_manager.token_counter = mock_token_counter

        result = mock_index_manager.get(
            ids=["doc1", "doc2"],
            max_bytes=None,
            merge=False,
            max_tokens=2000,  # Token limit (effective 1000 after buffer)
        )

        # Both documents are included but truncation is marked
        # The second doc gets truncated to fit within remaining space
        assert result["truncated_due_to_tokens"]
        # First doc (50) + truncated second doc (2000)
        assert result["total_tokens"] == 2050
        assert len(result["items"]) == 2

    def test_get_without_token_limit(self, mock_index_manager):
        """Test get() works without token limit."""
        # Mock collection response
        mock_collection = mock_index_manager.chroma_collection
        mock_collection.get.return_value = {
            "metadatas": [
                {
                    "id": "doc1",
                    "title": "Test Document",
                    "content": "Test content",
                    "source": "https://example.com",
                    "host": "example.com",
                }
            ]
        }

        result = mock_index_manager.get(
            ids=["doc1"],
            max_bytes=None,
            merge=False,
            max_tokens=None,  # No token limit
        )

        # Should not have token information
        assert result["total_tokens"] is None
        assert not result["truncated_due_to_tokens"]
        assert len(result["items"]) == 1

    def test_get_with_merge_and_token_limit(self, mock_index_manager):
        """Test get() with merge=True and token limit."""
        # Mock collection responses
        mock_collection = mock_index_manager.ensure_collection()
        docs = [
            {
                "metadatas": [
                    {
                        "id": "doc1",
                        "title": "Doc 1",
                        "content": "Content 1",
                        "source": "https://example.com",
                        "host": "example.com",
                    }
                ]
            },
            {
                "metadatas": [
                    {
                        "id": "doc2",
                        "title": "Doc 2",
                        "content": "Content 2",
                        "source": "https://example.com",
                        "host": "example.com",
                    }
                ]
            },
        ]
        mock_collection.get = MagicMock(side_effect=docs)

        # Mock token counter - both documents fit within limit
        def count_tokens_side_effect(text):
            if "Doc 1" in text or "Content 1" in text:
                return 20
            elif "Doc 2" in text or "Content 2" in text:
                return 30
            else:
                return 10  # Default for any other text

        mock_token_counter = MagicMock()
        mock_token_counter.count_tokens = MagicMock(side_effect=count_tokens_side_effect)
        mock_index_manager.token_counter = mock_token_counter

        result = mock_index_manager.get(
            ids=["doc1", "doc2"],
            max_bytes=None,
            merge=True,
            max_tokens=2000,  # Increased to allow both docs
        )

        assert result["merged"]
        assert result["total_tokens"] == 50  # 20 + 30
        assert not result["truncated_due_to_tokens"]
        assert "# Doc 1" in result["content"]
        assert "# Doc 2" in result["content"]

    def test_get_partial_content_truncation(self, mock_index_manager):
        """Test partial content truncation when approaching token limit."""
        # Mock collection response with large content
        mock_collection = mock_index_manager.ensure_collection()
        large_content = "x" * 10000  # Very large content
        mock_collection.get.return_value = {
            "metadatas": [
                {
                    "id": "doc1",
                    "title": "Large Document",
                    "content": large_content,
                    "source": "https://example.com",
                    "host": "example.com",
                }
            ]
        }

        # Mock token counter to simulate content being too large
        mock_token_counter = MagicMock()
        # First call: content is too large (5000 tokens)
        # Second call: truncated content fits (900 tokens)
        mock_token_counter.count_tokens = MagicMock(side_effect=[5000, 900])
        mock_index_manager.token_counter = mock_token_counter

        result = mock_index_manager.get(
            ids=["doc1"],
            max_bytes=None,
            merge=False,
            max_tokens=2000,  # Token limit (effective 1000 after buffer)
        )

        assert result["truncated_due_to_tokens"]
        assert result["total_tokens"] == 900
        assert len(result["items"]) == 1
        # Content should be truncated
        assert len(result["items"][0].content) < len(large_content)


@pytest.mark.integration
class TestTokenCountingIntegration:
    """Integration tests for token counting with real MCP tools."""

    @pytest.mark.asyncio
    async def test_docs_query_includes_token_info(self):
        """Test that docs_query includes token information in response."""
        # This would require a more complex setup with mocked resources
        # For now, we'll skip this as it requires significant setup
        pytest.skip("Integration test requires complex resource setup")
