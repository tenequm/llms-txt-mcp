"""Tests for MCP server."""

from llms_txt_mcp.server import mcp


def test_server_initialization() -> None:
    """Test that server initializes correctly."""
    assert mcp.name == "llms-txt-mcp"