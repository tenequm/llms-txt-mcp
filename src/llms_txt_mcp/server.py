"""MCP server for llms.txt processing."""

import logging

from mcp.server.fastmcp import FastMCP

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mcp = FastMCP("llms-txt-mcp")


if __name__ == "__main__":
    mcp.run()