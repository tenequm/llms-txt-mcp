# llms-txt-mcp

MCP server that enables Claude to fetch and parse llms.txt files from websites. The llms.txt format provides AI-friendly documentation in a standardized markdown structure.

## Installation

Requires Python 3.13+ and [uv](https://github.com/astral-sh/uv).

```bash
git clone https://github.com/yourusername/llms-txt-mcp.git
cd llms-txt-mcp
uv sync --all-extras
```

## Claude Desktop Configuration

Add to your Claude Desktop config:

```json
{
  "mcpServers": {
    "llms-txt": {
      "command": "uv",
      "args": ["--directory", "/path/to/llms-txt-mcp", "run", "llms-txt-mcp"]
    }
  }
}
```

## Development

```bash
uv run pytest           # Run tests
uv run ruff format .    # Format code
uv run mypy src/        # Type check
```

## License

MIT