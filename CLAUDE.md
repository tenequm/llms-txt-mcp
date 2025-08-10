# llms-txt-mcp

MCP server for fetching and parsing llms.txt files.

## Core Principles

- **uv FIRST** - This project uses `uv` for ALL Python operations. Never use `pip`, `python -m`, or virtualenv directly.
- **Follow official MCP SDK patterns** - Always check latest docs at modelcontextprotocol.io
- **KISS** - Keep it simple, stupid. Simplest solution that works.
- **DRY** - Don't repeat yourself. Reuse, don't duplicate.
- **Less is more** - The less code, the better. Every line must justify its existence.
- **No bloat** - No unnecessary dependencies, configs, or documentation.
- **ALWAYS use 2025 as the current year** when searching for recent information

## Implementation

Use FastMCP for everything. Don't reinvent wheels.

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("llms-txt-mcp")

@mcp.tool()
def fetch_llms_txt(url: str) -> str:
    """Fetch llms.txt from URL."""
    # Implementation here
```

## Commands

Use Make targets (they call uv under the hood):

```bash
make install      # Install deps (uv sync --all-extras)
make check        # Format check, lint, type-check, tests
make fix          # Auto-format and fix lint
make format       # Format code (ruff)
make lint         # Lint (ruff)
make type         # Type check (mypy)
make test         # Run tests (pytest)
make test-fast    # Run tests skipping performance ones
make clean        # Remove caches
make help         # List commands
```

## Remember

- If it's not in the official MCP docs, you probably don't need it
- Start with the minimum viable implementation
- Add complexity only when proven necessary
- Delete code that isn't being used
- Your code will be used in ways you can't anticipate - make it robust
- Security and error handling are not optional - fail gracefully
- Write code for the next developer (it might be you in 6 months)