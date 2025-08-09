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

**ALWAYS USE UV - Never pip, python -m, or virtualenv!**

```bash
# Quick development with uv
make check              # Run all checks (format, lint, test) 
make fix                # Auto-fix issues and run checks

# Individual tasks (all use uv internally)
make format             # Format code with uv run ruff
make lint               # Lint code with uv run ruff  
make test               # Run tests with uv run pytest
make type               # Type check with uv run mypy

# Setup with uv
make install            # uv sync dependencies
make clean              # Clean cache files
make help               # Show all commands

# Direct uv commands for reference
uv sync                 # Install/sync dependencies
uv run pytest          # Run tests
uv run ruff check .     # Lint code
uv run ruff format .    # Format code  
uv run mypy src/        # Type check
uv build                # Build package
uv publish              # Publish to PyPI
```

## Remember

- If it's not in the official MCP docs, you probably don't need it
- Start with the minimum viable implementation
- Add complexity only when proven necessary
- Delete code that isn't being used
- Your code will be used in ways you can't anticipate - make it robust
- Security and error handling are not optional - fail gracefully
- Write code for the next developer (it might be you in 6 months)