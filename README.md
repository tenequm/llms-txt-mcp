## llms-txt-mcp

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/) [![MCP SDK 1.12+](https://img.shields.io/badge/MCP%20SDK-1.12+-purple.svg)](https://github.com/modelcontextprotocol/python-sdk) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Fast, surgical access to big docs in Claude Code via llms.txt. Search first, fetch only what matters.

### Why this exists
- Hitting token limits and timeouts on huge `llms.txt` files hurts flow and drowns context.
- This MCP keeps responses tiny and relevant. No dumps, no noise — just the parts you asked for.

### Quick start (Claude Desktop)
Add to `~/Library/Application Support/Claude/claude_desktop_config.json` or `.mcp.json` in your project:
```json
{
  "mcpServers": {
    "llms-txt-mcp": {
      "command": "uvx",
      "args": [
        "llms-txt-mcp",
        "https://ai-sdk.dev/llms.txt",
        "https://nextjs.org/docs/llms.txt",
        "https://orm.drizzle.team/llms.txt"
      ]
    }
  }
}
```
Now Claude Code|Desktop can instantly search and retrieve exactly what it needs from those docs.

### How it works
URL → Parse YAML/Markdown → Embed → Search → Get Section
- Parses multiple llms.txt formats (YAML frontmatter + Markdown)
- Embeds sections and searches semantically
- Retrieves only the top matches with a byte cap (default: 75KB)

### Features
- Instant startup with lazy loading and background indexing
- Search-first; no full-document dumps
- Byte-capped responses to protect context windows
- Human-readable IDs (e.g. `https://ai-sdk.dev/llms.txt#rag-agent`)

### Source resolution and crawling behavior
- Always checks for `llms-full.txt` first, even when `llms.txt` is configured. If present, it uses `llms-full.txt` for richer structure.
- For a plain `llms.txt` that only lists links, it indexes those links in the collection but does not crawl or scrape the pages behind them. Link-following/scraping may be added later.

### Talk to it in Claude Code|Desktop
- "Search Next.js docs for middleware routing. Give only the most relevant sections and keep it under 60 KB."
- "From Drizzle ORM docs, show how to define relations. Retrieve the exact section content."
- "List which sources are indexed right now."
- "Refresh the Drizzle docs so I get the latest version, then search for migrations."
- "Get the section for app router dynamic routes from Next.js using its canonical ID."

### Configuration (optional)
- **--store-path PATH** (default: none) Absolute path to persist embeddings. If set, disk persistence is enabled automatically. Prefer absolute paths (e.g., `/Users/you/.llms-cache`).
- **--ttl DURATION** (default: `24h`) Refresh cadence for sources. Supports `30m`, `24h`, `7d`.
- **--timeout SECONDS** (default: `30`) HTTP timeout.
- **--embed-model MODEL** (default: `BAAI/bge-small-en-v1.5`) SentenceTransformers model id.
- **--max-get-bytes N** (default: `75000`) Byte cap for retrieved content.
- **--auto-retrieve-threshold FLOAT** (default: `0.1`) Score threshold (0–1) to auto-retrieve matches.
- **--auto-retrieve-limit N** (default: `5`) Max docs to auto-retrieve per query.
- **--no-preindex** (default: off) Disable automatic pre-indexing on launch.
- **--no-background-preindex** (default: off) If preindexing is on, wait for it to finish before serving.
- **--no-snippets** (default: off) Disable content snippets in search results.
- **--sources ... / positional sources** One or more `llms.txt` or `llms-full.txt` URLs.

- **--store {memory|disk}** (default: auto) Not usually needed. Auto-selected based on `--store-path`. Use only to explicitly override behavior.

### Development
```bash
make install  # install deps
make test     # run tests
make check    # format check, lint, type-check, tests
make fix      # auto-format and fix lint
```

Built on [FastMCP](https://github.com/modelcontextprotocol/python-sdk) and the [Model Context Protocol](https://modelcontextprotocol.io). MIT license — see `LICENSE`.