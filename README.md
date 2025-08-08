# docs-mcp

Fast, predictable, minimal‑context docs for Claude Code via llms.txt. URL‑only config. Search‑first. Human‑readable IDs (canonical URLs). Freshness you control.

## Why docs-mcp

- **AI SDK llms.txt support**: AI SDK’s `llms.txt` uses repeated YAML frontmatter blocks; many tools either miss sections or dump huge context. `docs-mcp` parses both AI SDK’s format and the official llms.txt cleanly. See the source format in the AI SDK docs: [ai-sdk.dev/llms.txt](https://ai-sdk.dev/llms.txt).
- **Clean context**: Search first (tiny top‑k), then fetch only requested sections with a byte cap.
- **Freshness you control**: Per‑source TTL (default 24h) + ETag/Last‑Modified revalidation. Avoid mixing stale vs. fresh versions (e.g., AI SDK v4 vs. v5 after migration). Overview: [AI SDK Introduction](https://ai-sdk.dev/docs/introduction).
- **Deterministic sources**: Strict allowlist — only the URLs you pass are indexed.

## Quickstart

Requires Python 3.13+ and [uv](https://github.com/astral-sh/uv).

```bash
git clone https://github.com/yourusername/docs-mcp.git
cd docs-mcp
uv sync --all-extras
```

Claude Desktop/Code config example:

```json
{
  "mcpServers": {
    "docs-mcp": {
      "command": "uvx",
      "args": [
        "docs-mcp",
        "https://ai-sdk.dev/llms.txt"
      ]
    }
  }
}
```

## CLI

- `URL...` (positional) or `--sources URL...`: one or more llms.txt URLs (URLs only, no names)
- `--ttl 24h`: refresh cadence (supports s/m/h, e.g., `30m`)
- `--timeout 30`: HTTP timeout (seconds)
- `--embed-model thenlper/gte-small`: default fast CPU model (384‑d)
- `--preindex`: pre‑index on launch
- `--store memory|disk`: index store (default memory)
- `--store-path /path/to/store`: required if `--store=disk`
- `--max-get-bytes 60000`: default byte cap for multi‑id get

Example:

```bash
uv run docs-mcp https://ai-sdk.dev/llms.txt --ttl 24h
```

## Tools

- `docs_sources()` → list sources with `host`, `lastIndexed`, `docCount`
- `docs_search(q, hosts?: string[], k=5)` → tiny results `{ id, source, title, snippet, score }`
- `docs_get(ids: string[], max_bytes?: 60000, merge?: false, depth?: 0)` → fetch 1–N sections by canonical URL; server enforces a byte cap
- `docs_refresh(source?: string)` → reindex now (async)

IDs are canonical URLs (+ anchors), e.g.:
- `https://ai-sdk.dev/llms.txt#rag-agent-guide`

Filter by host with `hosts: ["ai-sdk.dev"]` in `docs_search` when needed.

## Behavior & guarantees

- Search‑first UX; no “list 100+ sections” dumps
- Human‑readable IDs; easy copy/paste; no hidden hashes
- Byte‑capped multi‑id `docs_get` to protect context windows
- TTL + ETag/Last‑Modified revalidation for consistent freshness
- Unified cross‑source search by default; optional host filters

## Alternatives

- **mcpdoc**: Open‑source MCP for exposing llms.txt with a `fetch_docs` tool. Useful for transparent retrieval, but not optimized for AI SDK’s YAML‑block `llms.txt` or minimal context.
  - [github.com/langchain-ai/mcpdoc](https://github.com/langchain-ai/mcpdoc)
- **Context7**: Broad hosted knowledge MCP with many sources. Convenient, but mixed sources/freshness can yield inconsistent answers (e.g., AI SDK v4 vs. v5 after migration).
  - [github.com/upstash/context7](https://github.com/upstash/context7)

## Development

```bash
uv run pytest
uv run ruff format .
uv run mypy src/
```

## License

MIT
