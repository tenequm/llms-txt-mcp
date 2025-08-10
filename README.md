# llms-txt-mcp

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-26%20passing-brightgreen.svg)](#development)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![MCP SDK 1.12+](https://img.shields.io/badge/MCP%20SDK-1.12+-purple.svg)](https://github.com/modelcontextprotocol/python-sdk)

Fast documentation access for Claude Code via llms.txt parsing.

## The Problem

Ever seen this error?
```
Error: MCP tool "fetch_docs" response (251,431 tokens) exceeds maximum allowed tokens (25,000)
```

You're not alone. This is mcpdoc failing on AI SDK documentation.

**mcpdoc fails at scale:**
- 🐌 **5+ second** structure discovery
- 💣 **1,500 tokens** wasted just to list sections  
- ❌ **Timeouts** on files like AI SDK's 30K+ line llms.txt
- 🗑️ **Context pollution** - your conversation drowns in documentation dumps

AI SDK's documentation ([ai-sdk.dev/llms.txt](https://ai-sdk.dev/llms.txt)) breaks mcpdoc due to size.

## The Problem in Action

Here's what happens when you try to get AI SDK documentation for building a chatbot:

### mcpdoc: Token Limit Exceeded
```
> use mcpdoc to get ai-sdk documentation on how to build chatbot app

⏺ mcpdoc - fetch_docs(url: "https://ai-sdk.dev/llms.txt")
  ⎿ Error: MCP tool "fetch_docs" response (251,431 tokens) exceeds maximum 
    allowed tokens (25,000). Please use pagination, filtering, or limit 
    parameters to reduce the response size.
```
**Result:** 251,431 tokens attempted → Token limit exceeded

### Context7: Drowning in Noise
```
⏺ Context7 - get-library-docs(topic: "chatbot building guide", tokens: 15000)
  ⎿ CODE SNIPPETS
    ========================
    … +2380 lines (ctrl+r to expand)
```
**Result:** 15,000 tokens of context pollution

### llms-txt-mcp
```
> Search for "chatbot" in AI SDK docs
⏺ docs_query(query: "chatbot", limit: 5, auto_retrieve: false)
  ⎿ Found 5 relevant sections (≈50 tokens)
```
**Result:** <100 tokens

## Why This Exists

Built to solve the problem of large documentation files timing out or consuming excessive tokens.

## Solution

| Operation | mcpdoc | Context7 | llms-txt-mcp |
|-----------|--------|----------|--------------|
| AI SDK Chatbot Docs | 251,431 tokens → ERROR | 15,000 tokens | <100 tokens |
| Structure Discovery | 5+ seconds | 2-3 seconds | Fast |
| Context Usage | Fails completely | 15K tokens | 50 tokens |
| Large File Support | Timeouts | Truncates | Streams |
| AI SDK llms.txt (30K+ lines) | Fails | Partial | 132 sections |

### Token Usage
```
mcpdoc:     [████████████████████████████████] 251K tokens → ERROR
Context7:   [████████████████] 15K tokens 
llms-txt:   [▪] <100 tokens
```

## Quick Start

```bash
# Install and run in one line (requires uv)
uvx llms-txt-mcp https://ai-sdk.dev/llms.txt

# Or install from source
git clone https://github.com/tenequm/llms-mcp-txt.git
cd llms-mcp-txt && uv sync
uv run llms-txt-mcp https://ai-sdk.dev/llms.txt
```

**For Claude Desktop:**

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "llms-txt-mcp": {
      "command": "uvx",
      "args": ["llms-txt-mcp", "https://ai-sdk.dev/llms.txt"]
    }
  }
}
```

That's it. Claude Code can now access AI SDK docs instantly.

## How It Works

```
URL → Parse YAML/Markdown → Embed → Search → Get Section
```

**Key insight:** Search first, fetch later. Never dump entire documentation.

1. **Parse**: Handles both AI SDK's YAML frontmatter and standard markdown
2. **Index**: Embeds sections with `BAAI/bge-small-en-v1.5`
3. **Search**: Semantic search returns top-k results (default: 10)
4. **Get**: Fetch exactly what you need with byte-capped responses

## Features

### 🚀 Instant Startup
- Lazy model loading for fast server startup
- Preindexing with stale source detection
- Background indexing - server available immediately

### 🎯 Surgical Access
- Search first - find relevant sections without dumping everything
- Byte-capped responses - protect your context window (default: 75KB)
- Human-readable IDs - use canonical URLs like `https://ai-sdk.dev/llms.txt#rag-agent`

### 📦 Zero Config Required
```bash
# Just works
uvx llms-txt-mcp https://ai-sdk.dev/llms.txt

# Multiple sources? Easy
uvx llms-txt-mcp https://ai-sdk.dev/llms.txt https://nextjs.org/llms.txt
```

### 🔄 Smart Caching
- TTL-based refresh (default: 24h)
- ETag/Last-Modified validation
- Persistent storage option for instant subsequent starts

### 🎨 Claude Code Optimized
- Minimal tool signatures
- Predictable responses
- No timeout surprises

## Usage Examples

### Search Documentation
```typescript
// In Claude Code (search only)
await docs_query({ 
  query: "RAG agent", 
  limit: 5,
  auto_retrieve: false
})

// Returns tiny, focused results without content
```

### Retrieve Specific Sections
```typescript
// Auto-retrieve top matches (recommended)
await docs_query({
  query: "RAG agent",
  limit: 5,
  auto_retrieve: true,
  max_bytes: 75000,
  merge: false
})

// Or retrieve explicit IDs
await docs_query({
  retrieve_ids: [
    "https://ai-sdk.dev/llms.txt#rag-agent-000"
  ],
  max_bytes: 75000,
  merge: false
})
```

### List Available Sources
```typescript
await docs_sources()

// Returns:
[
  {
    host: "ai-sdk.dev",
    docCount: 132,
    lastIndexed: "2024-01-..."
  }
]
```

## Configuration

### Basic (Most Users)
```bash
uvx llms-txt-mcp https://ai-sdk.dev/llms.txt
```

### With Options
```bash
uvx llms-txt-mcp https://ai-sdk.dev/llms.txt \
  --ttl 1h                    # Refresh every hour
  --store disk                # Persist embeddings
  --store-path ~/.llms-cache # Cache location
```

### Advanced Flags
- `--max-get-bytes N` - Byte limit for responses (default: 75000)
- `--embed-model MODEL` - Change embedding model (default: BAAI/bge-small-en-v1.5)
- `--no-preindex` - Disable automatic pre-indexing on startup
- `--no-background-preindex` - Wait for indexing to complete before serving

Note: The default `max-get-bytes` is 75KB. In practice, going 80KB+ can push responses close to a 25,000-token cap in some clients, so 75KB is a safe default.

## Performance

**Benchmarks on AI SDK llms.txt (30K+ lines, 132 sections):**

| Metric | Performance |
|--------|------------|
| Parse time | Fast (<2s for 30K+ lines) |
| Index time (first run) | Fast initial indexing |
| Index time (cached) | Instant (0ms) |
| Search latency | Fast semantic search |
| Memory usage | Lightweight |
| Model size | Small embedding model |

**Test Results:**
```
17 tests passing
Fast parsing performance verified
Minimal context usage confirmed
Handles 30K+ line files without breaking
```

## When to Use What

| Tool | Best For | Avoid When |
|------|----------|------------|
| **llms-txt-mcp** | AI SDK, large docs, Claude Code, search-first access | You need non-llms.txt formats |
| **mcpdoc** | Simple markdown files, small documentation | Large files, AI SDK docs, context matters |
| **context7** | Broad knowledge base, multiple sources | You need freshness control, deterministic sources |

## Development

### Setup
```bash
git clone https://github.com/tenequm/llms-mcp-txt.git
cd llms-mcp-txt
uv sync --all-extras
```

### Development Workflow
```bash
# Run the tool
uv run llms-txt-mcp --version

# Run from anywhere
uv run --directory /path/to/llms-mcp-txt llms-txt-mcp --version

# Development commands
uv run pytest                    # Run tests
uv run ruff check .             # Check code quality  
uv run ruff format .            # Format code
uv run mypy                     # Type check

# With arguments
uv run llms-txt-mcp https://ai-sdk.dev/llms.txt
```

### Shell Integration
```bash
llms() {
    uv run --directory /path/to/llms-mcp-txt llms-txt-mcp "$@"
}
```

### Local Testing with Inspector
```bash
npx @modelcontextprotocol/inspector uv run llms-txt-mcp https://ai-sdk.dev/llms.txt https://nextjs.org/docs/llms.txt https://hono.dev/llms.txt https://orm.drizzle.team/llms.txt https://zod.dev/llms.txt https://docs.docker.com/llms.txt
```

## Architecture

```
src/
├── server.py           # FastMCP server with all tools
├── parsers/
│   ├── __init__.py
│   ├── format_detector.py
│   └── parser.py       # Multi-format llms.txt parsing
└── __init__.py         # Package exports
```

**Key Design Decisions:**
- Simple, flat structure following KISS principles
- Streaming parser for large file support
- Lazy loading for instant startup
- Search-first to minimize context usage

## Contributing

Issues and PRs welcome! Please ensure:
- Tests pass (`uv run pytest`)
- Code is formatted (`uv run ruff format .`)
- Types check (`uv run mypy src/`)

## Credits

Built on [FastMCP](https://github.com/modelcontextprotocol/python-sdk) and the [Model Context Protocol](https://modelcontextprotocol.io).

## License

MIT - See [LICENSE](LICENSE)

---