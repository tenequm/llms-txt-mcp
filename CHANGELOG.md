## [0.1.0] - 2025-01-09

Initial release of llms-txt-mcp - a lean MCP server for fast documentation access via llms.txt parsing.

### Features
- **MCP Tools**: `docs_query` (unified search + retrieval), `docs_sources`, `docs_refresh`
- **Multi-format parsing**: AI SDK YAML frontmatter and standard markdown formats
- **Semantic search**: Using BAAI/bge-small-en-v1.5 embeddings with ChromaDB
- **Smart caching**: TTL-based with ETag/Last-Modified validation
- **Byte-capped responses**: Default 75KB limit to protect context window
- **Large file support**: Handles 30K+ line files without issues
- **Configuration**: Flexible CLI options for TTL, storage, embedding model, and retrieval settings


