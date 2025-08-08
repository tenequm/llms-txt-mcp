# Plan 1: llms-txt-mcp (Lean Documentation MCP)
*Fast, predictable, minimal‑context documentation access for Claude Code via llms.txt, with URL‑only config, human‑readable IDs, and freshness under your control.*

## Problem Statement

- **Real Impact**
  - AI SDK exposes its docs via `llms.txt` using repeated YAML frontmatter blocks; many tools either miss sections or dump huge context, making Claude Code outputs noisy and slow. Example: AI SDK `llms.txt` is 25K+ lines and >100 sections; naive listing pollutes the context.
  - Third‑party knowledge MCPs (e.g., broad, multi‑source indices) sometimes mix outdated and fresh pages (e.g., AI SDK v4 vs v5 during migration), leading to inconsistent answers and broken code.
- **Root Cause**
  - Parsing assumptions (expecting plain markdown headings) don’t fit AI SDK’s YAML‑block format; tools lack structured section boundaries.
  - Tool schemas return too much text (TOCs or full files) instead of small, surgical payloads.
  - No source allowlist or freshness controls (TTL + ETag) → randomness in versions.
- **Business Impact**
  - Developer time lost due to context bloat and wrong/outdated guidance.
  - Higher token costs; slower iteration; reduced trust in assistant outputs.

### Current State Analysis
- **✅ What’s Working**
  - The need is clear: lean, deterministic doc access improves Claude Code productivity.
  - Chroma provides simple ephemeral/persistent modes; sentence‑transformers provide fast small embedding models.
- **🔧 What Needs Improvement**
  - Robust parser for both AI SDK’s YAML‑block llms.txt and the official llms.txt.
  - Search‑first tool flow with tiny results; byte‑capped section retrieval; human‑readable IDs.
  - Strict source allowlist, TTL + ETag/Last‑Modified freshness, and unified index with host filtering.
- **📊 Data/Metrics**
  - AI SDK `llms.txt` >25K lines; target structure discovery <1s; search <150ms on ~1–5k chunks; `docs_get` payload capped at ~60KB.

## Core Objectives
- **Primary Goal**: Provide Claude Code with fast, minimal‑context, deterministic documentation retrieval from user‑specified `llms.txt` URLs.
- **Secondary Goals**:
  - Parse both AI SDK YAML‑frontmatter llms.txt and official llms.txt formats cleanly.
  - Unified index across multiple sources; filter by host when needed.
  - Human‑readable IDs (canonical URLs + anchors) and small, predictable tool schemas.
  - Freshness you control: TTL + ETag/Last‑Modified revalidation.
- **Success Metrics**:
  - Search p95 <150ms; single `docs_get` p95 <50ms (local, indexed).
  - `docs_get` responses respect default 60KB cap and preserve order.
  - No full TOC or full‑file dumps in normal flows; tools remain <~250 tokens of schema.

### MVP Decision Filter *(30 seconds max)*
- [x] **Serve users/revenue?** Yes — immediate productivity gains, lower token use.
- [x] **Build now or defer?** Build now — core to your daily workflow.
- [x] **Maintenance burden?** Low — small surface; modest deps; TTL/ETag reduce churn.

---

## Phase-Based Implementation

> Each phase ends in a working, committable state. Fewer phases where possible; no time estimates.

### Phase 1: Foundation/Core Implementation

**Goals:**
- Implement a lean MCP (“docs‑mcp”) that indexes user‑provided `llms.txt` URLs and exposes four tiny tools: `docs_sources`, `docs_search`, `docs_get`, `docs_refresh`.
- Ensure search‑first UX and byte‑capped retrieval with human‑readable IDs.

**Phase Scope:**
- **Must Have**
  - URL‑only config; strict allowlist; Chroma backend (ephemeral by default); `thenlper/gte-small` embeddings.
  - Parsers for AI SDK YAML‑block llms.txt and official llms.txt (H2 sections, H1 fallback).
  - Unified collection `docs` with host metadata; `docs_search(q, hosts?, k)`; `docs_get(ids[], max_bytes, merge?)`.
  - Freshness: TTL + ETag/Last‑Modified; `docs_refresh` to force reindex.
  - Streaming fetch + incremental parse for large files.
- **Can Defer**
  - Disk persistence flag (`--store=disk`), stale doc cleanup on reindex (can land in P2 if needed), diversity (MMR).
- **Won’t Do**
  - Name:url configs; backend switching flags; listing full TOCs; complex multi‑page crawl.

**Implementation:**
```python
# CLI (positional URLs also accepted)
URL... --ttl 24h --timeout 30 --embed-model thenlper/gte-small \
--store memory [--store-path PATH] --max-get-bytes 60000
# Or with flag form:
--sources URL... --ttl 24h --timeout 30 --embed-model thenlper/gte-small \
--store memory [--store-path PATH] --max-get-bytes 60000

# Tools (FastMCP)
- docs_sources() -> [{ source_url, host, lastIndexed, docCount }]
- docs_search(q, hosts?: [str], k?: int=5) -> [{ id, source, title, snippet, score }]
- docs_get(ids: [str], max_bytes?: int=60000, merge?: bool=false, depth?: int=0)
- docs_refresh(source?: str) -> { refreshed, counts }

# IDs
id = canonical_url (+ '#anchor')

# Parser selection
try AI SDK YAML-frontmatter blocks -> blocks to sections
else parse official headings -> H2 sections (H1 fallback)
```

**Quick Validation:**
- [ ] `docs_search` returns tiny items (no content blobs), `k<=20`.
- [ ] `docs_get` respects `max_bytes` and preserves order.
- [ ] Indexing AI SDK `llms.txt` completes and `docCount>0`.

**Success Criteria:**
- [ ] `uv run llms-txt-mcp --sources https://ai-sdk.dev/llms.txt --preindex` produces working tools.
- [ ] Search p95 <150ms (local, indexed); `docs_get` p95 <50ms single section.
- [ ] Schemas stay small and deterministic.

---

### Phase 2: Enhancement/Extension

**Goals:**
- Improve reliability and extend storage options without expanding the tool surface.

**Phase Scope:**
- **Must Have**
  - Disk persistence: `--store=disk --store-path PATH` via Chroma PersistentClient.
  - Stale doc removal on reindex per source (remove ids no longer present).
  - Background TTL refresh triggered by tool calls (non‑blocking).
- **Can Defer**
  - Search diversity (MMR), per‑host caps in results.
- **Won’t Do**
  - Switching to alternate backends publicly; keep internal abstraction only.

**Implementation:**
- Add PersistentClient wiring; guard writes; delete stale by `source_url`.
- Schedule refresh when `now - last_indexed > ttl` on `docs_search`/`docs_get`.

**Quick Validation:**
- [ ] Restart preserves index if `--store=disk`.
- [ ] Reindex removes stale content; no orphaned hits.
- [ ] Tool latency unaffected by background refresh.

**Success Criteria:**
- [ ] Persistent runs behave the same as ephemeral with data retained.
- [ ] Stale removal verified by changing source content.
- [ ] Background refresh does not block tool responses.

---

### Phase 3: Polish/Integration (if needed)

**Goals:**
- Add optional capabilities and prepare for broader ingestion.

**Phase Scope:**
- **Must Have**
  - None, if P1–P2 already meet needs.
- **Can Defer**
  - Doc‑site ingestion (`https://ai-sdk.dev/docs/...`) into same chunk schema.
  - Diversity (MMR), per‑host caps, version pinning.
- **Won’t Do**
  - Over‑optimization or large tool surface expansion.

**Implementation:**
- Introduce a `site` ingest pipeline that crawls allowed roots and reuses the unified index.

**Quick Validation:**
- [ ] Site‑ingested chunks interoperate seamlessly with `docs_search`/`docs_get`.
- [ ] No regressions to llms.txt flows.

**Success Criteria:**
- [ ] Cross‑source answers (AI SDK + Tailwind + shadcn + Next.js) from one search.
- [ ] Payload caps and schemas remain stable.

---

## Key Technical Decisions

### Architecture Choices
- **Chroma (ephemeral by default)**: Zero‑ops locally; optional persistence later. Simple metadata filtering and a single unified collection.
- **Human‑readable IDs**: Canonical URLs (+ anchors) for transparent, copy‑pasteable references.

### Implementation Patterns
- **Follow Existing Conventions**: Search‑first → get flow; tiny tool schemas; deterministic shapes.
- **Error Handling**: Return `{ "error": "message" }` for tool errors; validate inputs; strict allowlist for URLs ending with `/llms.txt`.
- **Testing Strategy**: Unit tests for parsers; mocked HTTP (ETag/304); index/retrieval integration; performance smoke tests on large `llms.txt`.

### Dependencies and Integrations
- **External**: `httpx`, `sentence-transformers` (default `thenlper/gte-small`), `chromadb`, `mcp`.
- **Database Changes**: None (Chroma collection only).
- **Frontend Impact**: None; tool consumers are MCP clients (e.g., Claude Code).

## Risk Mitigation

- **Rollback Plan**: Disable server entry or revert CLI flags; ephemeral mode leaves no on‑disk state.
- **Compatibility**: Small, stable schemas; keep tool list minimal; avoid breaking param renames.
- **Testing**: Per‑phase validation; CI for unit/integration tests; perf checks on large fixtures.
- **Monitoring**: Log fetch/index times, chunk counts, query latency, response sizes.

## Success Metrics

**Technical:**
- [ ] Search p95 <150ms; get p95 <50ms (local, indexed)
- [ ] Payload cap enforced (default 60KB) and configurable
- [ ] Parser handles AI SDK YAML‑block and official llms.txt accurately

**Business:**
- [ ] Reduced token usage vs prior tools (no TOC/full‑file dumps)
- [ ] Faster doc lookup → improved developer throughput
- [ ] Deterministic/fresh results (TTL + ETag)

**Quality:**
- [ ] No regressions in tool outputs
- [ ] All tests pass (unit + integration)
- [ ] README and CLI help are up‑to‑date

---

## Implementation Notes

### Solo Dev Context (for future you)
- **Key files**: `src/llms_txt_mcp/server.py` (server + tools), `README.md`, `docs/implementation-plan.md`.
- **Gotchas**: AI SDK files are huge; ensure streaming fetch; YAML variance; ensure slug stability; guard Chroma upsert vs add; embedding time dominates.
- **Quick wins**: Host filter in search; byte cap in get; preindex on launch for instant first query.
- **Debug shortcuts**: Log counts and durations; test against a local cached `llms.txt` to iterate quickly.

### Before Starting
- [ ] Review MCP tool shapes for minimality
- [ ] Confirm CLI flags match README
- [ ] Identify acceptance tests for AI SDK `llms.txt`

### During Implementation
- [ ] Commit at end of each phase
- [ ] Run unit/integration tests locally
- [ ] Update README and examples as you go
- [ ] Monitor logs for fetch/index timings

### Phase Completion Criteria
- ✅ Working end‑to‑end
- ✅ Tested (unit + integration)
- ✅ Documented
- ✅ Committable

### Quality Gates
- [ ] No breaking changes to existing functionality
- [ ] Lint + type checks pass
- [ ] Integration tests pass
- [ ] Performance targets maintained

---

## Template Usage Guidelines (kept for reference)
- Use multiple phases for complex/high‑risk changes; otherwise prefer fewer phases with complete states.
- Each phase must add user value; avoid setup‑only phases.
- Plan for early feedback; keep tool schemas tiny and stable.
