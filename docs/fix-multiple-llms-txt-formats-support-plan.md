# Plan: llms.txt Parser with Format Detection
*TDD approach to build parsers for different llms.txt formats*

## Problem Statement

Current server can't accurately parse different llms.txt formats. Each site uses different structures, leading to wrong article counts.

- **Real Impact**: Can't count articles correctly across different formats
- **Root Cause**: No format detection or specialized parsers
- **Business Impact**: Poor data extraction

## Core Objectives

- Detect llms.txt format automatically
- Parse each format correctly
- Return accurate article counts

---

## Phase 1: TDD Foundation

**Goals:**
- Write tests for format detection
- Write tests for article counting
- All tests should fail initially

**File Structure:**
- Tests: `tests/test_format_detection.py`, `tests/test_parsers.py`
- Parser code: `src/llms_txt_mcp/parsers/` (to be created in Phase 2)

**Test Requirements:**

Format detection tests:
- `zod-dev-llms-full.txt` → 'standard-full-llms-txt'
- `hono-dev-llms-full.txt` → 'standard-full-llms-txt'  
- `orm-drizzle-team-llms-full.txt` → 'standard-full-llms-txt'
- `ai-sdk-dev-llms.txt` → 'yaml-frontmatter-full-llms-txt'
- `nextjs-org-docs-llms-full.txt` → 'yaml-frontmatter-full-llms-txt'
- `vercel-com-docs-llms-full.txt` → 'yaml-frontmatter-full-llms-txt'
- `docs-docker-com-llms.txt` → 'standard-llms-txt'

Article count tests:
- `docs-docker-com-llms.txt` → 721 articles
- `ai-sdk-dev-llms.txt` → 132 articles
- `hono-dev-llms-full.txt` → 88 articles
- `nextjs-org-docs-llms-full.txt` → 363 articles
- `orm-drizzle-team-llms-full.txt` → 140 articles
- `vercel-com-docs-llms-full.txt` → 640 articles
- `zod-dev-llms-full.txt` → 17 articles

**Format Detection Notes:**
- Check if URL ends with `/llms-full.txt` - if `/llms.txt` provided, should check if `/llms-full.txt` exists and prefer it
- YAML frontmatter detection: Must verify document has separators (3+ dashes minimum), must contain both `title:` and `description:` fields

**Article Definition:**
- An article is defined by having at least a title

**Success Criteria:**
- [ ] Tests written for format detection
- [ ] Tests written for article counting
- [ ] All tests fail (no implementation yet)

---

## Phase 2: Implementation

**Goals:**
- Make all tests pass
- Keep it simple

**Success Criteria:**
- [ ] All format detection tests pass
- [ ] All article count tests pass  
- [ ] Code is minimal and clean

---

## Phase 3: Integration

**Goals:**
- Integrate into existing server

**Success Criteria:**
- [ ] Parser integrated into server
- [ ] All functionality works