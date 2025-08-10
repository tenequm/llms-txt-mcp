"""Parser for llms.txt files."""

from __future__ import annotations

import re

import yaml
from pydantic import BaseModel


class ParsedDoc(BaseModel):
    """Structure for parsed document sections."""

    title: str
    description: str = ""
    content: str = ""
    url: str = ""
    original_title: str = ""
    section: str = ""


class ParseResult(BaseModel):
    """Result from parsing llms.txt files."""

    docs: list[ParsedDoc]
    format: str


def detect_format(content: str) -> str:
    """Detect the format of an llms.txt file."""
    # Check for YAML frontmatter format (including both --- and ---- separators)
    if _is_yaml_frontmatter_format(content):
        return "yaml-frontmatter-full-llms-txt"

    # Check if it's a standard format (simple list)
    if _is_standard_format(content):
        return "standard-llms-txt"

    # Default to standard-full format
    return "standard-full-llms-txt"


def _is_yaml_frontmatter_format(content: str) -> bool:
    """Check if content has YAML frontmatter format."""
    lines = content.strip().split("\n")
    if not lines:
        return False

    # Check for both --- and ---- separators (vercel uses ----)
    separator_pattern = re.compile(r"^-{3,}$")

    # Look for YAML frontmatter pattern
    in_frontmatter = False
    has_title = False
    has_description = False

    for line in lines:
        if separator_pattern.match(line.strip()):
            if not in_frontmatter:
                in_frontmatter = True
            else:
                # End of a frontmatter block
                if has_title and has_description:
                    return True
                # Reset for next potential block
                has_title = False
                has_description = False
                in_frontmatter = False
        elif in_frontmatter:
            if line.startswith("title:"):
                has_title = True
            elif line.startswith("description:"):
                has_description = True

    return False


def _is_standard_format(content: str) -> bool:
    """Check if content appears to be an index (has markdown links in structured format)."""
    # Check for markdown links in the content
    link_pattern = r"^\s*-\s*\[.+\]\(.+\)"
    links = re.findall(link_pattern, content, re.MULTILINE)

    # If we have links, check if it's primarily a link index
    if links:
        lines = content.split("\n")
        non_empty_lines = [line for line in lines if line.strip()]

        # Case 1: H2 sections with link lists (original robust logic)
        h2_sections = re.findall(r"^## .+$", content, re.MULTILINE)
        if h2_sections:
            h2_with_links_pattern = r"^## .+\n+(?:^[\s-]*\[.+\]\(.+\).*$\n?)+"
            h2_with_links = re.findall(h2_with_links_pattern, content, re.MULTILINE)
            ratio = len(h2_with_links) / len(h2_sections)
            return ratio > 0.5

        # Case 2: H1 + high link density (primarily an index, not content)
        has_h1 = bool(re.search(r"^# [^#]", content, re.MULTILINE))
        if has_h1 and non_empty_lines:
            # Calculate link density: links / non-empty lines
            link_density = len(links) / len(non_empty_lines)
            # Only consider it standard format if >10% of lines are links
            # This filters out content-heavy documents with sparse links
            return link_density > 0.1

    return False


def parse_llms_txt(content: str) -> ParseResult:
    """Parse llms.txt content and return structured data."""
    format_type = detect_format(content)

    if format_type == "yaml-frontmatter-full-llms-txt":
        result = _parse_yaml_frontmatter(content)
    elif format_type == "standard-llms-txt":
        result = _parse_standard(content)
    else:  # standard-full-llms-txt
        result = _parse_standard_full(content)

    # Return ParseResult with format type
    return ParseResult(docs=result["docs"], format=format_type)


def _parse_yaml_frontmatter(content: str) -> dict[str, list[ParsedDoc]]:
    """Parse YAML frontmatter format."""
    docs = []
    lines = content.splitlines()
    i = 0
    n = len(lines)

    while i < n:
        # Check for separator (3+ dashes)
        if re.match(r"^-{3,}$", lines[i].strip()):
            i += 1
            yaml_lines = []
            # Collect YAML content until closing separator
            while i < n and not re.match(r"^-{3,}$", lines[i].strip()):
                yaml_lines.append(lines[i])
                i += 1

            # Check if we found a closing separator
            if i < n and re.match(r"^-{3,}$", lines[i].strip()):
                i += 1  # consume closing separator
            else:
                continue

            # Try to parse YAML
            try:
                meta = yaml.safe_load("\n".join(yaml_lines)) or {}
            except yaml.YAMLError:
                meta = {}

            # Skip if not a dict or missing title
            if not isinstance(meta, dict) or "title" not in meta:
                continue

            title = str(meta.get("title", ""))
            description = str(meta.get("description", ""))  # Empty string if missing

            content_lines = []
            # Collect content until we hit another YAML frontmatter block or EOF
            while i < n:
                # Check if we're at the start of a new YAML block
                if re.match(r"^-{3,}$", lines[i].strip()):
                    # Look ahead to see if this is a real YAML frontmatter
                    j = i + 1
                    found_title = False
                    # Look for title within next 20 lines (reasonable YAML header size)
                    while j < n and j < i + 20 and not re.match(r"^-{3,}$", lines[j].strip()):
                        if lines[j].startswith("title:"):
                            found_title = True
                            break
                        j += 1

                    # If we found a title and a closing separator, this is a new section
                    if found_title and j < n and j < i + 20:
                        # Check if there's a closing separator after the title
                        while j < n and j < i + 20:
                            if re.match(r"^-{3,}$", lines[j].strip()):
                                # This is definitely a new YAML block, stop collecting content
                                break
                            j += 1
                        else:
                            # No closing separator, treat as content
                            content_lines.append(lines[i])
                            i += 1
                            continue
                        # Found a real YAML block, stop here
                        break
                    # Not a YAML block, just content with dashes
                    content_lines.append(lines[i])
                    i += 1
                else:
                    content_lines.append(lines[i])
                    i += 1

            docs.append(
                ParsedDoc(
                    title=title,
                    description=description,
                    content="\n".join(content_lines).strip(),
                )
            )
        else:
            i += 1

    return {"docs": docs}


def _parse_standard(content: str) -> dict[str, list[ParsedDoc]]:
    """Parse standard llms.txt format with contextual titles."""
    docs = []
    lines = content.strip().split("\n")

    # Track document title (H1) and current section (H2) for context
    document_title = None
    current_section = None

    # Match markdown links: - [Title](URL) with optional description
    link_pattern = re.compile(r"^\s*-\s*\[([^\]]+)\]\(([^)]*)\)(?:\s*:\s*(.*))?")

    for line in lines:
        stripped = line.strip()

        # Capture H1 document title
        if stripped.startswith("# ") and not stripped.startswith("## "):
            document_title = stripped[2:].strip()
        # Track H2 section headers
        elif stripped.startswith("## "):
            current_section = stripped[3:].strip()
        # Extract links
        else:
            match = link_pattern.match(line)
            if match:
                original_title = match.group(1)
                url = match.group(2).strip()
                description = match.group(3).strip() if match.group(3) else None

                # Only add if URL is not empty
                if url:
                    # Build contextual title: "Document Section Original"
                    title_parts = []
                    if document_title:
                        title_parts.append(document_title)
                    if current_section:
                        title_parts.append(current_section)
                    title_parts.append(original_title)

                    contextual_title = " ".join(title_parts)

                    docs.append(
                        ParsedDoc(
                            title=contextual_title,
                            url=url,
                            original_title=original_title,
                            description=description
                            or contextual_title,  # Use contextual title as fallback
                            section=current_section or "",
                        )
                    )

    return {"docs": docs}


def _parse_standard_full(content: str) -> dict[str, list[ParsedDoc]]:
    """Parse standard-full llms.txt format."""
    docs = []
    lines = content.strip().split("\n")

    current_article = None
    current_content: list[str] = []

    for line in lines:
        # Match top-level headers (# Title)
        if line.startswith("# ") and not line.startswith("## "):
            # Save previous article if exists
            if current_article:
                current_article["content"] = "\n".join(current_content).strip()
                docs.append(ParsedDoc(**current_article))

            # Start new article
            title = line[2:].strip()
            if title:  # Only add non-empty titles
                current_article = {"title": title}
                current_content = []
        elif current_article:
            # Accumulate content for current article
            current_content.append(line)

    # Don't forget the last article
    if current_article:
        current_article["content"] = "\n".join(current_content).strip()
        docs.append(ParsedDoc(**current_article))

    return {"docs": docs}
