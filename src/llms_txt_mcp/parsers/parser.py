"""Parser for llms.txt files."""

import re
from typing import Any

import yaml

from .format_detector import detect_format


def parse_llms_txt(content: str) -> dict[str, Any]:
    """Parse llms.txt content and return structured data."""
    format_type = detect_format(content)

    if format_type == "yaml-frontmatter-full-llms-txt":
        return _parse_yaml_frontmatter(content)
    elif format_type == "standard-llms-txt":
        return _parse_standard(content)
    else:  # standard-full-llms-txt
        return _parse_standard_full(content)


def _parse_yaml_frontmatter(content: str) -> dict[str, Any]:
    """Parse YAML frontmatter format."""
    articles = []
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
                    else:
                        # Not a YAML block, just content with dashes
                        content_lines.append(lines[i])
                        i += 1
                else:
                    content_lines.append(lines[i])
                    i += 1

            articles.append(
                {
                    "title": title,
                    "description": description,
                    "content": "\n".join(content_lines).strip(),
                }
            )
        else:
            i += 1

    return {"articles": articles}


def _parse_standard(content: str) -> dict[str, Any]:
    """Parse standard llms.txt format."""
    articles = []
    lines = content.strip().split("\n")

    # Track current section for context
    current_section = None

    # Match markdown links: - [Title](URL) with optional description
    link_pattern = re.compile(r"^\s*-\s*\[([^\]]+)\]\(([^)]*)\)(?:\s*:\s*(.*))?")

    for line in lines:
        # Track section headers
        if line.strip().startswith("## "):
            current_section = line.strip()[3:].strip()
        # Extract links
        else:
            match = link_pattern.match(line)
            if match:
                title = match.group(1)
                url = match.group(2).strip()
                description = match.group(3).strip() if match.group(3) else None

                # Only add if URL is not empty
                if url:
                    article = {"title": title, "url": url}
                    if description:
                        article["description"] = description
                    if current_section:
                        article["section"] = current_section
                    articles.append(article)

    return {"articles": articles}


def _parse_standard_full(content: str) -> dict[str, Any]:
    """Parse standard-full llms.txt format."""
    articles = []
    lines = content.strip().split("\n")

    current_article = None
    current_content = []

    for line in lines:
        # Match top-level headers (# Title)
        if line.startswith("# ") and not line.startswith("## "):
            # Save previous article if exists
            if current_article:
                current_article["content"] = "\n".join(current_content).strip()
                articles.append(current_article)

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
        articles.append(current_article)

    return {"articles": articles}
