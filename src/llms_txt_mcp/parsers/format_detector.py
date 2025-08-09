"""Format detection for llms.txt files."""

import re


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
    """Check if content is standard llms.txt format (simple list)."""
    lines = content.strip().split("\n")

    # Standard format characteristics:
    # - Has a title (# Title)
    # - Has section headers (## Section)
    # - Mostly list items with links (high ratio of links to total lines)
    has_title = False
    section_headers = 0
    link_items = 0
    total_content_lines = 0

    for line in lines:
        stripped = line.strip()
        if stripped:  # Non-empty line
            total_content_lines += 1
            if stripped.startswith("# ") and not stripped.startswith("## "):
                has_title = True
            elif stripped.startswith("## "):
                section_headers += 1
            elif re.match(r"^\s*-\s*\[.+\]\(.+\)", line):
                link_items += 1

    if total_content_lines == 0:
        return False

    link_ratio = link_items / total_content_lines

    # Standard format: has title, sections, and mostly links (>50% of content)
    return has_title and section_headers > 0 and link_ratio > 0.5
