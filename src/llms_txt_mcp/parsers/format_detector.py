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
