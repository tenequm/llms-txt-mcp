"""Test article counting for different llms.txt file formats."""

from pathlib import Path

import pytest

from src.llms_txt_mcp.parsers.parser import parse_llms_txt


@pytest.fixture
def fixtures_dir() -> Path:
    """Return the fixtures directory path."""
    return Path("tests/fixtures")


@pytest.fixture
def load_fixture(fixtures_dir: Path):
    """Factory fixture to load file content."""

    def _load(filename: str) -> str:
        with open(fixtures_dir / filename, encoding="utf-8") as f:
            return f.read()

    return _load


class TestArticleCounting:
    """Test article counting across different llms.txt formats."""

    @pytest.mark.parametrize(
        "fixture_name,expected_count",
        [
            ("docs-docker-com-llms.txt", 1110),
            ("ai-sdk-dev-llms.txt", 132),
            ("hono-dev-llms-full.txt", 89),
            ("nextjs-org-docs-llms-full.txt", 366),
            ("orm-drizzle-team-llms-full.txt", 140),
            ("vercel-com-docs-llms-full.txt", 637),
            ("zod-dev-llms-full.txt", 17),
        ],
    )
    def test_article_count(self, load_fixture, fixture_name: str, expected_count: int):
        """Test that each fixture has the expected number of articles."""
        content = load_fixture(fixture_name)
        result = parse_llms_txt(content)

        assert "docs" in result, f"Parser result missing 'docs' key for {fixture_name}"
        assert len(result["docs"]) == expected_count, (
            f"{fixture_name}: Expected {expected_count} articles, got {len(result['docs'])}"
        )


class TestErrorHandling:
    """Test parser handles errors gracefully."""

    def test_parser_handles_empty_file(self):
        """Parser doesn't crash on empty input."""
        result = parse_llms_txt("")
        assert "docs" in result
        assert isinstance(result["docs"], list)
        assert len(result["docs"]) == 0

    def test_parser_handles_malformed_input(self):
        """Parser doesn't crash on garbage input."""
        garbage_inputs = [
            "@#$%^&*()_+",
            "---\nbroken: yaml: :\n---\n",
            "\x00\x01\x02\x03",  # null bytes
            "# " * 10000,  # many headers
            "\n" * 10000,  # many newlines
        ]

        for garbage in garbage_inputs:
            result = parse_llms_txt(garbage)
            assert "docs" in result, f"Parser crashed on: {garbage[:20]}..."
            assert isinstance(result["docs"], list), (
                f"Parser returned non-list for: {garbage[:20]}..."
            )
