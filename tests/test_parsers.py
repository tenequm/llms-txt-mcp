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
    
    @pytest.mark.parametrize("fixture_name,expected_count", [
        ("docs-docker-com-llms.txt", 721),
        ("ai-sdk-dev-llms.txt", 132),
        ("hono-dev-llms-full.txt", 88),
        ("nextjs-org-docs-llms-full.txt", 363),
        ("orm-drizzle-team-llms-full.txt", 140),
        ("vercel-com-docs-llms-full.txt", 640),
        ("zod-dev-llms-full.txt", 17),
    ])
    def test_article_count(self, load_fixture, fixture_name: str, expected_count: int):
        """Test that each fixture has the expected number of articles."""
        content = load_fixture(fixture_name)
        result = parse_llms_txt(content)
        
        assert "articles" in result, f"Parser result missing 'articles' key for {fixture_name}"
        assert len(result["articles"]) == expected_count, (
            f"{fixture_name}: Expected {expected_count} articles, "
            f"got {len(result['articles'])}"
        )