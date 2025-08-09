"""Test format detection for different llms.txt file formats."""

from pathlib import Path

import pytest

from src.llms_txt_mcp.parsers.format_detector import detect_format


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


class TestStandardFullFormat:
    """Test detection of standard-full-llms-txt format."""

    @pytest.mark.parametrize(
        "fixture_name",
        [
            "zod-dev-llms-full.txt",
            "hono-dev-llms-full.txt",
            "orm-drizzle-team-llms-full.txt",
        ],
    )
    def test_detect_standard_full_format(self, load_fixture, fixture_name: str):
        """Test that standard full format files are correctly detected."""
        content = load_fixture(fixture_name)
        assert detect_format(content) == "standard-full-llms-txt"


class TestYamlFrontmatterFormat:
    """Test detection of yaml-frontmatter-full-llms-txt format."""

    @pytest.mark.parametrize(
        "fixture_name",
        [
            "ai-sdk-dev-llms.txt",
            "nextjs-org-docs-llms-full.txt",
            "vercel-com-docs-llms-full.txt",
        ],
    )
    def test_detect_yaml_frontmatter_format(self, load_fixture, fixture_name: str):
        """Test that YAML frontmatter format files are correctly detected."""
        content = load_fixture(fixture_name)
        assert detect_format(content) == "yaml-frontmatter-full-llms-txt"


class TestStandardFormat:
    """Test detection of standard-llms-txt format."""

    def test_detect_standard_format_docker(self, load_fixture):
        """Test that Docker docs use standard format."""
        content = load_fixture("docs-docker-com-llms.txt")
        assert detect_format(content) == "standard-llms-txt"
