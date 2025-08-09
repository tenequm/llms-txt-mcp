"""Pytest configuration and shared fixtures."""

from collections import defaultdict

import pytest


def pytest_terminal_summary(terminalreporter, exitstatus):
    """Print detailed test results summary."""
    if not hasattr(terminalreporter, "stats"):
        return

    passed_tests = terminalreporter.stats.get("passed", [])
    failed_tests = terminalreporter.stats.get("failed", [])

    # Categorize tests
    categories = defaultdict(list)
    for item in passed_tests + failed_tests:
        test_path = str(item.nodeid)
        if "test_format_detection" in test_path:
            categories["Format Detection"].append(item)
        elif "test_parsers" in test_path:
            categories["Parser Accuracy"].append(item)
        elif "test_server" in test_path:
            if "Performance" in test_path:
                categories["Performance"].append(item)
            else:
                categories["MCP Tools"].append(item)

    # Print summary
    print("\n" + "=" * 60)
    print("Test Summary:")
    print("=" * 60)

    for category, items in sorted(categories.items()):
        passed = sum(1 for item in items if item in passed_tests)
        failed = sum(1 for item in items if item in failed_tests)
        status = "✓" if failed == 0 else "✗"
        print(f"{status} {category}: {passed} passed", end="")
        if failed > 0:
            print(f", {failed} failed", end="")
        print()

    # Overall summary
    print("-" * 60)
    total_passed = len(passed_tests)
    total_failed = len(failed_tests)
    print(f"Total: {total_passed} passed", end="")
    if total_failed > 0:
        print(f", {total_failed} failed", end="")
    print()


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "performance: marks tests that validate performance claims")


def pytest_collection_modifyitems(session, config, items):
    """Add description to test items for better output."""
    for item in items:
        # Extract test class and method
        test_class = item.cls.__name__ if item.cls else ""
        test_name = item.name

        # Create readable description
        if "test_format_detection" in str(item.fspath):
            if "yaml" in test_name.lower():
                item.add_marker(pytest.mark.format_detection)
        elif "test_parsers" in str(item.fspath):
            item.add_marker(pytest.mark.parsing)
        elif "test_server" in str(item.fspath):
            if "Performance" in test_class:
                item.add_marker(pytest.mark.performance)
            else:
                item.add_marker(pytest.mark.mcp_tools)


@pytest.fixture(scope="session")
def test_session_info():
    """Provide test session information."""
    return {
        "test_count": 24,
        "categories": ["format_detection", "parsing", "mcp_tools", "performance"],
    }
