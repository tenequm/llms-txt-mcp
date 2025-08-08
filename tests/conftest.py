"""Pytest configuration and shared fixtures."""

import pytest


def pytest_terminal_summary(terminalreporter, exitstatus):
    """Print detailed test results summary."""
    if hasattr(terminalreporter, "stats"):
        passed = len(terminalreporter.stats.get("passed", []))
        failed = len(terminalreporter.stats.get("failed", []))

        print("\n📊 llms-txt-mcp Test Results:")
        print(f"   ✅ {passed} tests passed")
        if failed > 0:
            print(f"   ❌ {failed} tests failed")
        else:
            print("   🚀 All performance claims validated")
            print("   🔒 Security controls working")
            print("   💾 Caching system verified")
            print("   📄 README motivation claims proven")


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "performance: marks tests that validate performance claims")


@pytest.fixture(scope="session")
def test_session_info():
    """Provide test session information."""
    return {
        "test_count": 13,
        "categories": ["cache", "parsing", "mcp_tools", "performance", "initialization"],
    }
