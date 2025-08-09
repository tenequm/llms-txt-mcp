# Release Process

This document outlines the automated release process for llms-txt-mcp.

## Overview

The project uses modern Python packaging automation with:
- **uv**: Fast Python package management (NEVER use pip!)
- **hatch-vcs**: Automatic versioning from git tags
- **git-cliff**: Automatic changelog generation from conventional commits  
- **GitHub Actions**: Automated PyPI publishing and GitHub releases
- **Trusted Publishing**: Secure PyPI authentication (no API keys)

## One-Time Setup (Already Done)

1. **PyPI Trusted Publishing Configuration**:
   - Go to [PyPI Manage](https://pypi.org/manage/projects/) → llms-txt-mcp → Settings → Publishing
   - Add GitHub publisher: `tenequm/llms-txt-mcp`
   - Environment: `pypi`

2. **GitHub Environment Protection**:
   - Repository Settings → Environments → `pypi` 
   - Add yourself as required reviewer
   - Protection rules: Selected branches and tags

## Daily Workflow

### Making Changes
```bash
# Use conventional commit messages for automatic changelog generation
git commit -m "feat: add new parsing feature"
git commit -m "fix: resolve edge case in YAML parsing" 
git commit -m "docs: update README examples"
```

### Creating a Release

1. **Tag the release**:
```bash
git tag v0.2.0
git push --tags
```

2. **That's it!** GitHub Actions automatically:
   - ✅ Generates/updates CHANGELOG.md from commit history
   - ✅ Commits changelog back to repository  
   - ✅ Creates GitHub release with release notes
   - ✅ Builds Python package with correct version (from tag)
   - ✅ Publishes to PyPI with trusted publishing
   - ✅ Verifies successful upload

## Conventional Commit Types

Use these prefixes for automatic changelog categorization:

- `feat:` - New features
- `fix:` - Bug fixes
- `docs:` - Documentation changes
- `refactor:` - Code refactoring
- `perf:` - Performance improvements
- `test:` - Test changes
- `chore:` - Maintenance tasks

## Version Numbering

Follow [Semantic Versioning](https://semver.org/):
- `v1.0.0` - Major release (breaking changes)
- `v0.2.0` - Minor release (new features)  
- `v0.1.1` - Patch release (bug fixes)

## Troubleshooting

### Release Failed
- Check GitHub Actions logs in the "Actions" tab
- Verify PyPI trusted publishing configuration
- Ensure you're added as reviewer for `pypi` environment

### Version Issues  
- Version automatically comes from git tags (no manual changes needed)
- Development versions show as `0.1.1.dev3+g1a2b3c4`
- Tagged versions show as clean `0.1.1`

### Changelog Issues
- Ensure commits follow conventional commit format
- Check git-cliff configuration in `pyproject.toml`
- Manually run `git-cliff --output CHANGELOG.md` locally to test

## Manual Release (Emergency Only)

If automation fails, manual release:
```bash
# Build locally with uv
uv sync
uv build

# Upload manually with uv (requires PyPI token)
uv publish
```

## Key Files

- `pyproject.toml` - Project configuration, versioning, changelog settings
- `.github/workflows/release.yml` - Automated release workflow
- `CHANGELOG.md` - Auto-generated changelog (don't edit manually)
- `src/_version.py` - Auto-generated version file (git-ignored)