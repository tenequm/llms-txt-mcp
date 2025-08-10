## Release Quick Guide

Minimal steps to publish a new version. Version comes from the git tag. No PyPI token needed (trusted publishing).

### 1) Tag and push
```bash
git tag vX.Y.Z
git push --tags
```

### 2) What happens automatically
- Changelog is generated from commit history (using git-cliff)
- GitHub Release is created with the changelog
- Package is built and published to PyPI (trusted publishing)
- Package upload is verified

### 3) Verify
- Check GitHub Actions → release workflow is green
- Check the GitHub Release page for the changelog
- Confirm version on PyPI project page

### Fix a wrong tag
```bash
git tag -d vX.Y.Z
git push --delete origin vX.Y.Z
git tag vX.Y.Z
git push --tags
```

### Manual release (if automation fails)

#### Generate changelog locally
```bash
# For unreleased changes (first release)
git-cliff --config pyproject.toml --unreleased --strip all

# For latest tag
git-cliff --config pyproject.toml --latest --strip all

# For specific tag range
git-cliff --config pyproject.toml v0.1.0..v0.2.0 --strip all
```

#### Build and publish manually
```bash
uv sync
uv build
uv publish
```

### Key references
- `pyproject.toml` – versioning and git-cliff changelog config
- `.github/workflows/release.yml` – automated release workflow
- GitHub Releases – where changelogs are stored
- `src/_version.py` – auto-generated version file (git-ignored)

### Notes
- No manual CHANGELOG.md file - all changelogs are in GitHub Releases
- First release (v0.1.0) gets a custom introduction message
- Subsequent releases use auto-generated changelogs from commits
- Commit messages should follow conventional commits format for better changelogs