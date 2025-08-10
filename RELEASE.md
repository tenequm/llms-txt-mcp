## Release Quick Guide

Minimal steps to publish a new version. Version comes from the git tag. No PyPI token needed (trusted publishing).

### 1) Tag and push
```bash
git tag vX.Y.Z
git push --tags
```

### 2) What happens automatically
- Changelog is generated and committed
- GitHub Release is created
- Package is built and published to PyPI (trusted publishing)

### 3) Verify
- Check GitHub Actions → release workflow is green
- Confirm version on PyPI project page

### Fix a wrong tag
```bash
git tag -d vX.Y.Z
git push --delete origin vX.Y.Z
git tag vX.Y.Z
git push --tags
```

### Manual publish (only if automation fails)
```bash
uv sync
uv build
uv publish
```

### Key references
- `pyproject.toml` – versioning and changelog config
- `.github/workflows/release.yml` – release workflow
- `CHANGELOG.md` – auto-generated
- `src/_version.py` – auto-generated (git-ignored)