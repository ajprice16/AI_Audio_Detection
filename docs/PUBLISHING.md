# PyPI Publishing Guide

This guide explains how to publish the AI Audio Detector package to PyPI.

## Automatic Publishing (Recommended)

The package uses GitHub Actions for automatic publishing when you create a Git tag.

### Setup Required

1. **PyPI Account**: Create accounts on [PyPI](https://pypi.org) and [Test PyPI](https://test.pypi.org)

2. **API Tokens**: Generate API tokens for both PyPI and Test PyPI:
   - Go to Account Settings → API tokens
   - Create tokens with scope limited to this project
   - Add them as GitHub repository secrets:
     - `PYPI_API_TOKEN` - for production PyPI
     - `TEST_PYPI_API_TOKEN` - for test PyPI

3. **GitHub Secrets**: In your repository settings, add these secrets:
   ```
   PYPI_API_TOKEN=pypi-...your-token...
   TEST_PYPI_API_TOKEN=pypi-...your-test-token...
   ```

### Release Process

1. **Update Version**: Update version in `setup.py` and `pyproject.toml`

2. **Use Release Script** (Recommended):
   ```bash
   # Prepare release locally
   python release.py 1.0.1

   # Prepare and push immediately
   python release.py 1.0.1 --push
   ```

3. **Manual Release**:
   ```bash
   # Update version, commit changes
   git add .
   git commit -m "Release version 1.0.1"

   # Create and push tag
   git tag -a v1.0.1 -m "Version 1.0.1"
   git push origin main
   git push origin v1.0.1
   ```

4. **Automatic Publishing**: GitHub Actions will:
   - Run all tests and quality checks
   - Build the package
   - Publish to Test PyPI first
   - Test installation from Test PyPI
   - Publish to production PyPI
   - Create a GitHub release

### Monitoring

- Check [GitHub Actions](https://github.com/ajprice16/AI_Audio_Detection/actions) for build status
- Verify package on [Test PyPI](https://test.pypi.org/project/ai-audio-detector/)
- Verify package on [PyPI](https://pypi.org/project/ai-audio-detector/)

## Manual Publishing

If you need to publish manually:

```bash
# Install build tools
pip install build twine

# Build package
python -m build

# Check package
twine check dist/*

# Upload to Test PyPI (optional)
twine upload --repository testpypi dist/*

# Upload to PyPI
twine upload dist/*
```

## Version Management

### Semantic Versioning

Follow [semantic versioning](https://semver.org/):
- `MAJOR.MINOR.PATCH` (e.g., `1.0.0`)
- `MAJOR.MINOR.PATCH-PRERELEASE` (e.g., `1.0.0-beta.1`)

### Version Update Checklist

- [ ] Update version in `setup.py`
- [ ] Update version in `pyproject.toml`
- [ ] Update `CHANGELOG.md`
- [ ] Run tests: `pytest tests/`
- [ ] Run linting: `flake8 .`
- [ ] Build package: `python -m build`
- [ ] Check package: `twine check dist/*`

## Troubleshooting

### Common Issues

1. **Version Conflicts**: Ensure versions match in `setup.py` and `pyproject.toml`
2. **Missing Dependencies**: Verify all dependencies are in `requirements.txt`
3. **API Token Issues**: Check token permissions and expiration
4. **Build Failures**: Review GitHub Actions logs for specific errors

### Testing Installation

```bash
# Test from Test PyPI
pip install --index-url https://test.pypi.org/simple/ ai-audio-detector

# Test from PyPI
pip install ai-audio-detector

# Verify installation
python -c "import ai_audio_detector; print('Success!')"
ai-audio-detector --help
```

## Package Information

- **Package Name**: `ai-audio-detector`
- **PyPI URL**: https://pypi.org/project/ai-audio-detector/
- **Test PyPI URL**: https://test.pypi.org/project/ai-audio-detector/
- **Documentation**: See [README.md](../README.md)

## Security

- API tokens are stored as GitHub secrets
- Packages are published from GitHub Actions (trusted environment)
- All security scans must pass before publishing
- Both Test PyPI and production PyPI are used for validation
