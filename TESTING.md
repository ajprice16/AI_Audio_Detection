# Testing Guide for AI Audio Detector

This document describes the comprehensive testing setup for the AI Audio Detector package.

## Quick Start

```bash
# Install development dependencies
make install-dev

# Set up pre-commit hooks
make setup-hooks

# Run all tests
make test

# Run quick development checks
make quick-check
```

## Test Structure

```
tests/
├── __init__.py              # Test package initialization
├── conftest.py              # Pytest configuration and fixtures
├── test_feature_extractor.py  # Unit tests for feature extraction
├── test_detector.py         # Unit tests for main detector class
├── test_integration.py      # Integration tests
└── benchmark.py             # Performance benchmarks
```

## Running Tests

### All Tests
```bash
pytest tests/ -v
# or
make test
```

### Unit Tests Only
```bash
pytest tests/ -v -m "unit"
# or
make test-unit
```

### Integration Tests Only
```bash
pytest tests/ -v -m "integration"
# or
make test-integration
```

### With Coverage Report
```bash
pytest tests/ --cov=ai_audio_detector --cov-report=html
# or
make test-coverage
```

### Benchmark Tests
```bash
python tests/benchmark.py
# or
make test-benchmark
```

## Code Quality Checks

### Linting
```bash
flake8 .
# or
make lint
```

### Code Formatting
```bash
black .
# or
make format
```

### Type Checking
```bash
mypy ai_audio_detector.py --ignore-missing-imports
# or
make type-check
```

### Security Checks
```bash
bandit -r . -x tests/
safety check
# or
make security
```

## Pre-commit Hooks

Pre-commit hooks automatically run quality checks before each commit:

```bash
# Install hooks
pre-commit install
# or
make setup-hooks

# Run hooks manually
pre-commit run --all-files
# or
make pre-commit
```

## CI/CD Pipeline

The GitHub Actions workflow (`.github/workflows/ci.yml`) runs:

1. **Multi-platform testing** on Ubuntu, macOS, and Windows
2. **Multi-Python version testing** (3.8, 3.9, 3.10, 3.11)
3. **Code quality checks** (linting, type checking)
4. **Security scanning** (bandit, safety)
5. **Coverage reporting** to Codecov
6. **Performance benchmarks**
7. **Package building and validation**

## Test Configuration

### pytest.ini
- Test discovery patterns
- Coverage settings (80% minimum)
- Warning filters
- Custom markers

### .flake8
- Line length: 127 characters
- Complexity limit: 10
- Import order: Google style
- Docstring convention: Google

### .pre-commit-config.yaml
- Code formatting (black)
- Linting (flake8)
- Type checking (mypy)
- Security (bandit)
- Basic checks (trailing whitespace, etc.)

## Test Dependencies

Development dependencies are specified in `requirements-dev.txt`:

- **pytest**: Test framework
- **pytest-cov**: Coverage reporting
- **flake8**: Linting
- **black**: Code formatting
- **mypy**: Type checking
- **bandit**: Security analysis
- **safety**: Dependency vulnerability scanning
- **pre-commit**: Git hooks
- **soundfile**: Audio file generation for tests

## Writing New Tests

### Unit Tests
```python
def test_feature_extraction():
    extractor = AudioFeatureExtractor()
    features = extractor.extract_benford_features([1.23, 4.56, 7.89])
    assert 'chi2_p' in features
    assert isinstance(features['chi2_p'], float)
```

### Integration Tests
```python
@pytest.mark.integration
def test_end_to_end_workflow(temp_dir):
    detector = AIAudioDetector(base_dir=temp_dir)
    # ... test complete workflow
```

### Using Fixtures
```python
def test_with_detector(ai_detector, sample_features_data):
    results = ai_detector.train_models(sample_features_data)
    assert isinstance(results, dict)
```

## Performance Testing

The benchmark suite measures:
- **Feature extraction** throughput
- **Model training** time
- **Prediction** performance (sequential vs batch)
- **Memory usage** profiling

Run benchmarks with:
```bash
python tests/benchmark.py
```

Results are saved to `benchmark-results.json`.

## Troubleshooting

### Common Issues

1. **soundfile import error**: Install with `pip install soundfile`
2. **FFmpeg not found**: Install system dependency
3. **Memory profiler issues**: Install with `pip install memory-profiler`

### CI Failures

Check the GitHub Actions logs for:
- Dependency installation failures
- Test failures with detailed output
- Code quality violations
- Security vulnerabilities

### Coverage Issues

If coverage drops below 80%:
1. Identify uncovered lines in the HTML report
2. Add tests for missing coverage
3. Consider excluding non-critical code from coverage

## Best Practices

1. **Write tests first** (TDD approach)
2. **Use descriptive test names** that explain what is being tested
3. **Test edge cases** and error conditions
4. **Keep tests independent** and isolated
5. **Use fixtures** for common setup
6. **Mock external dependencies** when appropriate
7. **Test both success and failure paths**
8. **Maintain high code coverage** (aim for >90%)

## Continuous Integration

The CI pipeline ensures:
- All tests pass on multiple platforms
- Code quality standards are met
- Security vulnerabilities are caught early
- Performance regressions are detected
- Package builds correctly

Push to any branch triggers the full CI pipeline, ensuring code quality and test coverage before merging.
