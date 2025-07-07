# Contributing to AI Audio Detector

Thank you for your interest in contributing to the AI Audio Detector! This document provides guidelines and information for contributors.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [How to Contribute](#how-to-contribute)
- [Development Setup](#development-setup)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Submitting Changes](#submitting-changes)

## Code of Conduct

This project adheres to a code of conduct. By participating, you are expected to uphold this code. Please be respectful and constructive in all interactions.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/yourusername/ai-audio-detector.git
   cd ai-audio-detector
   ```
3. **Create a branch** for your feature or bug fix:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## How to Contribute

### Reporting Bugs

Before creating bug reports, please check the existing issues to avoid duplicates. When creating a bug report, include:

- **Clear title and description**
- **Steps to reproduce** the issue
- **Expected vs actual behavior**
- **Environment details** (OS, Python version, package versions)
- **Audio file details** if relevant (format, duration, source)
- **Error messages or logs**

### Suggesting Enhancements

Enhancement suggestions are welcome! Please include:

- **Clear title and description**
- **Use case and motivation**
- **Detailed explanation** of the proposed functionality
- **Examples** if applicable

### Pull Requests

Good pull requests include:

- **Focused changes** that address a single issue
- **Clear commit messages**
- **Updated documentation** if needed
- **Tests** for new functionality
- **No regression** in existing functionality

## Development Setup

### Prerequisites

- Python 3.7+
- Git
- Audio processing libraries (automatically installed with requirements)

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/ai-audio-detector.git
   cd ai-audio-detector
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install -e .  # Install in development mode
   ```

4. **Install development dependencies**:
   ```bash
   pip install pytest pytest-cov flake8 black isort
   ```

### Project Structure

```
ai-audio-detector/
├── ai_audio_detector.py      # Main implementation
├── example_usage.py          # Usage examples
├── config.yaml              # Configuration file
├── requirements.txt          # Python dependencies
├── setup.py                 # Package setup
├── README.md                # Project documentation
├── CHANGELOG.md             # Version history
├── CONTRIBUTING.md          # This file
├── LICENSE                  # License information
├── .gitignore              # Git ignore rules
└── .github/                # GitHub workflows
    └── workflows/
        └── ci.yml          # CI/CD pipeline
```

## Coding Standards

### Python Style

- Follow **PEP 8** style guidelines
- Use **Black** for code formatting:
  ```bash
  black ai_audio_detector.py
  ```
- Use **isort** for import sorting:
  ```bash
  isort ai_audio_detector.py
  ```
- Use **flake8** for linting:
  ```bash
  flake8 ai_audio_detector.py
  ```

### Documentation

- Use **docstrings** for all public functions and classes
- Follow **Google style** docstrings
- Update **README.md** for user-facing changes
- Update **CHANGELOG.md** for all changes

### Code Quality

- **Error handling**: Use try-except blocks appropriately
- **Type hints**: Add type hints where beneficial
- **Comments**: Explain complex logic and algorithms
- **Modularity**: Keep functions focused and single-purpose
- **Configuration**: Use config.yaml for configurable parameters

## Testing

### Running Tests

```bash
# Run basic functionality tests
python -c "
from ai_audio_detector import AIAudioDetector
detector = AIAudioDetector()
print('Tests passed')
"

# Run linting
flake8 ai_audio_detector.py

# Test imports
python -c "from ai_audio_detector import AIAudioDetector, AudioFeatureExtractor"
```

### Test Coverage

- Test new features and bug fixes
- Ensure existing functionality still works
- Test edge cases and error conditions
- Test with different audio formats and configurations

### Audio Test Data

When contributing tests that require audio files:

- Use **small, synthetic** audio files when possible
- **Don't commit large audio files** to the repository
- Provide **instructions** for generating test data
- Use **publicly available** test datasets when appropriate

## Submitting Changes

### Commit Messages

Use clear, descriptive commit messages:

```
Add support for additional audio formats

- Added support for M4A and AAC formats
- Updated configuration to include new formats
- Added tests for new format support
- Updated documentation

Closes #123
```

### Pull Request Process

1. **Update documentation** if needed
2. **Add or update tests** for your changes
3. **Ensure all tests pass** locally
4. **Update CHANGELOG.md** with your changes
5. **Create a pull request** with:
   - Clear title and description
   - Reference to related issues
   - Summary of changes made
   - Testing performed

### Review Process

- All pull requests require **code review**
- **Automated tests** must pass
- **Documentation** must be updated if needed
- **Backward compatibility** should be maintained

## Feature Areas

### Priority Areas for Contribution

1. **Audio Format Support**
   - Additional audio formats
   - Better format detection
   - Improved audio loading

2. **Feature Extraction**
   - New audio features
   - Optimization of existing features
   - Feature selection methods

3. **Model Improvements**
   - New model architectures
   - Hyperparameter optimization
   - Model ensemble techniques

4. **Performance**
   - Processing speed optimization
   - Memory usage reduction
   - Parallel processing improvements

5. **Usability**
   - CLI improvements
   - Better error messages
   - Configuration enhancements

6. **Documentation**
   - Usage examples
   - API documentation
   - Performance benchmarks

## Questions?

If you have questions about contributing:

1. **Check existing issues** and documentation
2. **Create a new issue** for discussion
3. **Join the discussion** in existing issues

Thank you for contributing to AI Audio Detector!
