# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.2.1] - 2025-10-17

### Added
- Comprehensive Copilot instructions for AI coding agents
- JOSS compliance verification documentation
- CI/CD optimization documentation

### Changed
- Optimized CI/CD pipeline with enhanced pip caching (30-40% faster)
- Updated pre-commit hooks to check aiaa/ directory only
- Removed PyPI publishing job (package available as ai-audio-detector on PyPI)

### Fixed
- Type annotation for frequencies list in audio_analyzer.py
- Pre-commit hooks now correctly target aiaa/ instead of old ai_audio_detector/
- CI dependency installation now includes editable package install for type checking

### Removed
- Old ai_audio_detector/ workspace directory
- PyPI publishing workflow (to avoid naming conflicts)

## [1.1.0] - 2025-8-1

### Changed
- Rebranded package from "ai-audio-detector" to "aiaa" (AIAA: AI Audio Authenticity)
- Updated all documentation, examples, and code references to reflect new branding
- Changed console command from `ai-audio-detector` to `aiaa`
- Updated model filename from `ai_audio_detector.joblib` to `aiaa.joblib`

### Added
- JOSS Journal submission compliance and information

## [1.0.0] - 2025-07-07

### Added
- Initial release of AI Audio Detector
- Multi-model ensemble learning (Random Forest, Gradient Boosting, SGD, Passive Aggressive)
- Benford's Law analysis for AI detection
- Comprehensive audio feature extraction (spectral, temporal, compression)
- Adaptive learning capabilities with incremental model updates
- Batch processing with multiprocessing support
- Interactive command-line interface
- Programmatic API for integration
- Spectrogram generation and comparison tools
- Configuration file support (YAML)
- Command-line argument support
- Training history and data balance analysis
- Visualization tools for analysis results

### Features
- **Audio Format Support**: WAV, MP3, FLAC, OGG, M4A, AAC
- **Feature Extraction**:
  - Benford's Law statistics (Chi-square, KS test, MAD, entropy)
  - Spectral features (centroid, bandwidth, rolloff, MFCCs, chroma, contrast)
  - Temporal features (RMS, tempo, flatness, dynamic range)
  - Compression features (bit depth, clipping, DC offset, high-freq content)
- **Model Architecture**: Ensemble of 4 different classifiers with feature standardization
- **Processing**: Automatic multiprocessing for large datasets
- **Adaptability**: Incremental learning without full retraining
- **Visualization**: Training analysis plots and spectrogram comparisons
- **Configuration**: Flexible YAML-based configuration system

### Technical Details
- Python 3.7+ compatibility
- Robust error handling and logging
- Memory-efficient processing with proper cleanup
- Cross-platform support (Windows, macOS, Linux)
- Configurable output directories and processing parameters
