# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.2.2] - 2026-02-19

### Added
- Centralized logging system with `logging_config.py` module
  - `setup_logging()` for configurable logging with level/file support
  - `get_logger()` for module-level logger creation
  - Rotating file handler (10MB max, 5 backups) for persistent logs
  - Support for DEBUG/INFO/WARNING/ERROR/CRITICAL levels
  - CLI `--verbose` flag enables DEBUG level output
- Audio file validation utilities in `audio_validation.py`
  - `validate_audio_file()` pre-validates before expensive processing
  - `validate_batch_audio_files()` validates entire directories
  - `get_audio_sample_rate()` safe sample rate detection
  - AudioValidationError exception class
- Configuration validation with detailed logging
  - `validate_config()` verifies required config structure
  - Logs missing keys with helpful context

### Changed
- Replaced 30+ print() statements with structured logging throughout codebase
- Enhanced docstrings with complete return value documentation
- Type hints upgraded (e.g., `List[Path]` instead of `list`, `List[str]` for formats)
- Detector logs training progress, model persistence, and predictions
- Audio analyzer logs validation and feature extraction details
- Configuration loading now validates structure and warns about issues

### Fixed
- Removed unused `validate_audio_file` import from detector.py
- Corrected `validate_batch_audio_files()` return type to `Tuple[List[Path], List[Tuple[Optional[Path], str]]]`
- Added missing `List` import to audio_validation.py for proper typing
- Graceful error handling for invalid audio files (returns error dict vs crashing)
- Configuration errors now logged with specific details

### Improved
- Production-ready logging with persistent rotating file handlers
- Early detection of audio issues before expensive feature extraction
- Better debugging with timestamps, function names, line numbers
- Batch processing gracefully skips invalid files instead of failing completely
- CLI provides detailed feedback with DEBUG mode support

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
