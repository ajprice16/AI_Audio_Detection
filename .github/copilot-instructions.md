# AI Audio Detector - Copilot Instructions

This is a machine learning system for detecting AI-generated audio using Benford's Law analysis and advanced audio feature extraction.

## Architecture Overview

The system uses a **dual-model ensemble** approach:
- **Incremental models** (SGD, PassiveAggressive) for adaptive learning with new data
- **Batch models** (RandomForest, GradientBoosting) retrained from scratch for maximum accuracy
- Final predictions use **ensemble averaging** across all models

### Core Components

1. **`AIAudioDetector`** (`detector.py`): Main orchestrator class handling training, prediction, and model persistence
2. **`AudioAnalyzer`** (`audio_analyzer.py`): Coordinates feature extraction and spectrogram generation
3. **`AudioFeatureExtractor`** (`feature_extraction.py`): Implements 4 feature categories:
   - **Benford's Law features** (chi-square, KS tests, entropy, MAD, max deviation)
   - **Spectral features** (MFCCs, centroid, bandwidth, chroma, contrast)
   - **Temporal features** (RMS, ZCR, tempo, dynamic range, spectral flatness)
   - **Compression features** (bit depth estimation, clipping ratio, DC offset, high-freq ratio)

### Data Flow

```
WAV/Audio Files → librosa.load() → AudioAnalyzer → AudioFeatureExtractor →
Feature DataFrame → StandardScaler → Ensemble Models → Averaged Prediction
                 ↓
            STFT Analysis → Peak Frequencies → Benford's Law Analysis
```

### Benford's Law: Experimental Hypothesis

**CRITICAL**: This project tests whether Benford's Law can distinguish AI-generated audio from human audio. This is **experimental research** - the hypothesis is unproven.

**Implementation details** (`audio_analyzer.py:67-86`, `feature_extraction.py:15-103`):
1. Extract STFT magnitudes from audio signal: `librosa.stft(y)`
2. For each frame, identify peaks above `mean + std` threshold
3. Convert peak indices to Hz frequencies using `librosa.fft_frequencies()`
4. Analyze **first-digit distribution** of peak frequencies (1-9)
5. Compare observed vs expected Benford distribution:
   - **Chi-square test**: Tests goodness-of-fit (`chi2_p`, `chi2_stat`)
   - **KS test**: Compares empirical vs expected distributions (`ks_p`, `ks_stat`)
   - **MAD**: Mean absolute deviation from expected
   - **Entropy**: Information entropy of digit distribution
   - **Max deviation**: Largest single deviation from Benford's Law

**Why it might work**: AI audio generators may produce mathematically regular frequency patterns that deviate from natural Benford distributions found in human-created audio.

## Development Workflows

### Testing & Benchmarks

```bash
# Run full test suite with coverage
pytest --cov=ai_audio_detector --cov-report=html

# Run benchmarks (creates synthetic test data)
./run_benchmark.sh  # Requires .venv/bin/python

# Run specific test categories
pytest -m "not slow"  # Skip slow tests
pytest -m integration  # Integration tests only
```

### Package Management

The project uses **dual packaging**:
- `setup.py` for development installs: `pip install -e .`
- `pyproject.toml` for distribution: `pip install ai-audio-detector`

**Key requirement**: `librosa` for audio processing, `scikit-learn` for ML models

### CLI Usage Patterns

The CLI supports both **interactive mode** and **direct commands**:
```bash
ai-audio-detector --interactive  # Multi-command session
ai-audio-detector --predict-file audio.wav  # Single prediction
ai-audio-detector --train --human-dir path1 --ai-dir path2  # Training

# Spectrogram generation (saved to spectrograms/)
ai-audio-detector --spectrogram audio.wav

# Compare two spectrograms (saved to spectrogram_comparisons/)
ai-audio-detector --compare human.wav ai_generated.wav
```

**Interactive mode commands**:
- `predict <file>` - Single file prediction
- `batch <dir>` - Batch prediction
- `spectrogram <file>` - Generate mel-spectrogram visualization
- `compare <file1> <file2>` - Side-by-side spectrogram comparison
- `train <ai_dir> <human_dir>` - Initial training
- `update <ai_dir> <human_dir>` - Adaptive learning update
- `status` - Show model status and feature count

## Project-Specific Conventions

### Spectral Analysis & Feature Extraction

**Audio Loading** (`audio_analyzer.py:113-119`):
```python
y, sr = librosa.load(file_path, sr=None)  # Preserves original sample rate
```

**Feature Extraction Pipeline** (`audio_analyzer.py:50-90`):
1. **Spectral features**: STFT-based (MFCCs with mean+std, chroma, contrast, centroid, bandwidth)
2. **Temporal features**: Time-domain (RMS energy, ZCR, tempo via beat tracking)
3. **Compression features**: Quantization analysis (bit depth, clipping, DC offset)
4. **Benford features**: STFT peaks → frequency values → first-digit distribution analysis

**STFT Peak Detection for Benford's Law** (`audio_analyzer.py:67-78`):
```python
stft = librosa.stft(y)
for frame in magnitudes.T:
    peaks = np.where(frame > np.mean(frame) + np.std(frame))[0]
    freq_hz = librosa.fft_frequencies(sr=sr, n_fft=len(frame)*2-1)
    frequencies.extend(freq_hz[peaks])  # Peak frequencies for Benford analysis
```

### Spectrogram & Visualization Workflow

**Spectrogram Generation** (`audio_analyzer.py:94-142`, `detector.py:613-621`):
- **Type**: Mel-scaled spectrogram (128 mel bins)
- **Transform**: `melspectrogram → power_to_db` (dB scale relative to max)
- **Frequency range**: 0-8000 Hz (`fmax=8000`)
- **Output**: PNG saved to `spectrograms/` directory (300 DPI by default)
- **Configuration**: `config.yaml:visualization` controls figsize, DPI, colorbar

**Comparison Workflow** (`audio_analyzer.py:144-199`, `detector.py:623-638`):
- **Layout**: 2-row subplot (file1 top, file2 bottom)
- **Synchronization**: Both use same mel-scaling and frequency range for valid comparison
- **Output**: PNG saved to `spectrogram_comparisons/` directory
- **Use case**: Visual comparison of AI vs human audio spectral characteristics

**Key differences to observe**:
- AI audio may show more uniform spectral energy distribution
- Human audio typically has more irregular harmonic structures
- Compression artifacts visible in high-frequency content

### Feature Engineering

- **All features are normalized** using `StandardScaler` before training
- **Benford's Law analysis** extracts first-digit distributions from frequency data
- **MFCC features** include both mean and standard deviation (26 total features)
- **Missing audio files are skipped** rather than causing failures

### Model Persistence

Models are saved as a **single joblib file** containing:
```python
{
    'models': {model_name: model_object},
    'scaler': StandardScaler(),
    'feature_columns': List[str],
    'training_history': List[Dict],
    'is_trained': bool
}
```

### Multiprocessing Strategy

- **Batch threshold**: 3+ files trigger multiprocessing (`config.yaml`)
- **ProcessPoolExecutor** used for feature extraction (CPU-bound)
- **Max workers**: Configurable via `processing.max_workers`

### Error Handling Patterns

```python
# Feature extraction returns empty dict on failure
features = extract_spectral_features(y, sr)  # Returns {} if error
if not features:
    # Skip this file, continue processing

# Prediction returns error dict rather than raising
result = detector.predict_file("missing.wav")  # {'error': 'File not found'}
```

## Configuration System

`config.yaml` drives all behavior:
- **Model hyperparameters**: `models.batch.random_forest.n_estimators`
- **Audio formats**: `audio.supported_formats` (used by `rglob` patterns)
- **Processing**: `processing.batch_threshold` controls multiprocessing trigger
- **Paths**: All output directories configurable under `output.*`
- **Visualization**: `visualization.figsize`, `dpi`, `colorbar` control spectrogram appearance
- **Feature extraction**: `features.spectral.n_mfcc`, `n_mels` for spectral resolution

## Integration Points

### External Dependencies

- **librosa**: Audio loading and feature extraction (core dependency)
  - `librosa.load()` - Audio file loading with sample rate preservation
  - `librosa.stft()` - Short-Time Fourier Transform for spectral analysis
  - `librosa.feature.melspectrogram()` - Mel-scaled spectrograms
  - `librosa.feature.mfcc()` - Mel-frequency cepstral coefficients
  - `librosa.display.specshow()` - Spectrogram visualization
- **scikit-learn**: All ML models and preprocessing
- **matplotlib**: Spectrogram generation and visualization
- **scipy.stats**: Statistical tests for Benford's Law (chi-square, KS test)
- **joblib**: Model serialization (faster than pickle for numpy arrays)

### File System Patterns

```
project_root/
├── models/ai_audio_detector.joblib  # Single model file
├── results/training_results.csv     # Feature extraction results
├── spectrograms/*.png              # Generated spectrograms
└── spectrogram_comparisons/*.png   # Side-by-side comparisons
```

### Entry Points

The package provides both module (`python -m ai_audio_detector`) and console script (`ai-audio-detector`) entry points that call `cli.main()`.

## Common Debugging Scenarios

1. **"No audio files found"**: Check `get_audio_extensions()` matches your file types
2. **"Models not trained"**: Call `train_models()` before `predict_*()` methods
3. **Multiprocessing hangs**: Reduce `processing.max_workers` or `batch_threshold`
4. **Feature extraction fails**: Check librosa version compatibility and audio file integrity
5. **Benford features missing**: Ensure audio has sufficient peak frequencies (10+ required)
6. **Spectrogram generation fails**: Verify `spectrograms/` directory exists and has write permissions
7. **Empty Benford features**: Audio signal may be too quiet or have insufficient spectral peaks

## Key Files for Understanding

- `detector.py:150-200` - Multiprocessing implementation for batch feature extraction
- `feature_extraction.py:15-103` - Benford's Law calculation and statistical tests
- `audio_analyzer.py:50-90` - Feature extraction coordination and STFT peak detection
- `audio_analyzer.py:94-142` - Mel-spectrogram generation with librosa
- `audio_analyzer.py:144-199` - Side-by-side spectrogram comparison
- `config.yaml` - All configurable behavior (models, features, visualization)
- `tests/benchmark.py` - Performance testing patterns and synthetic audio generation
