# AIAA: AI Audio Authenticity

A machine learning research toolkit for exploring AI-generated audio detection using Benford's Law analysis and advanced audio feature extraction. The system employs ensemble learning with adaptive model updating capabilities.

> **Research status:** The Benford's Law signal used here is an experimental hypothesis, not a validated forensic guarantee. Treat predictions as model outputs that require dataset-specific evaluation before real-world use.

## Features

- **Multi-Model Ensemble**: Uses Random Forest, Gradient Boosting, SGD, and Passive Aggressive classifiers
- **Benford's Law Analysis**: Analyzes frequency distributions for AI detection patterns
- **Comprehensive Audio Features**: Extracts spectral, temporal, and compression-related features
- **Adaptive Learning**: Supports incremental model updates with new data
- **Batch Processing**: Parallel processing for large audio datasets
- **Spectrogram Generation**: Creates and compares various types of spectrograms
- **Interactive CLI**: User-friendly command-line interface

## Supported Audio Formats

- WAV (.wav)
- MP3 (.mp3)
- FLAC (.flac)
- OGG (.ogg)
- M4A (.m4a)
- AAC (.aac)

## Installation

### Option 1: Install from PyPI (Recommended)

```bash
pip install aiaa
```

### Option 2: Install from Source

1. Clone the repository:
```bash
git clone https://github.com/ajprice16/AI_Audio_Detection.git
cd AI_Audio_Detection
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### System Dependencies

On Ubuntu/Debian:
```bash
sudo apt-get install libsndfile1 ffmpeg
```

On macOS:
```bash
brew install libsndfile ffmpeg
```

## Quick Start

### Training Initial Models

1. **Prepare your data**: Organize your audio files into two directories:
   - `human_audio/` - Human-generated audio files
   - `ai_audio/` - AI-generated audio files

2. **Run the detector**:

**If installed from PyPI:**
```bash
aiaa --interactive
# or
aiaa --predict path/to/audio.wav
```

**If running from source:**
```bash
python -m aiaa --interactive
# or
python -m aiaa --predict path/to/audio.wav
```

3. In interactive mode, use `train <ai_dir> <human_dir>` to train new models.

### Command Line Usage

**Train models:**
```bash
aiaa --train --human-dir path/to/human/audio --ai-dir path/to/ai/audio
```

**Predict single file:**
```bash
aiaa --predict path/to/audio.wav
```

**Predict batch:**
```bash
aiaa --batch path/to/audio/directory
```

**Interactive mode:**
```bash
aiaa --interactive
```

### Predicting Single Files

**Interactive mode:**
```bash
aiaa --interactive
# Then run: predict path/to/audio.wav
```

**Direct command:**
```bash
aiaa --predict path/to/audio.wav
```

### Batch Prediction

**Interactive mode:**
```bash
aiaa --interactive
# Then run: batch path/to/audio/directory
```

**Direct command:**
```bash
aiaa --batch path/to/audio/directory
```

## Advanced Usage

### Programmatic Usage

```python
from aiaa import AIAudioDetector
from pathlib import Path
import pandas as pd

# Initialize detector
detector = AIAudioDetector(base_dir=Path.cwd())

# Train models
human_features = detector.extract_features_from_directory("human_audio/", is_ai_directory=False)
ai_features = detector.extract_features_from_directory("ai_audio/", is_ai_directory=True)

all_features = human_features + ai_features
df_results = pd.DataFrame(all_features)
training_results = detector.train_models(df_results)

# Make predictions
result = detector.predict_single_file("test_audio.wav")
print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.3f}")
```

### Adaptive Learning

The system supports adaptive learning to improve accuracy with new data:

```python
new_ai_features = detector.extract_features_from_directory("new_ai_audio/", is_ai_directory=True)
new_human_features = detector.extract_features_from_directory("new_human_audio/", is_ai_directory=False)
update_results = detector.update_with_new_data(new_ai_features + new_human_features)
```

## Features Extracted

### Benford's Law Features
- Chi-square test statistics
- Kolmogorov-Smirnov test statistics
- Mean absolute deviation from expected distribution
- Maximum deviation
- Entropy measures

### Spectral Features
- Spectral centroid, bandwidth, rolloff
- MFCCs (13 coefficients + standard deviations)
- Chroma features
- Spectral contrast
- Zero crossing rate

### Temporal Features
- RMS energy (mean and standard deviation)
- Tempo estimation
- Spectral flatness
- Dynamic range
- Peak-to-RMS ratio

### Compression Features
- Estimated bit depth
- Clipping detection
- DC offset
- High frequency content ratio

## Model Architecture

The system uses an ensemble of four different models:

1. **Incremental Models** (for adaptive learning):
   - SGD Classifier with log loss
   - Passive Aggressive Classifier

2. **Batch Models** (for maximum accuracy):
   - Random Forest (200 estimators)
   - Gradient Boosting (200 estimators)

All features are standardized using StandardScaler, and final predictions use a confidence-weighted ensemble vote.

## Configuration

Modify `config.yaml` to customize:
- Model parameters
- Feature extraction settings
- Processing options
- Output directories

## Command Line Options

- `--train --ai-dir <dir> --human-dir <dir>` - Initial training from audio directories
- `--predict <file>` - Analyze one audio file
- `--batch <dir>` - Analyze all supported audio files in a directory
- `--update --ai-dir <dir> --human-dir <dir>` - Incrementally update trained models
- `--interactive` - Run a multi-command session
- `--spectrogram <file>` - Generate a mel spectrogram
- `--compare <file1> <file2>` - Generate a side-by-side spectrogram comparison

## Output Files

- `models/aiaa.joblib` - Trained models and metadata
- `results/training_results.csv` - Training metrics for fitted models
- `spectrograms/` - Generated spectrogram images
- `spectrogram_comparisons/` - Side-by-side comparisons

## Performance Considerations

- **Multiprocessing**: Automatically used for batches > 3 files
- **Memory Management**: Spectrograms are generated efficiently with proper cleanup
- **Scalability**: Incremental learning allows handling large datasets over time

## Requirements

- Python 3.8+
- librosa (audio processing)
- scikit-learn (machine learning)
- pandas, numpy (data manipulation)
- matplotlib (visualization)
- scipy (statistical tests)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Uses Benford's Law for detecting artificial patterns in audio
- Built on librosa for robust audio feature extraction
- Employs scikit-learn for machine learning capabilities

## Citation

If you use this work in your research, please cite:

```bibtex
@software{aiaa,
  title={AIAA: AI Audio Authenticity - Machine Learning System for Detecting AI-Generated Audio},
  author={Alex Price},
  year={2025},
  url={https://github.com/ajprice16/AI_Audio_Detection}
}
```
