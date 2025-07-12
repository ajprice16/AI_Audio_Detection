# AI Audio Detector

A machine learning system for detecting AI-generated audio using Benford's Law analysis and advanced audio feature extraction. The system employs ensemble learning with adaptive model updating capabilities.

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
pip install ai-audio-detector
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
ai-audio-detector --interactive
# or
ai-audio-detector --predict-file path/to/audio.wav
```

**If running from source:**
```bash
python -m ai_audio_detector --interactive
# or
python -m ai_audio_detector --predict-file path/to/audio.wav
```

3. **Choose option 1** to train new models and follow the prompts.

### Command Line Usage

**Train models:**
```bash
ai-audio-detector --train --human-dir path/to/human/audio --ai-dir path/to/ai/audio
```

**Predict single file:**
```bash
ai-audio-detector --predict-file path/to/audio.wav
```

**Predict batch:**
```bash
ai-audio-detector --predict-batch path/to/audio/directory
```

**Interactive mode:**
```bash
ai-audio-detector --interactive
```

### Predicting Single Files

**Interactive mode:**
```bash
ai-audio-detector --interactive
# Choose option 2 and enter the path to your audio file
```

**Direct command:**
```bash
ai-audio-detector --predict-file path/to/audio.wav
```

### Batch Prediction

**Interactive mode:**
```bash
ai-audio-detector --interactive
# Choose option 3 and enter the directory path
```

**Direct command:**
```bash
ai-audio-detector --predict-batch path/to/audio/directory
```

## Advanced Usage

### Programmatic Usage

```python
from ai_audio_detector import AIAudioDetector
from pathlib import Path

# Initialize detector
detector = AIAudioDetector(base_dir=Path.cwd())

# Train models
human_features = detector.extract_features_from_directory("human_audio/", is_ai_directory=False)
ai_features = detector.extract_features_from_directory("ai_audio/", is_ai_directory=True)

all_features = human_features + ai_features
df_results = pd.DataFrame(all_features)
training_results = detector.train_models(df_results)

# Make predictions
result = detector.predict_file("test_audio.wav")
print(f"Prediction: {'AI' if result['is_ai'] else 'Human'}")
print(f"Confidence: {result['confidence']:.3f}")
```

### Adaptive Learning

The system supports adaptive learning to improve accuracy with new data:

```python
# Add new AI data
detector.add_ai_data("new_ai_audio/", retrain_batch_models=True)

# Add new human data
detector.add_human_data("new_human_audio/", retrain_batch_models=True)

# Add mixed data batch
directories = [
    {'path': 'dataset1/', 'is_ai': True},
    {'path': 'dataset2/', 'is_ai': False}
]
detector.add_mixed_data_batch(directories, retrain_batch_models=True)
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

All features are standardized using StandardScaler, and final predictions use ensemble averaging.

## Configuration

Modify `config.yaml` to customize:
- Model parameters
- Feature extraction settings
- Processing options
- Output directories

## Command Line Options

1. **Train new models** - Initial training from audio directories
2. **Predict single file** - Analyze one audio file
3. **Predict batch** - Analyze all files in a directory
4. **Update models** - Adaptive learning with new data
5. **Add AI data** - Add new AI samples to existing models
6. **Add Human data** - Add new human samples to existing models
7. **Batch directories** - Add multiple directories at once
8. **Training history** - View model training history
9. **Data balance** - Check AI vs Human data balance
10. **Create visualizations** - Generate analysis plots
11. **Generate spectrograms** - Create spectrograms for audio files
12. **Spectrogram comparison** - Compare AI vs Human spectrograms

## Output Files

- `models/ai_audio_detector.joblib` - Trained models and metadata
- `training_results.csv` - Detailed training data and features
- `ai_detection_analysis.png` - Visualization plots
- `spectrograms/` - Generated spectrogram images
- `spectrogram_comparisons/` - Side-by-side comparisons

## Performance Considerations

- **Multiprocessing**: Automatically used for batches > 3 files
- **Memory Management**: Spectrograms are generated efficiently with proper cleanup
- **Scalability**: Incremental learning allows handling large datasets over time

## Requirements

- Python 3.7+
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
@software{ai_audio_detector,
  title={AI Audio Detector: Machine Learning System for Detecting AI-Generated Audio},
  author={Alex Price},
  year={2025},
  url={https://github.com/yourusername/ai-audio-detector}
}
```
