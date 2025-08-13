"""
AI Audio Detector using Benford's Law and librosa audio feature extraction.
Trains Random Forest, Gradient Boosting, SGD, and Passive Aggressive classifiers.
Returns highest confidence prediction with detailed feature analysis.
"""

__version__ = "1.1.0"

# Import main classes and functions
from .config import load_config
from .feature_extraction import AudioFeatureExtractor
from .audio_analyzer import AudioAnalyzer
from .detector import AIAudioDetector, process_single_audio_file, process_single_prediction
from .cli import main, run_interactive_mode

# For backward compatibility, expose main classes at package level
__all__ = [
    "AIAudioDetector",
    "AudioFeatureExtractor",
    "AudioAnalyzer",
    "load_config",
    "main",
    "run_interactive_mode",
    "process_single_audio_file",
    "process_single_prediction",
]
