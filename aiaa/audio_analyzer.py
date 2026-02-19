"""
Audio analysis utilities for AIAA: AI Audio Authenticity.
"""

from typing import Dict, Any, Optional, Tuple, Union
import numpy as np
import librosa
from pathlib import Path
import matplotlib.pyplot as plt
import librosa.display
import logging

from .feature_extraction import AudioFeatureExtractor
from .config import load_config
from .audio_validation import validate_audio_file

# Load configuration
config = load_config()
logger = logging.getLogger(__name__)


class AudioAnalyzer:
    """Main audio analysis class."""

    def __init__(self) -> None:
        """Initialize the audio analyzer."""
        self.extractor = AudioFeatureExtractor()

    def analyze_audio_file(self, file_path: Union[str, Path], include_benford: bool = True) -> Dict[str, Any]:
        """
        Analyze an audio file and extract comprehensive features.

        Args:
            file_path: Path to the audio file
            include_benford: Whether to include Benford's Law analysis

        Returns:
            Dictionary containing extracted features
        """

        # Ensure file_path is a Path object
        file_path = Path(file_path)

        try:
            # Validate audio file first
            is_valid, message = validate_audio_file(file_path)
            if not is_valid:
                logger.warning(f"Audio validation failed for {file_path}: {message}")
                return {
                    "filename": file_path.name,
                    "full_path": str(file_path),
                    "error": message,
                }

            # Load audio
            y, sr = librosa.load(file_path, sr=config.get("audio", {}).get("default_sample_rate"))
            logger.debug(f"Loaded audio: {file_path} (sr={sr}, duration={len(y)/sr:.2f}s)")

            # Initialize features dictionary
            features = {
                "filename": file_path.name,
                "full_path": str(file_path),
                "duration": len(y) / sr,
                "sample_rate": sr,
            }

            # Extract spectral features
            spectral_features = self.extractor.extract_spectral_features(y, int(sr))
            features.update(spectral_features)

            # Extract temporal features
            temporal_features = self.extractor.extract_temporal_features(y, int(sr))
            features.update(temporal_features)

            # Extract compression features
            compression_features = self.extractor.extract_compression_features(y, int(sr))
            features.update(compression_features)

            # Extract Benford's Law features if requested
            if include_benford:
                # Use spectral centroid frequencies for Benford analysis
                stft = librosa.stft(y)
                magnitudes = np.abs(stft)
                frequencies: list[float] = []

                for frame in magnitudes.T:
                    if np.sum(frame) > 0:
                        # Get peak frequencies in this frame
                        peaks = np.where(frame > np.mean(frame) + np.std(frame))[0]
                        if len(peaks) > 0:
                            freq_hz = librosa.fft_frequencies(sr=sr, n_fft=len(frame) * 2 - 1)
                            frequencies.extend(freq_hz[peaks])

                benford_features = self.extractor.extract_benford_features(frequencies)
                features.update(benford_features)

            return features

        except Exception as e:
            logger.error(f"Error analyzing {file_path}: {e}", exc_info=True)
            return {
                "filename": file_path.name,
                "full_path": str(file_path),
                "error": str(e),
            }

    def generate_spectrogram(self, file_path: Path, output_path: Path, config: Optional[Dict[str, Any]] = None) -> bool:
        """
        Generate and save a spectrogram for an audio file.

        Args:
            file_path: Path to the audio file.
            output_path: Path to save the spectrogram.
            config: Configuration dictionary.

        Returns:
            True if successful, False otherwise.
        """
        config = config or {}
        viz_config = config.get("visualization", {})

        try:
            # Load audio
            y, sr = librosa.load(file_path, sr=config.get("audio", {}).get("default_sample_rate"))

            # Create spectrogram
            plt.figure(figsize=viz_config.get("figsize", [12, 8]))

            # Compute mel-scaled spectrogram
            S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
            S_dB = librosa.power_to_db(S, ref=np.max)

            # Plot
            librosa.display.specshow(S_dB, x_axis="time", y_axis="mel", sr=sr, fmax=8000)

            if viz_config.get("colorbar", True):
                plt.colorbar(format="%+2.0f dB")

            plt.title(f"Mel-frequency spectrogram - {file_path.stem}")
            plt.tight_layout()

            # Save
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=viz_config.get("dpi", 300), bbox_inches="tight")
            plt.close()

            return True

        except Exception as e:
            logger.error(f"Error generating spectrogram for {file_path}: {e}", exc_info=True)
            plt.close()  # Ensure plot is closed even on error
            return False

    def compare_spectrograms(
        self, file1_path: Path, file2_path: Path, output_path: Path, config: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Generate a comparison of two spectrograms.

        Args:
            file1_path: Path to the first audio file.
            file2_path: Path to the second audio file.
            output_path: Path to save the comparison image.
            config: Configuration dictionary.

        Returns:
            True if successful, False otherwise.
        """
        config = config or {}
        viz_config = config.get("visualization", {})

        try:
            # Load both audio files
            y1, sr1 = librosa.load(file1_path, sr=config.get("audio", {}).get("default_sample_rate"))
            y2, sr2 = librosa.load(file2_path, sr=config.get("audio", {}).get("default_sample_rate"))

            # Create figure with subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=viz_config.get("figsize", [12, 8]))

            # First spectrogram
            S1 = librosa.feature.melspectrogram(y=y1, sr=sr1, n_mels=128)
            S1_dB = librosa.power_to_db(S1, ref=np.max)

            img1 = librosa.display.specshow(S1_dB, x_axis="time", y_axis="mel", sr=sr1, fmax=8000, ax=ax1)
            ax1.set_title(f"{file1_path.stem}")

            if viz_config.get("colorbar", True):
                plt.colorbar(img1, ax=ax1, format="%+2.0f dB")

            # Second spectrogram
            S2 = librosa.feature.melspectrogram(y=y2, sr=sr2, n_mels=128)
            S2_dB = librosa.power_to_db(S2, ref=np.max)

            img2 = librosa.display.specshow(S2_dB, x_axis="time", y_axis="mel", sr=sr2, fmax=8000, ax=ax2)
            ax2.set_title(f"{file2_path.stem}")

            if viz_config.get("colorbar", True):
                plt.colorbar(img2, ax=ax2, format="%+2.0f dB")

            plt.tight_layout()

            # Save
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=viz_config.get("dpi", 300), bbox_inches="tight")
            plt.close()

            return True

        except Exception as e:
            logger.error(f"Error comparing spectrograms: {e}", exc_info=True)
            plt.close()  # Ensure plot is closed even on error
            return False
