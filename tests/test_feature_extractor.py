"""
Unit tests for AudioFeatureExtractor class
"""

import unittest
import numpy as np
import sys
from pathlib import Path

# Add the parent directory to path to import the module
sys.path.insert(0, str(Path(__file__).parent.parent))

from ai_audio_detector import AudioFeatureExtractor


def is_numeric(value):
    """Helper function to check if value is numeric (int, float, or numpy numeric)"""
    return isinstance(value, (int, float, np.integer, np.floating))


class TestAudioFeatureExtractor(unittest.TestCase):
    """Test cases for AudioFeatureExtractor"""

    def setUp(self):
        """Set up test fixtures"""
        self.extractor = AudioFeatureExtractor()

        # Create sample data for testing
        self.sample_frequencies = [
            1.23,
            4.56,
            7.89,
            2.34,
            5.67,
            8.90,
            3.45,
            6.78,
            9.01,
            1.11,
        ]
        self.empty_frequencies = []
        self.small_frequencies = [1.0, 2.0]

        # Create sample audio data (sine wave)
        self.sample_rate = 22050
        duration = 1.0  # 1 second
        t = np.linspace(0, duration, int(self.sample_rate * duration))
        self.sample_audio = np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave

    def test_extract_benford_features_valid_data(self):
        """Test Benford's Law feature extraction with valid data"""
        features = self.extractor.extract_benford_features(self.sample_frequencies)

        # Check that all expected features are present
        expected_features = [
            "chi2_p",
            "chi2_stat",
            "ks_p",
            "ks_stat",
            "mad",
            "max_deviation",
            "entropy",
        ]
        for feature in expected_features:
            self.assertIn(feature, features)
            self.assertTrue(is_numeric(features[feature]))
            self.assertFalse(np.isnan(features[feature]))

    def test_extract_benford_features_empty_data(self):
        """Test Benford's Law feature extraction with empty data"""
        features = self.extractor.extract_benford_features(self.empty_frequencies)
        self.assertEqual(features, {})

    def test_extract_benford_features_insufficient_data(self):
        """Test Benford's Law feature extraction with insufficient data"""
        features = self.extractor.extract_benford_features(self.small_frequencies)
        self.assertEqual(features, {})

    def test_extract_spectral_features(self):
        """Test spectral feature extraction"""
        features = self.extractor.extract_spectral_features(
            self.sample_audio, self.sample_rate
        )

        # Check that basic spectral features are present
        expected_features = [
            "spectral_centroid",
            "spectral_bandwidth",
            "spectral_rolloff",
            "zero_crossing_rate",
            "chroma_mean",
            "chroma_std",
            "spectral_contrast",
        ]

        for feature in expected_features:
            self.assertIn(feature, features)
            self.assertTrue(is_numeric(features[feature]))
            self.assertFalse(np.isnan(features[feature]))

        # Check MFCC features
        for i in range(13):
            self.assertIn(f"mfcc_{i}", features)
            self.assertIn(f"mfcc_{i}_std", features)

    def test_extract_temporal_features(self):
        """Test temporal feature extraction"""
        features = self.extractor.extract_temporal_features(
            self.sample_audio, self.sample_rate
        )

        expected_features = [
            "rms_mean",
            "rms_std",
            "tempo",
            "spectral_flatness",
            "dynamic_range",
            "peak_to_rms",
        ]

        for feature in expected_features:
            self.assertIn(feature, features)
            self.assertTrue(is_numeric(features[feature]))
            self.assertFalse(np.isnan(features[feature]))

    def test_extract_compression_features(self):
        """Test compression feature extraction"""
        features = self.extractor.extract_compression_features(
            self.sample_audio, self.sample_rate
        )

        expected_features = [
            "estimated_bit_depth",
            "clipping_ratio",
            "dc_offset",
            "high_freq_ratio",
        ]

        for feature in expected_features:
            self.assertIn(feature, features)
            self.assertTrue(is_numeric(features[feature]))
            self.assertFalse(np.isnan(features[feature]))

    def test_feature_extraction_error_handling(self):
        """Test that feature extraction handles errors gracefully"""
        # Test with invalid audio data
        invalid_audio = np.array([])

        spectral_features = self.extractor.extract_spectral_features(
            invalid_audio, self.sample_rate
        )
        temporal_features = self.extractor.extract_temporal_features(
            invalid_audio, self.sample_rate
        )
        compression_features = self.extractor.extract_compression_features(
            invalid_audio, self.sample_rate
        )

        # Should return empty dict or handle gracefully
        self.assertIsInstance(spectral_features, dict)
        self.assertIsInstance(temporal_features, dict)
        self.assertIsInstance(compression_features, dict)


if __name__ == "__main__":
    unittest.main()
