"""
Unit tests for AudioFeatureExtractor class
"""

import unittest
import numpy as np
from pathlib import Path
from typing import Any, Union, List

from ai_audio_detector import AudioFeatureExtractor


def is_numeric(value: Any) -> bool:
    """Helper function to check if value is numeric (int, float, or numpy numeric)"""
    return isinstance(value, (int, float, np.integer, np.floating))


class TestAudioFeatureExtractor(unittest.TestCase):
    """Test cases for AudioFeatureExtractor"""

    def setUp(self) -> None:
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
        self.empty_frequencies: List[float] = []
        self.small_frequencies = [1.0, 2.0]

        # Create sample audio data (sine wave)
        self.sample_rate = 22050
        duration = 1.0  # 1 second
        t = np.linspace(0, duration, int(self.sample_rate * duration))
        self.sample_audio = np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave

    def test_extract_benford_features_valid_data(self) -> None:
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

    def test_extract_benford_features_empty_data(self) -> None:
        """Test Benford's Law feature extraction with empty data"""
        features = self.extractor.extract_benford_features(self.empty_frequencies)
        self.assertEqual(features, {})

    def test_extract_benford_features_insufficient_data(self) -> None:
        """Test Benford's Law feature extraction with insufficient data"""
        features = self.extractor.extract_benford_features(self.small_frequencies)
        self.assertEqual(features, {})

    def test_extract_spectral_features(self) -> None:
        """Test spectral feature extraction"""
        features = self.extractor.extract_spectral_features(self.sample_audio, self.sample_rate)

        # Check that basic spectral features are present
        expected_features = [
            "spectral_centroid",
            "spectral_bandwidth",
            "spectral_rolloff",
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

    def test_extract_temporal_features(self) -> None:
        """Test temporal feature extraction"""
        features = self.extractor.extract_temporal_features(self.sample_audio, self.sample_rate)

        expected_features = [
            "temporal_rms_mean",
            "temporal_rms_std",
            "temporal_zcr_mean",
            "temporal_zcr_std",
            "temporal_tempo",
            "temporal_spectral_flatness",
            "temporal_dynamic_range",
        ]

        for feature in expected_features:
            self.assertIn(feature, features)
            self.assertTrue(is_numeric(features[feature]))
            self.assertFalse(np.isnan(features[feature]))

    def test_extract_compression_features(self) -> None:
        """Test compression feature extraction"""
        features = self.extractor.extract_compression_features(self.sample_audio, self.sample_rate)

        expected_features = [
            "compression_estimated_bit_depth",
            "compression_clipping_ratio",
            "compression_dc_offset",
            "compression_high_freq_ratio",
        ]

        for feature in expected_features:
            self.assertIn(feature, features)
            self.assertTrue(is_numeric(features[feature]))
            self.assertFalse(np.isnan(features[feature]))

    def test_feature_extraction_error_handling(self) -> None:
        """Test that feature extraction handles errors gracefully"""
        # Test with invalid audio data
        invalid_audio = np.array([])

        spectral_features = self.extractor.extract_spectral_features(invalid_audio, self.sample_rate)
        temporal_features = self.extractor.extract_temporal_features(invalid_audio, self.sample_rate)
        compression_features = self.extractor.extract_compression_features(invalid_audio, self.sample_rate)

        # Should return empty dict or handle gracefully
        self.assertIsInstance(spectral_features, dict)
        self.assertIsInstance(temporal_features, dict)
        self.assertIsInstance(compression_features, dict)


if __name__ == "__main__":
    unittest.main()
