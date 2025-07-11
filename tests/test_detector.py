"""
Unit tests for AIAudioDetector class
"""

import unittest
import tempfile
import shutil
import numpy as np
import pandas as pd
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add the parent directory to path to import the module
sys.path.insert(0, str(Path(__file__).parent.parent))

from ai_audio_detector import AIAudioDetector, load_config


class TestAIAudioDetector(unittest.TestCase):
    """Test cases for AIAudioDetector"""

    def setUp(self):
        """Set up test fixtures"""
        # Create temporary directory for tests
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)

        # Initialize detector with temporary directory
        self.detector = AIAudioDetector(base_dir=self.temp_dir)

        # Create sample training data
        self.sample_data = pd.DataFrame(
            {
                "filename": [
                    "file1.wav",
                    "file2.wav",
                    "file3.wav",
                    "file4.wav",
                    "file5.wav",
                    "file6.wav",
                ],
                "full_path": [
                    "/path/file1.wav",
                    "/path/file2.wav",
                    "/path/file3.wav",
                    "/path/file4.wav",
                    "/path/file5.wav",
                    "/path/file6.wav",
                ],
                "source_directory": ["AI", "AI", "AI", "Human", "Human", "Human"],
                "file_extension": [".wav", ".wav", ".wav", ".wav", ".wav", ".wav"],
                "is_ai": [True, True, True, False, False, False],
                "audio_duration": [2.5, 3.0, 2.8, 3.2, 2.9, 3.1],
                "sample_rate": [22050, 22050, 22050, 22050, 22050, 22050],
                "file_size_mb": [1.2, 1.5, 1.3, 1.6, 1.4, 1.7],
                "benford_chi2_p": [0.1, 0.05, 0.12, 0.8, 0.9, 0.85],
                "benford_chi2_stat": [15.2, 18.5, 16.1, 3.2, 2.1, 2.8],
                "spectral_centroid": [1500, 1600, 1550, 2000, 2100, 2050],
                "spectral_bandwidth": [800, 850, 825, 900, 950, 925],
                "temporal_rms_mean": [0.1, 0.12, 0.11, 0.15, 0.14, 0.145],
                "compression_clipping_ratio": [0.01, 0.02, 0.015, 0.005, 0.008, 0.006],
            }
        )

    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir)

    def test_initialization(self):
        """Test detector initialization"""
        self.assertIsInstance(self.detector, AIAudioDetector)
        self.assertEqual(self.detector.base_dir, self.temp_path)
        self.assertFalse(self.detector.is_trained)
        self.assertEqual(len(self.detector.training_history), 0)

    def test_get_audio_extensions(self):
        """Test audio extension retrieval"""
        extensions = self.detector.get_audio_extensions()
        expected_extensions = (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac")
        self.assertEqual(extensions, expected_extensions)

    def test_train_models(self):
        """Test model training"""
        results = self.detector.train_models(self.sample_data)

        # Check that training was successful
        self.assertIsInstance(results, dict)
        self.assertTrue(self.detector.is_trained)
        self.assertGreater(len(self.detector.feature_columns), 0)
        self.assertGreater(len(self.detector.training_history), 0)

        # Check that all models were trained
        expected_models = [
            "SGD",
            "PassiveAggressive",
            "RandomForest",
            "GradientBoosting",
        ]
        for model_name in expected_models:
            self.assertIn(model_name, results)
            self.assertIn("train_accuracy", results[model_name])
            self.assertIn("test_accuracy", results[model_name])

    def test_load_config(self):
        """Test configuration loading"""
        config = load_config()

        # Check that config has expected structure
        self.assertIn("models", config)
        self.assertIn("audio", config)
        self.assertIn("features", config)
        self.assertIn("processing", config)
        self.assertIn("output", config)

        # Check specific config values
        self.assertIn("supported_formats", config["audio"])
        self.assertIn("max_workers", config["processing"])

    @patch("ai_audio_detector.AudioAnalyzer")
    def test_predict_file_not_trained(self, mock_analyzer):
        """Test prediction when models are not trained"""
        result = self.detector.predict_file("/fake/path.wav")
        self.assertIsNone(result)

    def test_save_and_load_models(self):
        """Test model saving and loading"""
        # Train models first
        self.detector.train_models(self.sample_data)
        original_feature_columns = self.detector.feature_columns.copy()

        # Save models
        self.detector.save_models()

        # Create new detector and load models
        new_detector = AIAudioDetector(base_dir=self.temp_dir)
        load_success = new_detector.load_models()

        self.assertTrue(load_success)
        self.assertTrue(new_detector.is_trained)
        self.assertEqual(new_detector.feature_columns, original_feature_columns)

    def test_update_with_new_data(self):
        """Test adaptive model updating"""
        # Train initial models
        self.detector.train_models(self.sample_data)

        # Create new data for updating
        new_data = pd.DataFrame(
            {
                "filename": ["new_file1.wav", "new_file2.wav"],
                "full_path": ["/path/new_file1.wav", "/path/new_file2.wav"],
                "source_directory": ["AI", "Human"],
                "file_extension": [".wav", ".wav"],
                "is_ai": [True, False],
                "audio_duration": [2.7, 2.9],
                "sample_rate": [22050, 22050],
                "file_size_mb": [1.4, 1.1],
                "benford_chi2_p": [0.08, 0.85],
                "benford_chi2_stat": [16.1, 2.8],
                "spectral_centroid": [1550, 2050],
                "spectral_bandwidth": [825, 925],
                "temporal_rms_mean": [0.11, 0.145],
                "compression_clipping_ratio": [0.015, 0.006],
            }
        )

        # Update models
        update_results = self.detector.update_with_new_data(
            new_data, retrain_batch_models=True
        )

        self.assertIsInstance(update_results, dict)
        self.assertGreater(len(self.detector.training_history), 1)

    def test_show_data_balance(self):
        """Test data balance display"""
        # Train models first to have training history
        self.detector.train_models(self.sample_data)

        # This should not raise an exception
        try:
            self.detector.show_data_balance()
        except Exception as e:
            self.fail(f"show_data_balance raised an exception: {e}")

    def test_model_persistence_after_update(self):
        """Test that models persist correctly after updates"""
        # Train initial models
        self.detector.train_models(self.sample_data)
        initial_history_length = len(self.detector.training_history)

        # Create new data
        new_data = self.sample_data.copy()
        new_data["filename"] = ["new_" + f for f in new_data["filename"]]

        # Update models
        self.detector.update_with_new_data(new_data)

        # Check that training history was updated
        self.assertEqual(
            len(self.detector.training_history), initial_history_length + 1
        )

        # Save and reload to test persistence
        self.detector.save_models()
        new_detector = AIAudioDetector(base_dir=self.temp_dir)
        load_success = new_detector.load_models()

        self.assertTrue(load_success)
        self.assertEqual(len(new_detector.training_history), initial_history_length + 1)


if __name__ == "__main__":
    unittest.main()
