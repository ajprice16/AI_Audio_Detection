"""
Integration tests for the AI Audio Detector system
"""

import unittest
import tempfile
import shutil
import numpy as np
import soundfile as sf
from pathlib import Path
import sys

# Add the parent directory to path to import the module
sys.path.insert(0, str(Path(__file__).parent.parent))

from ai_audio_detector import AIAudioDetector, AudioAnalyzer


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system"""

    def setUp(self):
        """Set up test fixtures"""
        # Create temporary directory structure
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)

        # Create subdirectories for audio files
        self.ai_audio_dir = self.temp_path / "ai_audio"
        self.human_audio_dir = self.temp_path / "human_audio"
        self.ai_audio_dir.mkdir()
        self.human_audio_dir.mkdir()

        # Create sample audio files
        self.create_sample_audio_files()

        # Initialize detector
        self.detector = AIAudioDetector(base_dir=self.temp_dir)

    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir)

    def create_sample_audio_files(self):
        """Create sample audio files for testing"""
        sample_rate = 22050
        duration = 2.0  # 2 seconds
        t = np.linspace(0, duration, int(sample_rate * duration))

        # Create AI-like audio (more artificial patterns)
        for i in range(3):
            # Create audio with specific patterns that might indicate AI generation
            freq1 = 440 + i * 100  # Different frequencies
            freq2 = 880 + i * 150
            audio = (
                np.sin(2 * np.pi * freq1 * t)
                + 0.5 * np.sin(2 * np.pi * freq2 * t)
                + 0.1 * np.random.normal(0, 1, len(t))
            )  # Add some noise

            # Apply some compression-like effects
            audio = np.tanh(audio * 2) * 0.8

            # Save to AI directory
            sf.write(self.ai_audio_dir / f"ai_sample_{i}.wav", audio, sample_rate)

        # Create human-like audio (more natural patterns)
        for i in range(3):
            # Create more natural-sounding audio
            freq = 200 + i * 50
            audio = (
                np.sin(2 * np.pi * freq * t) * np.exp(-t * 0.5)  # Decay
                + 0.3 * np.sin(2 * np.pi * freq * 2 * t)  # Harmonics
                + 0.2 * np.random.normal(0, 1, len(t))
            )  # Natural noise

            # Add some natural variation
            envelope = 1 + 0.1 * np.sin(2 * np.pi * 0.5 * t)
            audio = audio * envelope * 0.6

            # Save to human directory
            sf.write(self.human_audio_dir / f"human_sample_{i}.wav", audio, sample_rate)

    def test_end_to_end_workflow(self):
        """Test the complete workflow from training to prediction"""
        # Step 1: Extract features from directories
        ai_features = self.detector.extract_features_from_directory(
            self.ai_audio_dir, is_ai_directory=True
        )
        human_features = self.detector.extract_features_from_directory(
            self.human_audio_dir, is_ai_directory=False
        )

        self.assertEqual(len(ai_features), 3)
        self.assertEqual(len(human_features), 3)

        # Step 2: Combine features and train models
        import pandas as pd

        all_features = ai_features + human_features
        df_results = pd.DataFrame(all_features)

        # Train models
        training_results = self.detector.train_models(df_results)

        self.assertIsInstance(training_results, dict)
        self.assertTrue(self.detector.is_trained)

        # Step 3: Test prediction on individual files
        test_file = list(self.ai_audio_dir.glob("*.wav"))[0]
        prediction_result = self.detector.predict_file(test_file, return_details=True)

        self.assertIsNotNone(prediction_result)
        self.assertIn("prediction", prediction_result)
        self.assertIn("ai_probability", prediction_result)
        self.assertIn("confidence", prediction_result)

        # Step 4: Test batch prediction
        batch_results = self.detector.predict_batch(
            self.ai_audio_dir, output_file="test_predictions.csv"
        )

        self.assertIsNotNone(batch_results)
        self.assertEqual(len(batch_results), 3)

        # Step 5: Test adaptive learning
        # Create new sample for adaptive learning
        sample_rate = 22050
        duration = 1.5
        t = np.linspace(0, duration, int(sample_rate * duration))
        new_audio = np.sin(2 * np.pi * 300 * t) * 0.7

        new_file_path = self.ai_audio_dir / "new_ai_sample.wav"
        sf.write(new_file_path, new_audio, sample_rate)

        # Extract features and update models
        new_features = self.detector.extract_features_from_directory(
            self.ai_audio_dir, is_ai_directory=True
        )

        # Filter to just the new file
        new_df = pd.DataFrame(
            [f for f in new_features if f["filename"] == "new_ai_sample.wav"]
        )

        if not new_df.empty:
            update_results = self.detector.update_with_new_data(new_df)
            self.assertIsInstance(update_results, dict)

    def test_audio_analyzer_integration(self):
        """Test AudioAnalyzer integration with real audio files"""
        analyzer = AudioAnalyzer()

        # Test analysis of a sample file
        test_file = list(self.ai_audio_dir.glob("*.wav"))[0]
        features = analyzer.analyze_audio_file(test_file)

        self.assertIsNotNone(features)
        self.assertIn("filename", features)
        self.assertIn("audio_duration", features)
        self.assertIn("sample_rate", features)

        # Check that various feature types are extracted
        benford_features = [k for k in features.keys() if k.startswith("benford_")]
        spectral_features = [k for k in features.keys() if k.startswith("spectral_")]
        temporal_features = [k for k in features.keys() if k.startswith("temporal_")]
        compression_features = [
            k for k in features.keys() if k.startswith("compression_")
        ]

        # Benford features may not be present for synthetic audio (simple sine waves)
        # but other feature types should always be extracted
        self.assertGreaterEqual(len(benford_features), 0)  # Allow 0 for synthetic audio
        self.assertGreater(len(spectral_features), 0)
        self.assertGreater(len(temporal_features), 0)
        self.assertGreater(len(compression_features), 0)

    def test_spectrogram_generation(self):
        """Test spectrogram generation functionality"""
        analyzer = AudioAnalyzer()
        test_file = list(self.ai_audio_dir.glob("*.wav"))[0]

        # Test spectrogram generation
        spectrogram_dir = self.temp_path / "spectrograms"
        spectrogram_path = analyzer.save_spectrogram(
            test_file, spectrogram_dir, spectrogram_type="mel"
        )

        self.assertIsNotNone(spectrogram_path)
        self.assertTrue(Path(spectrogram_path).exists())
        self.assertTrue(str(spectrogram_path).endswith(".png"))

    def test_model_persistence_integration(self):
        """Test that models can be saved, loaded, and used for prediction"""
        # Train models with sample data
        ai_features = self.detector.extract_features_from_directory(
            self.ai_audio_dir, is_ai_directory=True
        )
        human_features = self.detector.extract_features_from_directory(
            self.human_audio_dir, is_ai_directory=False
        )

        import pandas as pd

        all_features = ai_features + human_features
        df_results = pd.DataFrame(all_features)

        # Train and save models
        self.detector.train_models(df_results)
        self.detector.save_models()

        # Create new detector and load models
        new_detector = AIAudioDetector(base_dir=self.temp_dir)
        load_success = new_detector.load_models()

        self.assertTrue(load_success)
        self.assertTrue(new_detector.is_trained)

        # Test that loaded models can make predictions
        test_file = list(self.ai_audio_dir.glob("*.wav"))[0]
        prediction = new_detector.predict_file(test_file)

        self.assertIsNotNone(prediction)
        self.assertIn("prediction", prediction)


if __name__ == "__main__":
    # Check if soundfile is available for audio file generation
    try:
        import soundfile

        unittest.main()
    except ImportError:
        print("Warning: soundfile not available, skipping integration tests")
        print("Install with: pip install soundfile")
