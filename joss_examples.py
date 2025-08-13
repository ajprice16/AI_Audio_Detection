#!/usr/bin/env python3
"""
Example script demonstrating AI Audio Detector functionality for JOSS paper

This script provides reproducible examples of the package's core functionality,
suitable for use in academic validation and review.
"""

import os
import sys
import numpy as np
from pathlib import Path
import tempfile
import librosa
from scipy.io.wavfile import write as write_wav
from typing import Tuple

from ai_audio_detector import AIAudioDetector, AudioAnalyzer, AudioFeatureExtractor


def generate_synthetic_audio(
    duration: float = 5.0, sample_rate: int = 22050, frequency: float = 440
) -> Tuple[np.ndarray, int]:
    """Generate synthetic audio for testing purposes."""
    t = np.linspace(0, duration, int(sample_rate * duration))
    # Create a simple sine wave with some noise to simulate real audio
    audio = 0.5 * np.sin(2 * np.pi * frequency * t) + 0.1 * np.random.randn(len(t))
    return audio, sample_rate


def demonstrate_feature_extraction() -> None:
    """Demonstrate audio feature extraction capabilities."""
    print("=== Audio Feature Extraction Demo ===")

    # Generate test audio
    audio, sr = generate_synthetic_audio()

    # Initialize analyzer (which uses the feature extractor)
    analyzer = AudioAnalyzer()

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        write_wav(tmp.name, sr, (audio * 32767).astype(np.int16))

        try:
            # Extract features using the analyzer
            features = analyzer.analyze_audio_file(tmp.name)

            print(f"Extracted {len(features)} features:")
            for key, value in list(features.items())[:10]:  # Show first 10 features
                if isinstance(value, (int, float)):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {str(value)[:50]}...")

            print(f"  ... and {len(features) - 10} more features")

        finally:
            os.unlink(tmp.name)


def demonstrate_benford_analysis() -> None:
    """Demonstrate Benford's Law analysis on audio data."""
    print("\n=== Benford's Law Analysis Demo ===")

    # Generate test audio
    audio, sr = generate_synthetic_audio()

    # Initialize analyzer
    analyzer = AudioAnalyzer()

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        write_wav(tmp.name, sr, (audio * 32767).astype(np.int16))

        try:
            # Analyze audio file
            results = analyzer.analyze_audio_file(tmp.name)

            # Extract Benford's Law related features
            benford_features = {k: v for k, v in results.items() if "benford" in k.lower()}

            print(f"Benford's Law features extracted:")
            for key, value in benford_features.items():
                print(f"  {key}: {value:.4f}")

        finally:
            os.unlink(tmp.name)


def demonstrate_ensemble_classification() -> None:
    """Demonstrate ensemble classification with multiple models."""
    print("\n=== Ensemble Classification Demo ===")

    # Initialize detector
    detector = AIAudioDetector()

    # Check if models are available
    if not detector.is_trained:
        print("Loading pre-trained models...")
        if detector.load_models():
            print("✓ Models loaded successfully")
        else:
            print("⚠ No pre-trained models found. Run training first.")
            return

    # Generate test audio files
    test_files = []
    for i, freq in enumerate([220, 440, 880]):  # Different frequencies
        audio, sr = generate_synthetic_audio(frequency=freq)

        with tempfile.NamedTemporaryFile(suffix=f"_test_{i}.wav", delete=False) as tmp:
            write_wav(tmp.name, sr, (audio * 32767).astype(np.int16))
            test_files.append(tmp.name)

    try:
        print(f"Testing ensemble classification on {len(test_files)} audio samples...")

        for i, audio_file in enumerate(test_files):
            result = detector.predict_single_file(audio_file)

            if result:
                print(f"Sample {i+1}:")
                # Correctly handle boolean prediction values
                prediction = result.get("prediction", "Unknown")
                print(f"  Prediction: {prediction}")
                print(f"  Confidence: {result.get('confidence', 0):.3f}")

                # Show model-specific predictions if available
                if "model_predictions" in result:
                    print("  Individual model predictions:")
                    for model_name, pred in result["model_predictions"].items():
                        confidence = result.get("model_confidences", {}).get(model_name, 0)
                        pred_text = "AI" if pred else "Human"
                        print(f"    {model_name}: {pred_text} ({confidence:.3f})")
            else:
                print(f"Sample {i+1}: Analysis failed")

    finally:
        # Clean up temporary files
        for tmp_file in test_files:
            try:
                os.unlink(tmp_file)
            except:
                pass


def demonstrate_batch_processing() -> None:
    """Demonstrate batch processing capabilities."""
    print("\n=== Batch Processing Demo ===")

    detector = AIAudioDetector()

    if not detector.is_trained and not detector.load_models():
        print("⚠ No pre-trained models available for batch processing demo")
        return

    # Create a temporary directory with test files
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Generate multiple test files
        for i in range(3):
            audio, sr = generate_synthetic_audio(frequency=220 * (i + 1))
            file_path = temp_path / f"test_audio_{i}.wav"
            write_wav(str(file_path), sr, (audio * 32767).astype(np.int16))

        print(f"Processing {len(list(temp_path.glob('*.wav')))} files in batch...")

        # Process batch using the directory
        results = detector.predict_batch(str(temp_path))

        if results is not None and len(results) > 0:
            print("Batch processing results:")
            for idx, result in enumerate(results):
                # Use prediction result from the dictionary
                prediction = result.get("prediction", "Unknown")
                confidence = result.get("confidence", 0)
                filename = result.get("filename", f"file_{idx}")
                print(f"  {filename}: {prediction} (confidence: {confidence:.3f})")
        else:
            print("Batch processing completed with no results")


def performance_benchmark() -> None:
    """Simple performance benchmark."""
    print("\n=== Performance Benchmark ===")

    import time

    detector = AIAudioDetector()

    if not detector.is_trained and not detector.load_models():
        print("⚠ No pre-trained models available for benchmark")
        return

    # Generate test audio
    audio, sr = generate_synthetic_audio(duration=10.0)  # 10 second audio

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        write_wav(tmp.name, sr, (audio * 32767).astype(np.int16))

        try:
            # Time the analysis
            start_time = time.time()
            result = detector.predict_single_file(tmp.name)
            end_time = time.time()

            analysis_time = end_time - start_time
            audio_duration = len(audio) / sr

            print(f"Analysis completed in {analysis_time:.2f} seconds")
            print(f"Audio duration: {audio_duration:.2f} seconds")
            print(f"Processing speed: {audio_duration/analysis_time:.2f}x realtime")

        finally:
            os.unlink(tmp.name)


def main() -> int:
    """Run all demonstrations."""
    print("AI Audio Detector - JOSS Paper Examples")
    print("=" * 50)

    try:
        demonstrate_feature_extraction()
        demonstrate_benford_analysis()
        demonstrate_ensemble_classification()
        demonstrate_batch_processing()
        performance_benchmark()

        print("\n" + "=" * 50)
        print("All demonstrations completed successfully!")
        print("\nThis script demonstrates the core functionality of AI Audio Detector")
        print("including feature extraction, Benford's Law analysis, ensemble")
        print("classification, and batch processing capabilities.")

    except Exception as e:
        print(f"\nError during demonstration: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
