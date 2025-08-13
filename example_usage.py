#!/usr/bin/env python3
"""
Example usage of the AI Audio Detector

This script demonstrates how to use the AIAudioDetector programmatically.
"""

import pandas as pd
from pathlib import Path
from typing import Optional
from ai_audio_detector import AIAudioDetector


def example_training() -> None:
    """Example: Train models with audio directories."""
    print("=== TRAINING EXAMPLE ===")

    # Initialize detector
    detector = AIAudioDetector()

    # Specify your audio directories
    human_dir = "path/to/human/audio"  # Replace with actual path
    ai_dir = "path/to/ai/audio"  # Replace with actual path

    # Check if directories exist
    if not Path(human_dir).exists() or not Path(ai_dir).exists():
        print("Please update the directory paths in this example script")
        return

        # Extract features
    print("Extracting features from human audio...")
    human_features = detector.extract_features_from_directory(human_dir, is_ai_directory=False)

    print("Extracting features from AI audio...")
    ai_features = detector.extract_features_from_directory(ai_dir, is_ai_directory=True)

    if not human_features or not ai_features:
        print("Could not extract features")
        return

    # Combine and train
    all_features = human_features + ai_features
    df_results = pd.DataFrame(all_features)

    print(f"Training with {len(df_results)} files ({len(human_features)} human, {len(ai_features)} AI)")
    training_results = detector.train_models(df_results)

    # Show results
    print("\nTraining Results:")
    for model_name, results in training_results.items():
        if "test_accuracy" in results:
            print(f"  {model_name}: {results['test_accuracy']:.3f}")


def example_prediction(detector: Optional[AIAudioDetector] = None) -> None:
    """Example: Predict single file and batch."""
    print("\n=== PREDICTION EXAMPLE ===")

    if detector is None:
        detector = AIAudioDetector()
        if not detector.load_models():
            print("No trained models found. Train first.")
            return

    # Single file prediction
    audio_file = "path/to/test/audio.wav"  # Replace with actual file
    if Path(audio_file).exists():
        print(f"Analyzing: {audio_file}")
        result = detector.predict_single_file(audio_file)

        if result and "error" not in result:
            prediction = result.get("prediction", "Unknown")
            print(f"  Prediction: {prediction}")
            print(f"  Confidence: {result.get('confidence', 0):.3f}")
            print(f"  AI Probability: {result.get('ai_probability', 0):.3f}")
        else:
            print("  Could not analyze file")

    # Batch prediction
    test_dir = "path/to/test/directory"  # Replace with actual directory
    if Path(test_dir).exists():
        print(f"\nBatch analysis: {test_dir}")
        results = detector.predict_batch(test_dir)

        if results is not None and len(results) > 0:
            ai_count = sum(1 for r in results if r.get("is_ai_generated", False))
            total_count = len(results)
            print(f"  Analyzed {total_count} files")
            print(f"  AI files: {ai_count} ({100*ai_count/total_count:.1f}%)")
            print(f"  Human files: {total_count-ai_count} ({100*(total_count-ai_count)/total_count:.1f}%)")


def example_adaptive_learning(detector: Optional[AIAudioDetector] = None) -> None:
    """Example: Add new data to existing models."""
    print("\n=== ADAPTIVE LEARNING EXAMPLE ===")

    if detector is None:
        detector = AIAudioDetector()
        if not detector.load_models():
            print("No trained models found. Train first.")
            return

    # Update with new data (demonstration)
    new_ai_dir = "path/to/new/ai/audio"  # Replace with actual path
    new_human_dir = "path/to/new/human/audio"  # Replace with actual path

    if Path(new_ai_dir).exists() and Path(new_human_dir).exists():
        print(f"Extracting features from new data...")
        ai_features = detector.extract_features_from_directory(new_ai_dir, is_ai_directory=True)
        human_features = detector.extract_features_from_directory(new_human_dir, is_ai_directory=False)

        new_features = ai_features + human_features
        update_results = detector.update_with_new_data(new_features)

        if update_results:
            print("Update successful!")
            for model_name, result in update_results.items():
                if model_name != "models_saved":
                    print(f"  {model_name}: {result}")


def example_spectrograms() -> None:
    """Example: Generate spectrograms."""
    print("\n=== SPECTROGRAM EXAMPLE ===")

    detector = AIAudioDetector()

    # Generate spectrograms for a directory
    audio_dir = "path/to/audio/directory"  # Replace with actual path
    output_dir = "spectrograms_output"

    if Path(audio_dir).exists():
        print(f"Generating spectrograms for: {audio_dir}")

        # Generate spectrograms for individual files
        audio_files = list(Path(audio_dir).glob("*.wav"))
        generated_count = 0

        for audio_file in audio_files[:5]:  # Limit to first 5 files
            output_file = Path(output_dir) / f"{audio_file.stem}_spectrogram.png"
            Path(output_dir).mkdir(exist_ok=True)

            if detector.generate_spectrogram(audio_file, output_file):
                generated_count += 1

        print(f"Generated {generated_count} spectrograms in {output_dir}")

    # Create comparison spectrograms
    ai_dir = "path/to/ai/audio"  # Replace with actual path
    human_dir = "path/to/human/audio"  # Replace with actual path
    comparison_dir = "comparisons_output"

    if Path(ai_dir).exists() and Path(human_dir).exists():
        print(f"Creating AI vs Human comparisons...")

        # Get a few sample files from each directory
        ai_files = list(Path(ai_dir).glob("*.wav"))[:3]
        human_files = list(Path(human_dir).glob("*.wav"))[:3]

        Path(comparison_dir).mkdir(exist_ok=True)

        for i, (ai_file, human_file) in enumerate(zip(ai_files, human_files)):
            output_path = Path(comparison_dir) / f"comparison_{i}.png"
            detector.compare_spectrograms(ai_file, human_file, output_path)
        print(f"Comparison spectrograms saved to {comparison_dir}")


def main() -> None:
    """Run all examples."""
    print("AI Audio Detector - Example Usage")
    print("=" * 50)

    # Step 1: Training (commented out - requires actual audio directories)
    # detector = example_training()

    # Step 2: Load existing models and predict
    detector = AIAudioDetector()
    if detector.load_models():
        print("Loaded existing models")
        example_prediction(detector)
        example_adaptive_learning(detector)
    else:
        print("No existing models found.")
        print("To use this example:")
        print("1. Update directory paths in this script")
        print("2. Uncomment the training section")
        print("3. Run the script to train models first")

    # Step 3: Spectrogram examples
    example_spectrograms()

    print("\nExample complete!")


if __name__ == "__main__":
    main()
