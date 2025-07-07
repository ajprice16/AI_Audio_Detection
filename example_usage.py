#!/usr/bin/env python3
"""
Example usage of the AI Audio Detector

This script demonstrates how to use the AIAudioDetector programmatically.
"""

import pandas as pd
from pathlib import Path
from ai_audio_detector import AIAudioDetector

def example_training():
    """Example: Train models with audio directories."""
    print("=== TRAINING EXAMPLE ===")
    
    # Initialize detector
    detector = AIAudioDetector()
    
    # Specify your audio directories
    human_dir = "path/to/human/audio"  # Replace with actual path
    ai_dir = "path/to/ai/audio"        # Replace with actual path
    
    # Check if directories exist
    if not Path(human_dir).exists() or not Path(ai_dir).exists():
        print("Please update the directory paths in this example script")
        return None
    
    # Extract features
    print("Extracting features from human audio...")
    human_features = detector.extract_features_from_directory(human_dir, is_ai_directory=False)
    
    print("Extracting features from AI audio...")
    ai_features = detector.extract_features_from_directory(ai_dir, is_ai_directory=True)
    
    if not human_features or not ai_features:
        print("Could not extract features")
        return None
    
    # Combine and train
    all_features = human_features + ai_features
    df_results = pd.DataFrame(all_features)
    
    print(f"Training with {len(df_results)} files ({len(human_features)} human, {len(ai_features)} AI)")
    training_results = detector.train_models(df_results)
    
    # Show results
    print("\nTraining Results:")
    for model_name, results in training_results.items():
        print(f"  {model_name}: {results['test_accuracy']:.3f}")
    
    return detector

def example_prediction(detector=None):
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
        result = detector.predict_file(audio_file, return_details=True)
        
        if result:
            print(f"  Prediction: {'AI' if result['is_ai'] else 'Human'}")
            print(f"  Confidence: {result['confidence']:.3f}")
            print(f"  AI Probability: {result['ai_probability']:.3f}")
        else:
            print("  Could not analyze file")
    
    # Batch prediction
    test_dir = "path/to/test/directory"  # Replace with actual directory
    if Path(test_dir).exists():
        print(f"\nBatch analysis: {test_dir}")
        results = detector.predict_batch(test_dir, output_file="batch_results.csv")
        
        if results is not None:
            ai_count = results['is_ai'].sum()
            total_count = len(results)
            print(f"  Analyzed {total_count} files")
            print(f"  AI files: {ai_count} ({100*ai_count/total_count:.1f}%)")
            print(f"  Human files: {total_count-ai_count} ({100*(total_count-ai_count)/total_count:.1f}%)")

def example_adaptive_learning(detector=None):
    """Example: Add new data to existing models."""
    print("\n=== ADAPTIVE LEARNING EXAMPLE ===")
    
    if detector is None:
        detector = AIAudioDetector()
        if not detector.load_models():
            print("No trained models found. Train first.")
            return
    
    # Add new AI data
    new_ai_dir = "path/to/new/ai/audio"  # Replace with actual path
    if Path(new_ai_dir).exists():
        print(f"Adding new AI data from: {new_ai_dir}")
        update_results = detector.add_ai_data(new_ai_dir, retrain_batch_models=True)
        
        if update_results:
            print("Update successful!")
            for model_name, result in update_results.items():
                print(f"  {model_name}: accuracy = {result['accuracy']:.3f}")
    
    # Add new human data
    new_human_dir = "path/to/new/human/audio"  # Replace with actual path
    if Path(new_human_dir).exists():
        print(f"Adding new human data from: {new_human_dir}")
        update_results = detector.add_human_data(new_human_dir, retrain_batch_models=True)
        
        if update_results:
            print("Update successful!")
            for model_name, result in update_results.items():
                print(f"  {model_name}: accuracy = {result['accuracy']:.3f}")

def example_spectrograms():
    """Example: Generate spectrograms."""
    print("\n=== SPECTROGRAM EXAMPLE ===")
    
    detector = AIAudioDetector()
    
    # Generate spectrograms for a directory
    audio_dir = "path/to/audio/directory"  # Replace with actual path
    output_dir = "spectrograms_output"
    
    if Path(audio_dir).exists():
        print(f"Generating spectrograms for: {audio_dir}")
        generated_paths = detector.generate_spectrograms_batch(
            audio_dir, output_dir, spectrogram_type='mel'
        )
        print(f"Generated {len(generated_paths)} spectrograms in {output_dir}")
    
    # Create comparison spectrograms
    ai_dir = "path/to/ai/audio"      # Replace with actual path
    human_dir = "path/to/human/audio"  # Replace with actual path
    comparison_dir = "comparisons_output"
    
    if Path(ai_dir).exists() and Path(human_dir).exists():
        print(f"Creating AI vs Human comparisons...")
        detector.create_spectrogram_comparison(
            ai_dir, human_dir, comparison_dir, spectrogram_type='mel', num_samples=3
        )
        print(f"Comparison spectrograms saved to {comparison_dir}")

def main():
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
