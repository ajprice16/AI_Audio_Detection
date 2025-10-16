"""
Command-line interface for AIAA: AI Audio Authenticity.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any

from .detector import AIAudioDetector
from .config import load_config


def run_interactive_mode(detector: AIAudioDetector) -> None:
    """
    Run interactive mode for the detector.

    Args:
        detector: Initialized AIAudioDetector instance.
    """
    print("\n=== AIAA: AI Audio Authenticity - Interactive Mode ===")
    print("Commands:")
    print("  predict <file_path> - Predict if audio file is AI-generated")
    print("  batch <directory> - Predict for all files in directory")
    print("  spectrogram <file_path> - Generate spectrogram")
    print("  compare <file1> <file2> - Compare two spectrograms")
    print("  train <ai_dir> <human_dir> - Train models with new data")
    print("  update <ai_dir> <human_dir> - Update models with new data")
    print("  status - Show model status")
    print("  help - Show this help")
    print("  quit - Exit")

    while True:
        try:
            command = input("\nEnter command: ").strip().split()

            if not command:
                continue

            cmd = command[0].lower()

            if cmd == "quit":
                break

            elif cmd == "help":
                print("Commands:")
                print("  predict <file_path> - Predict if audio file is AI-generated")
                print("  batch <directory> - Predict for all files in directory")
                print("  spectrogram <file_path> - Generate spectrogram")
                print("  compare <file1> <file2> - Compare two spectrograms")
                print("  train <ai_dir> <human_dir> - Train models with new data")
                print("  update <ai_dir> <human_dir> - Update models with new data")
                print("  status - Show model status")
                print("  help - Show this help")
                print("  quit - Exit")

            elif cmd == "predict":
                if len(command) < 2:
                    print("Usage: predict <file_path>")
                    continue

                file_path = Path(command[1])
                if not file_path.exists():
                    print(f"File not found: {file_path}")
                    continue

                result = detector.predict_single_file(file_path)
                if "error" in result:
                    print(f"Error: {result['error']}")
                else:
                    print(f"File: {result['filename']}")
                    print(f"Prediction: {result['prediction']}")
                    print(f"Confidence: {result['confidence']:.3f}")

            elif cmd == "batch":
                if len(command) < 2:
                    print("Usage: batch <directory>")
                    continue

                directory = Path(command[1])
                if not directory.exists():
                    print(f"Directory not found: {directory}")
                    continue

                results = detector.predict_batch(directory)
                print(f"\nProcessed {len(results)} files:")

                for result in results:
                    if "error" in result:
                        print(f"{result['filename']}: Error - {result['error']}")
                    else:
                        print(f"{result['filename']}: {result['prediction']} " f"(confidence: {result['confidence']:.3f})")

            elif cmd == "spectrogram":
                if len(command) < 2:
                    print("Usage: spectrogram <file_path>")
                    continue

                file_path = Path(command[1])
                if not file_path.exists():
                    print(f"File not found: {file_path}")
                    continue

                if detector.generate_spectrogram(file_path):
                    print(f"Spectrogram generated successfully")
                else:
                    print("Error generating spectrogram")

            elif cmd == "compare":
                if len(command) < 3:
                    print("Usage: compare <file1> <file2>")
                    continue

                file1 = Path(command[1])
                file2 = Path(command[2])

                if not file1.exists():
                    print(f"File not found: {file1}")
                    continue
                if not file2.exists():
                    print(f"File not found: {file2}")
                    continue

                if detector.compare_spectrograms(file1, file2):
                    print("Spectrogram comparison generated successfully")
                else:
                    print("Error generating spectrogram comparison")

            elif cmd == "train":
                if len(command) < 3:
                    print("Usage: train <ai_dir> <human_dir>")
                    continue

                ai_dir = Path(command[1])
                human_dir = Path(command[2])

                if not ai_dir.exists():
                    print(f"AI directory not found: {ai_dir}")
                    continue
                if not human_dir.exists():
                    print(f"Human directory not found: {human_dir}")
                    continue

                print("Extracting features from AI audio...")
                ai_features = detector.extract_features_from_directory(ai_dir, is_ai_directory=True)

                print("Extracting features from human audio...")
                human_features = detector.extract_features_from_directory(human_dir, is_ai_directory=False)

                if not ai_features or not human_features:
                    print("Error: Could not extract features from directories")
                    continue

                import pandas as pd

                all_features = ai_features + human_features
                df_results = pd.DataFrame(all_features)

                detector.show_data_balance(df_results)
                training_results: Dict[str, Dict[str, Any]] = detector.train_models(df_results)

                print("\nTraining completed!")
                for model_name, model_results in training_results.items():
                    if "error" in model_results:
                        print(f"{model_name}: Error - {model_results['error']}")
                    else:
                        print(f"{model_name}: Accuracy = {model_results['accuracy']:.4f}")

            elif cmd == "update":
                if not detector.is_trained:
                    print("Error: Models must be trained before updating")
                    continue

                if len(command) < 3:
                    print("Usage: update <ai_dir> <human_dir>")
                    continue

                ai_dir = Path(command[1])
                human_dir = Path(command[2])

                if not ai_dir.exists():
                    print(f"AI directory not found: {ai_dir}")
                    continue
                if not human_dir.exists():
                    print(f"Human directory not found: {human_dir}")
                    continue

                print("Extracting new features...")
                ai_features = detector.extract_features_from_directory(ai_dir, is_ai_directory=True)
                human_features = detector.extract_features_from_directory(human_dir, is_ai_directory=False)

                new_features = ai_features + human_features
                update_results = detector.update_with_new_data(new_features)

                print("\nUpdate completed!")
                for model_name, results in update_results.items():
                    if model_name != "models_saved":
                        print(f"{model_name}: {results}")

            elif cmd == "status":
                print(f"Models trained: {detector.is_trained}")
                if detector.is_trained:
                    print(f"Available models: {list(detector.models.keys())}")
                    print(f"Feature count: {len(detector.feature_columns)}")
                else:
                    print("No trained models available")

            else:
                print(f"Unknown command: {cmd}. Type 'help' for available commands.")

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")


def main() -> None:
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="AIAA: AI Audio Authenticity - Detect AI-generated audio using machine learning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train models
  python -m aiaa --train --ai-dir path/to/ai/audio --human-dir path/to/human/audio

  # Predict single file
  python -m aiaa --predict path/to/audio.wav

  # Batch prediction
  python -m aiaa --batch path/to/audio/directory

  # Interactive mode
  python -m aiaa --interactive

  # Generate spectrogram
  python -m aiaa --spectrogram path/to/audio.wav
        """,
    )

    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--train", action="store_true", help="Train models")
    mode_group.add_argument("--predict", type=str, help="Predict single file")
    mode_group.add_argument("--batch", type=str, help="Batch prediction on directory")
    mode_group.add_argument("--update", action="store_true", help="Update models with new data")
    mode_group.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    mode_group.add_argument("--spectrogram", type=str, help="Generate spectrogram for file")
    mode_group.add_argument("--compare", nargs=2, help="Compare two spectrograms")

    # Training arguments
    parser.add_argument("--ai-dir", type=str, help="Directory containing AI-generated audio")
    parser.add_argument("--human-dir", type=str, help="Directory containing human-generated audio")

    # Configuration
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--base-dir", type=str, help="Base directory for models and results")

    # Output options
    parser.add_argument("--output", type=str, help="Output file for results")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    args = parser.parse_args()

    try:
        # Initialize detector
        config_path = Path(args.config) if args.config else None
        detector = AIAudioDetector(base_dir=args.base_dir, config_path=config_path)

        if args.train:
            if not args.ai_dir or not args.human_dir:
                parser.error("--train requires --ai-dir and --human-dir")

            ai_dir = Path(args.ai_dir)
            human_dir = Path(args.human_dir)

            if not ai_dir.exists():
                print(f"Error: AI directory not found: {ai_dir}")
                sys.exit(1)
            if not human_dir.exists():
                print(f"Error: Human directory not found: {human_dir}")
                sys.exit(1)

            # Extract features
            print("Extracting features from AI audio...")
            ai_features = detector.extract_features_from_directory(ai_dir, is_ai_directory=True)

            print("Extracting features from human audio...")
            human_features = detector.extract_features_from_directory(human_dir, is_ai_directory=False)

            if not ai_features or not human_features:
                print("Error: Could not extract features from directories")
                sys.exit(1)

            # Combine and train
            import pandas as pd

            all_features = ai_features + human_features
            df_results = pd.DataFrame(all_features)

            detector.show_data_balance(df_results)
            training_results: Dict[str, Dict[str, Any]] = detector.train_models(df_results)

            print("\nTraining Results:")
            for model_name, model_results in training_results.items():
                if "error" in model_results:
                    print(f"{model_name}: Error - {model_results['error']}")
                else:
                    print(f"{model_name}: Accuracy = {model_results['accuracy']:.4f}")

        elif args.predict:
            file_path = Path(args.predict)
            if not file_path.exists():
                print(f"Error: File not found: {file_path}")
                sys.exit(1)

            result = detector.predict_single_file(file_path)
            if "error" in result:
                print(f"Error: {result['error']}")
                sys.exit(1)

            print(f"File: {result['filename']}")
            print(f"Prediction: {result['prediction']}")
            print(f"Confidence: {result['confidence']:.3f}")

            if args.verbose:
                print("\nModel predictions:")
                for model, pred in result["model_predictions"].items():
                    conf = result["model_confidences"][model]
                    print(f"  {model}: {'AI' if pred else 'Human'} (confidence: {conf:.3f})")

        elif args.batch:
            directory = Path(args.batch)
            if not directory.exists():
                print(f"Error: Directory not found: {directory}")
                sys.exit(1)

            results: List[Dict[str, Any]] = detector.predict_batch(directory)

            print(f"Processed {len(results)} files:")
            for result in results:
                if "error" in result:
                    print(f"{result['filename']}: Error - {result['error']}")
                else:
                    print(f"{result['filename']}: {result['prediction']} " f"(confidence: {result['confidence']:.3f})")

            # Save results if output specified
            if args.output:
                import pandas as pd

                df_batch_results = pd.DataFrame(results)
                output_path = Path(args.output)

                if output_path.suffix.lower() == ".csv":
                    df_batch_results.to_csv(output_path, index=False)
                    print(f"Results saved to {output_path}")
                else:
                    print("Warning: Output format not supported. Use .csv extension.")

        elif args.update:
            if not detector.is_trained:
                print("Error: Models must be trained before updating")
                sys.exit(1)

            if not args.ai_dir or not args.human_dir:
                parser.error("--update requires --ai-dir and --human-dir")

            ai_dir = Path(args.ai_dir)
            human_dir = Path(args.human_dir)

            # Extract new features
            ai_features = detector.extract_features_from_directory(ai_dir, is_ai_directory=True)
            human_features = detector.extract_features_from_directory(human_dir, is_ai_directory=False)

            new_features = ai_features + human_features
            update_results = detector.update_with_new_data(new_features)

            print("Update completed!")
            for model_name, results in update_results.items():
                if model_name != "models_saved":
                    print(f"{model_name}: {results}")

        elif args.spectrogram:
            file_path = Path(args.spectrogram)
            if not file_path.exists():
                print(f"Error: File not found: {file_path}")
                sys.exit(1)

            if detector.generate_spectrogram(file_path):
                print("Spectrogram generated successfully")
            else:
                print("Error generating spectrogram")

        elif args.compare:
            file1, file2 = args.compare
            file1_path = Path(file1)
            file2_path = Path(file2)

            if not file1_path.exists():
                print(f"Error: File not found: {file1_path}")
                sys.exit(1)
            if not file2_path.exists():
                print(f"Error: File not found: {file2_path}")
                sys.exit(1)

            if detector.compare_spectrograms(file1_path, file2_path):
                print("Spectrogram comparison generated successfully")
            else:
                print("Error generating spectrogram comparison")

        elif args.interactive:
            run_interactive_mode(detector)

    except KeyboardInterrupt:
        print("\nOperation interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
