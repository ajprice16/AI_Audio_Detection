"""
Performance benchmarks for AI Audio Detector
"""

import time
import tempfile
import os
import sys
import numpy as np
import json
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional, Union

# Add the parent directory to path to import the module if running standalone
if __name__ == "__main__" and not sys.path[0].endswith("AI_Audio"):
    sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import soundfile as sf

    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False

try:
    from memory_profiler import profile

    MEMORY_PROFILER_AVAILABLE = True
except ImportError:
    MEMORY_PROFILER_AVAILABLE = False
    # Create a dummy decorator if memory_profiler is not available

    def profile(func: Any) -> Any:
        return func


from ai_audio_detector import AIAudioDetector, AudioAnalyzer


class BenchmarkRunner:
    """Performance benchmark runner"""

    def __init__(self) -> None:
        self.results: Dict[str, Dict[str, Any]] = {}
        self.temp_dir: Optional[str] = None

    def setup_test_data(self, num_files: int = 20) -> Path:
        """Create test audio files for benchmarking"""
        self.temp_dir = tempfile.mkdtemp()
        temp_path = Path(self.temp_dir)

        audio_dir = temp_path / "benchmark_audio"
        audio_dir.mkdir()

        sample_rate = 22050
        duration = 3.0  # 3 seconds each
        t = np.linspace(0, duration, int(sample_rate * duration))

        print(f"Creating {num_files} synthetic audio files for benchmarking...")
        print("Note: Synthetic audio files will have limited class diversity")

        for i in range(num_files):
            # Create varied audio samples
            freq = 200 + i * 50
            audio = np.sin(2 * np.pi * freq * t) + 0.3 * np.random.normal(0, 1, len(t))
            audio = audio * 0.7

            if SOUNDFILE_AVAILABLE:
                sf.write(audio_dir / f"test_file_{i:03d}.wav", audio, sample_rate)
            else:
                # Fallback to scipy.io.wavfile if soundfile is not available
                from scipy.io import wavfile

                wavfile.write(audio_dir / f"test_file_{i:03d}.wav", sample_rate, (audio * 32767).astype(np.int16))

        return audio_dir

    def cleanup(self) -> None:
        """Clean up test data"""
        if self.temp_dir:
            shutil.rmtree(self.temp_dir)

    def benchmark_feature_extraction(self, audio_dir: Path, num_files: int = 20) -> Dict[str, Any]:
        """Benchmark feature extraction performance"""
        print(f"Benchmarking feature extraction with {num_files} files...")

        detector = AIAudioDetector(base_dir=self.temp_dir)

        start_time = time.time()
        # Extract features for AI directory (half the files)
        ai_features = detector.extract_features_from_directory(audio_dir, is_ai_directory=True)
        # Also extract as human directory to create mixed dataset
        human_features = detector.extract_features_from_directory(audio_dir, is_ai_directory=False)
        all_features = ai_features + human_features
        end_time = time.time()

        total_time = end_time - start_time
        time_per_file = total_time / (len(all_features))

        feature_extraction_results = {
            "total_time": total_time,
            "time_per_file": time_per_file,
            "files_processed": len(all_features),
            "throughput_files_per_second": len(all_features) / total_time,
        }
        self.results["feature_extraction"] = feature_extraction_results

        print(f"  Total time: {total_time:.2f}s")
        print(f"  Time per file: {time_per_file:.3f}s")
        print(f"  Throughput: {len(all_features) / total_time:.1f} files/second")

        return {"benchmark_results": feature_extraction_results, "features_data": all_features}

    def benchmark_training(self, features_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Benchmark model training performance"""
        print("Benchmarking model training...")

        import pandas as pd

        df = pd.DataFrame(features_data)

        # Check if we have enough class diversity for training
        if "is_ai" in df.columns:
            unique_labels = df["is_ai"].nunique()
            print(f"  Found {unique_labels} unique labels in training data")

            if unique_labels < 2:
                print("  ⚠️  Insufficient class diversity for model training (need at least 2 classes)")
                print("  Skipping model training benchmark...")

                # Create a dummy detector for compatibility
                detector = AIAudioDetector(base_dir=self.temp_dir)

                training_results = {
                    "training_time": 0.0,
                    "samples_trained": len(df),
                    "models_trained": 0,
                    "time_per_sample": 0.0,
                    "skipped_reason": "insufficient_class_diversity",
                }
                self.results["training"] = training_results
                return training_results
        else:
            print("  ⚠️  No 'is_ai' column found in training data")
            print("  Skipping model training benchmark...")

            # Create a dummy detector for compatibility
            detector = AIAudioDetector(base_dir=self.temp_dir)

            training_results = {
                "training_time": 0.0,
                "samples_trained": len(df),
                "models_trained": 0,
                "time_per_sample": 0.0,
                "skipped_reason": "no_labels",
            }
            self.results["training"] = training_results
            return training_results

        detector = AIAudioDetector(base_dir=self.temp_dir)

        start_time = time.time()
        model_training_results: Dict[str, Dict[str, Any]] = detector.train_models(df)
        end_time = time.time()

        training_time = end_time - start_time

        benchmark_results = {
            "training_time": training_time,
            "samples_trained": len(df),
            "models_trained": len(model_training_results),
            "time_per_sample": training_time / len(df),
        }
        self.results["training"] = benchmark_results

        print(f"  Training time: {training_time:.2f}s")
        print(f"  Samples: {len(df)}")
        print(f"  Time per sample: {training_time / len(df):.4f}s")

        return benchmark_results

    def benchmark_prediction(self, detector: AIAudioDetector, audio_dir: Path, num_files: int = 20) -> Dict[str, Any]:
        """Benchmark prediction performance"""
        print(f"Benchmarking prediction with {num_files} files...")

        # Sequential prediction
        audio_files = list(Path(audio_dir).glob("*.wav"))[:num_files]

        start_time = time.time()
        for audio_file in audio_files:
            detector.predict_single_file(audio_file)
        end_time = time.time()

        sequential_time = end_time - start_time

        # Batch prediction
        start_time = time.time()
        batch_results = detector.predict_batch(audio_dir)
        end_time = time.time()

        batch_time = end_time - start_time

        prediction_results = {
            "sequential_time": sequential_time,
            "batch_time": batch_time,
            "speedup": sequential_time / batch_time if batch_time > 0 else 0,
            "sequential_throughput": num_files / sequential_time,
            "batch_throughput": num_files / batch_time if batch_time > 0 else 0,
        }
        self.results["prediction"] = prediction_results

        print(f"  Sequential time: {sequential_time:.2f}s")
        print(f"  Batch time: {batch_time:.2f}s")
        print(f"  Speedup: {sequential_time / batch_time:.1f}x" if batch_time > 0 else "  Speedup: N/A")

        return prediction_results

    @profile
    def benchmark_memory_usage(self, audio_dir: Path) -> Dict[str, Any]:
        """Benchmark memory usage during processing"""
        print("Benchmarking memory usage...")

        detector = AIAudioDetector(base_dir=self.temp_dir)

        # Process files to measure memory usage
        features = detector.extract_features_from_directory(audio_dir, is_ai_directory=True)

        if features:
            import pandas as pd

            df = pd.DataFrame(features)

            # Check if we have enough class diversity for training
            if "label" in df.columns and df["label"].nunique() >= 2:
                training_results = detector.train_models(df)

                # Make some predictions
                audio_files = list(Path(audio_dir).glob("*.wav"))[:5]
                for audio_file in audio_files:
                    detector.predict_single_file(audio_file)
            else:
                print("  Skipping model training in memory benchmark (insufficient class diversity)")

        print("  Memory profiling complete (see output above)")

        # Return basic memory info (actual profiling details are printed)
        return {"memory_profiling": "completed", "note": "Detailed output printed above"}

    def run_all_benchmarks(self, num_files: int = 20) -> Dict[str, Any]:
        """Run all benchmarks"""
        print("=" * 60)
        print("AI AUDIO DETECTOR PERFORMANCE BENCHMARKS")
        print("=" * 60)

        try:
            # Setup
            audio_dir = self.setup_test_data(num_files)

            # Run benchmarks
            extraction_results = self.benchmark_feature_extraction(audio_dir, num_files)
            features = extraction_results["features_data"]
            training_results = self.benchmark_training(features)

            # Create a fresh detector for prediction benchmark
            detector = AIAudioDetector(base_dir=self.temp_dir)
            # Train it with the features if we have sufficient data
            if features:
                import pandas as pd

                df = pd.DataFrame(features)
                if "is_ai" in df.columns and df["is_ai"].nunique() >= 2:
                    detector.train_models(df)

            self.benchmark_prediction(detector, audio_dir, num_files)

            # Memory benchmark (optional, requires memory_profiler)
            try:
                self.benchmark_memory_usage(audio_dir)
            except ImportError:
                print("Skipping memory benchmark (memory_profiler not available)")

            # Overall performance summary
            self.results["summary"] = {
                "total_files_processed": num_files,
                "end_to_end_time": sum(
                    [
                        self.results.get("feature_extraction", {}).get("total_time", 0),
                        self.results.get("training", {}).get("training_time", 0),
                        self.results.get("prediction", {}).get("batch_time", 0),
                    ]
                ),
            }

            print("\n" + "=" * 60)
            print("BENCHMARK SUMMARY")
            print("=" * 60)
            print(f"Total files processed: {num_files}")
            print(f"End-to-end time: {self.results['summary']['end_to_end_time']:.2f}s")
            print(f"Overall throughput: {num_files / self.results['summary']['end_to_end_time']:.1f} files/second")

        finally:
            self.cleanup()

        return self.results

    def save_results(self, filename: str = "benchmark-results.json") -> None:
        """Save benchmark results to JSON file"""
        with open(filename, "w") as f:
            json.dump(self.results, f, indent=2)
        print(f"\nBenchmark results saved to: {filename}")


def main() -> None:
    """Run benchmarks"""
    try:
        import soundfile
    except ImportError:
        print("Error: soundfile is required for benchmarks")
        print("Install with: pip install soundfile")
        return

    benchmark = BenchmarkRunner()

    # Run with different file counts based on environment
    if os.environ.get("CI"):
        # Smaller benchmark for CI
        num_files = 10
    else:
        # Full benchmark for local testing
        num_files = 20

    benchmark.run_all_benchmarks(num_files)
    benchmark.save_results()


if __name__ == "__main__":
    main()
