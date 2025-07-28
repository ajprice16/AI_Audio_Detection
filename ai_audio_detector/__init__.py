"""
AI Audio Detector using Benford's Law and librosa audio feature extraction.
Trains Random Forest, Gradient Boosting, SGD, and Passive Aggressive classifiers.
Returns highest confidence prediction with detailed feature analysis.
"""

__version__ = "1.1.0"

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from datetime import datetime
from tqdm import tqdm
import warnings
import functools
import random
import yaml
import argparse
import sys

warnings.filterwarnings("ignore", category=UserWarning)
from concurrent.futures import ProcessPoolExecutor, as_completed

# ML imports
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier, PassiveAggressiveClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Audio processing imports
import librosa
from scipy import stats
from scipy.stats import ks_2samp

# Visualization imports
import matplotlib.pyplot as plt
import librosa.display

# Get the directory where the script is located, or use current working directory
BASE_DIR = Path.cwd()


def load_config(config_path=None):
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = BASE_DIR / "config.yaml"

    default_config = {
        "models": {
            "incremental": {
                "sgd": {"random_state": 42, "loss": "log_loss"},
                "passive_aggressive": {"random_state": 42},
            },
            "batch": {
                "random_forest": {
                    "n_estimators": 200,
                    "random_state": 42,
                    "n_jobs": -1,
                },
                "gradient_boosting": {"n_estimators": 200, "random_state": 42},
            },
        },
        "audio": {
            "supported_formats": [".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"],
            "default_sample_rate": None,
        },
        "features": {
            "benford": {"min_frequencies": 10},
            "spectral": {"n_mfcc": 13, "n_mels": 128},
        },
        "processing": {"max_workers": 4, "batch_threshold": 3},
        "output": {
            "models_dir": "models",
            "results_dir": "results",
            "spectrograms_dir": "spectrograms",
            "comparisons_dir": "spectrogram_comparisons",
        },
        "visualization": {"figsize": [12, 8], "dpi": 300, "colorbar": True},
    }

    if Path(config_path).exists():
        try:
            with open(config_path, "r") as f:
                user_config = yaml.safe_load(f)
            # Merge with defaults
            config = {**default_config, **user_config}
            return config
        except Exception as e:
            print(f"Warning: Could not load config file {config_path}: {e}")
            print("Using default configuration")

    return default_config


class AudioFeatureExtractor:
    """Audio feature extraction."""

    @staticmethod
    def extract_benford_features(frequencies):
        """Extract Benford's Law features from frequency data."""
        try:
            if not frequencies or len(frequencies) < 10:
                return {}

            # Convert to positive values and remove zeros
            clean_freqs = [abs(f) for f in frequencies if f != 0 and not np.isnan(f)]
            if len(clean_freqs) < 10:
                return {}

            # First digit analysis
            first_digits = []
            for f in clean_freqs:
                first_char = str(abs(f)).split(".")[0][0]
                if first_char.isdigit():
                    first_digits.append(int(first_char))

            if len(first_digits) < 10:
                return {}

            # Expected Benford distribution
            expected_benford = [np.log10(1 + 1 / d) for d in range(1, 10)]

            # Observed distribution
            observed_counts = [first_digits.count(d) for d in range(1, 10)]
            total_count = sum(observed_counts)

            if total_count == 0:
                return {}

            observed_freq = [c / total_count for c in observed_counts]

            # Calculate statistics
            features = {}

            # Chi-square test
            try:
                expected_counts = [total_count * exp for exp in expected_benford]
                chi2_stat, chi2_p = stats.chisquare(observed_counts, expected_counts)
                features["chi2_p"] = chi2_p
                features["chi2_stat"] = chi2_stat
            except:
                features["chi2_p"] = 1.0
                features["chi2_stat"] = 0.0

            # KS test
            try:
                # Create empirical distribution
                empirical_data = []
                for digit, count in enumerate(observed_counts, 1):
                    empirical_data.extend([digit] * count)

                # Expected data based on Benford's law
                expected_data = []
                for digit in range(1, 10):
                    count = int(total_count * expected_benford[digit - 1])
                    expected_data.extend([digit] * count)

                if len(empirical_data) > 0 and len(expected_data) > 0:
                    ks_stat, ks_p = ks_2samp(empirical_data, expected_data)
                    features["ks_p"] = ks_p
                    features["ks_stat"] = ks_stat
                else:
                    features["ks_p"] = 1.0
                    features["ks_stat"] = 0.0
            except:
                features["ks_p"] = 1.0
                features["ks_stat"] = 0.0

            # Mean absolute deviation from expected
            mad = sum(
                abs(obs - exp) for obs, exp in zip(observed_freq, expected_benford)
            ) / len(expected_benford)
            features["mad"] = mad

            # Maximum deviation
            max_dev = max(
                abs(obs - exp) for obs, exp in zip(observed_freq, expected_benford)
            )
            features["max_deviation"] = max_dev

            # Entropy
            entropy = -sum(p * np.log(p) for p in observed_freq if p > 0)
            features["entropy"] = entropy

            return features

        except Exception as e:
            return {}

    @staticmethod
    def extract_spectral_features(y, sr):
        """Extract spectral features from audio."""
        try:
            features = {}

            # Basic spectral features
            features["spectral_centroid"] = np.mean(
                librosa.feature.spectral_centroid(y=y, sr=sr)
            )
            features["spectral_bandwidth"] = np.mean(
                librosa.feature.spectral_bandwidth(y=y, sr=sr)
            )
            features["spectral_rolloff"] = np.mean(
                librosa.feature.spectral_rolloff(y=y, sr=sr)
            )
            features["zero_crossing_rate"] = np.mean(
                librosa.feature.zero_crossing_rate(y)
            )

            # MFCC features
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            for i in range(13):
                features[f"mfcc_{i}"] = np.mean(mfccs[i])
                features[f"mfcc_{i}_std"] = np.std(mfccs[i])

            # Chroma features
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            features["chroma_mean"] = np.mean(chroma)
            features["chroma_std"] = np.std(chroma)

            # Spectral contrast
            contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
            features["spectral_contrast"] = np.mean(contrast)

            return features
        except:
            return {}

    @staticmethod
    def extract_temporal_features(y, sr):
        """Extract temporal features from audio"""
        try:
            features = {}

            # RMS energy
            rms = librosa.feature.rms(y=y)
            features["rms_mean"] = np.mean(rms)
            features["rms_std"] = np.std(rms)

            # Tempo
            try:
                tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
                features["tempo"] = float(tempo)
            except:
                features["tempo"] = 0.0

            # Spectral flatness
            flatness = librosa.feature.spectral_flatness(y=y)
            features["spectral_flatness"] = np.mean(flatness)

            # Dynamic range
            features["dynamic_range"] = np.max(y) - np.min(y)

            # Peak-to-RMS ratio
            peak = np.max(np.abs(y))
            rms_value = np.sqrt(np.mean(y**2))
            features["peak_to_rms"] = peak / (rms_value + 1e-10)

            return features
        except:
            return {}

    @staticmethod
    def extract_compression_features(y, sr):
        """Extract compression-related features."""
        try:
            features = {}

            # Bit depth estimation (rough)
            unique_values = len(np.unique(y))
            features["estimated_bit_depth"] = (
                np.log2(unique_values) if unique_values > 1 else 0
            )

            # Clipping detection
            max_val = np.max(np.abs(y))
            clipping_threshold = 0.99
            clipped_samples = np.sum(np.abs(y) > clipping_threshold)
            features["clipping_ratio"] = clipped_samples / len(y)

            # DC offset
            features["dc_offset"] = np.mean(y)

            # High frequency content (above 18kHz)
            if sr > 30000:
                fft = np.fft.fft(y)
                freqs = np.fft.fftfreq(len(fft), 1 / sr)
                high_freq_power = np.sum(np.abs(fft[freqs > 18000]) ** 2)
                total_power = np.sum(np.abs(fft) ** 2)
                features["high_freq_ratio"] = high_freq_power / (total_power + 1e-10)
            else:
                features["high_freq_ratio"] = 0.0

            return features
        except:
            return {}


class AudioAnalyzer:
    """Main audio analysis class."""

    def __init__(self):
        self.extractor = AudioFeatureExtractor()

    def analyze_audio_file(
        self,
        audio_path,
        source_dir=None,
        save_spectrogram=False,
        spectrogram_dir=None,
        spectrogram_type="mel",
    ):
        """Comprehensive audio analysis."""
        try:
            audio_path = Path(audio_path)

            # Load audio
            y, sr = librosa.load(audio_path, sr=None)
            duration = len(y) / sr

            # Initialize features dictionary
            features = {
                "filename": audio_path.name,
                "full_path": str(audio_path),
                "source_directory": (
                    source_dir if source_dir else audio_path.parent.name
                ),
                "file_extension": audio_path.suffix.lower(),
                "audio_duration": duration,
                "sample_rate": sr,
                "file_size_mb": audio_path.stat().st_size / (1024 * 1024),
            }

            # Save spectrogram if requested
            if save_spectrogram and spectrogram_dir:
                spectrogram_path = self.save_spectrogram(
                    audio_path, spectrogram_dir, spectrogram_type
                )
                features["spectrogram_path"] = (
                    str(spectrogram_path) if spectrogram_path else None
                )

            # Extract frequency data for Benford analysis
            try:
                # Use FFT frequencies
                fft = np.fft.fft(y)
                frequencies = np.abs(fft[fft != 0])  # Remove zeros

                if len(frequencies) > 0:
                    benford_features = self.extractor.extract_benford_features(
                        frequencies
                    )
                    for k, v in benford_features.items():
                        features[f"benford_{k}"] = v
            except:
                pass

            # Extract spectral features
            spectral_features = self.extractor.extract_spectral_features(y, sr)
            for k, v in spectral_features.items():
                features[f"spectral_{k}"] = v

            # Extract temporal features
            temporal_features = self.extractor.extract_temporal_features(y, sr)
            for k, v in temporal_features.items():
                features[f"temporal_{k}"] = v

            # Extract compression features
            compression_features = self.extractor.extract_compression_features(y, sr)
            for k, v in compression_features.items():
                features[f"compression_{k}"] = v

            return features

        except Exception as e:
            print(f"Error analyzing {audio_path}: {e}")
            return None

    def save_spectrogram(
        self,
        audio_path,
        output_dir,
        spectrogram_type="mel",
        figsize=(12, 8),
        dpi=300,
        include_colorbar=True,
    ):
        """
        Generate and save spectrogram for an audio file.

        Args:
            audio_path: Path to audio file
            output_dir: Directory to save spectrogram
            spectrogram_type: Type of spectrogram ('mel', 'stft', 'cqt', 'chroma', 'mfcc')
            figsize: Figure size tuple (width, height)
            dpi: DPI for saved image
            include_colorbar: Whether to include colorbar in plot

        Returns:
            Path to saved spectrogram file or None if error
        """
        try:
            audio_path = Path(audio_path)
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            # Load audio
            y, sr = librosa.load(audio_path, sr=None)

            # Create figure
            plt.figure(figsize=figsize)

            if spectrogram_type.lower() == "mel":
                # Mel spectrogram
                S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
                S_dB = librosa.power_to_db(S, ref=np.max)
                img = librosa.display.specshow(
                    S_dB, x_axis="time", y_axis="mel", sr=sr, fmax=8000
                )
                plt.title(f"Mel Spectrogram - {audio_path.name}", fontsize=14)
                plt.ylabel("Mel Frequency")

            elif spectrogram_type.lower() == "stft":
                # STFT spectrogram
                D = librosa.stft(y)
                S_dB = librosa.amplitude_to_db(np.abs(D), ref=np.max)
                img = librosa.display.specshow(S_dB, x_axis="time", y_axis="hz", sr=sr)
                plt.title(f"STFT Spectrogram - {audio_path.name}", fontsize=14)
                plt.ylabel("Frequency (Hz)")

            elif spectrogram_type.lower() == "cqt":
                # Constant-Q Transform
                C = librosa.cqt(y, sr=sr)
                C_dB = librosa.amplitude_to_db(np.abs(C), ref=np.max)
                img = librosa.display.specshow(
                    C_dB, x_axis="time", y_axis="cqt_hz", sr=sr
                )
                plt.title(f"CQT Spectrogram - {audio_path.name}", fontsize=14)
                plt.ylabel("Frequency (Hz)")

            elif spectrogram_type.lower() == "chroma":
                # Chromagram
                chroma = librosa.feature.chroma_stft(y=y, sr=sr)
                img = librosa.display.specshow(chroma, x_axis="time", y_axis="chroma")
                plt.title(f"Chromagram - {audio_path.name}", fontsize=14)
                plt.ylabel("Pitch Class")

            elif spectrogram_type.lower() == "mfcc":
                # MFCC
                mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
                img = librosa.display.specshow(mfccs, x_axis="time")
                plt.title(f"MFCC - {audio_path.name}", fontsize=14)
                plt.ylabel("MFCC Coefficients")

            else:
                # Default to mel spectrogram
                S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
                S_dB = librosa.power_to_db(S, ref=np.max)
                img = librosa.display.specshow(
                    S_dB, x_axis="time", y_axis="mel", sr=sr, fmax=8000
                )
                plt.title(f"Mel Spectrogram - {audio_path.name}", fontsize=14)
                plt.ylabel("Mel Frequency")

            plt.xlabel("Time (s)")

            if include_colorbar:
                plt.colorbar(
                    img,
                    format=(
                        "%+2.0f dB"
                        if spectrogram_type in ["mel", "stft", "cqt"]
                        else None
                    ),
                )

            plt.tight_layout()

            # Save spectrogram
            output_filename = f"{audio_path.stem}_{spectrogram_type}_spectrogram.png"
            output_path = output_dir / output_filename
            plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
            plt.close()  # Important: close figure to free memory

            return output_path

        except Exception as e:
            print(f"Error creating spectrogram for {audio_path}: {e}")
            plt.close()  # Close figure even on error
            return None


def process_single_audio_file(args):
    """Process a single audio file - for multiprocessing."""
    if len(args) == 3:
        # Legacy format for backward compatibility
        audio_file, is_ai, source_dir = args
        save_spectrogram = False
        spectrogram_dir = None
        spectrogram_type = "mel"
    else:
        # New format with spectrogram options
        (
            audio_file,
            is_ai,
            source_dir,
            save_spectrogram,
            spectrogram_dir,
            spectrogram_type,
        ) = args

    try:
        analyzer = AudioAnalyzer()
        features = analyzer.analyze_audio_file(
            audio_file, source_dir, save_spectrogram, spectrogram_dir, spectrogram_type
        )
        if features:
            features["is_ai"] = is_ai
            return features
        return None
    except Exception as e:
        print(f"Error processing {audio_file.name}: {e}")
        return None


def process_single_prediction(args):
    """Process a single audio file for prediction - multiprocessing version."""
    audio_file, feature_columns, scaler_data, models_data = args

    try:
        # Extract features
        analyzer = AudioAnalyzer()
        features = analyzer.analyze_audio_file(audio_file)
        if not features:
            return None

        # Prepare feature vector
        feature_dict = {}
        for col in feature_columns:
            # Map feature names
            base_name = col
            for prefix in ["benford_", "spectral_", "temporal_", "compression_"]:
                if col.startswith(prefix):
                    base_name = col[len(prefix) :]
                    break

            feature_dict[col] = features.get(base_name, features.get(col, 0))

        X = pd.DataFrame([feature_dict])

        # Reconstruct scaler
        scaler = StandardScaler()
        scaler.mean_ = scaler_data["mean_"]
        scaler.scale_ = scaler_data["scale_"]
        scaler.var_ = scaler_data["var_"]
        scaler.n_features_in_ = scaler_data["n_features_in_"]
        scaler.n_samples_seen_ = scaler_data["n_samples_seen_"]

        X_scaled = scaler.transform(X.values)

        # Get predictions from all models
        predictions = {}
        for name, model in models_data.items():
            try:
                pred = model.predict(X_scaled)[0]
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(X_scaled)[0]
                    ai_probability = proba[1] if len(proba) > 1 else 0.5
                    confidence = max(proba)
                else:
                    ai_probability = float(pred)
                    confidence = 0.8

                predictions[name] = {
                    "prediction": bool(pred),
                    "ai_probability": ai_probability,
                    "confidence": confidence,
                }
            except Exception as e:
                continue

        if not predictions:
            return None

        # Ensemble prediction
        avg_ai_prob = np.mean([p["ai_probability"] for p in predictions.values()])
        ensemble_prediction = avg_ai_prob > 0.5
        ensemble_confidence = np.mean([p["confidence"] for p in predictions.values()])

        result = {
            "filename": audio_file.name,
            "full_path": str(audio_file),
            "prediction": ensemble_prediction,
            "ai_probability": avg_ai_prob,
            "confidence": ensemble_confidence,
            "is_ai": ensemble_prediction,
            "individual_predictions": predictions,
            "audio_duration": features.get("audio_duration", 0),
            "sample_rate": features.get("sample_rate", 0),
            "file_size_mb": audio_file.stat().st_size / (1024 * 1024),
        }

        return result

    except Exception as e:
        print(f"Error processing {audio_file.name}: {e}")
        return None


class AIAudioDetector:
    """Main AI audio detection system."""

    def __init__(self, base_dir=None, config_path=None):
        self.base_dir = Path(base_dir) if base_dir else BASE_DIR
        self.config = load_config(config_path)

        # Create directories
        self.model_dir = self.base_dir / self.config["output"]["models_dir"]
        self.results_dir = self.base_dir / self.config["output"]["results_dir"]
        self.spectrograms_dir = (
            self.base_dir / self.config["output"]["spectrograms_dir"]
        )

        for directory in [self.model_dir, self.results_dir, self.spectrograms_dir]:
            directory.mkdir(exist_ok=True, parents=True)

        # Initialize models with config parameters
        self.incremental_models = {
            "SGD": SGDClassifier(**self.config["models"]["incremental"]["sgd"]),
            "PassiveAggressive": PassiveAggressiveClassifier(
                **self.config["models"]["incremental"]["passive_aggressive"]
            ),
        }

        self.batch_models = {
            "RandomForest": RandomForestClassifier(
                **self.config["models"]["batch"]["random_forest"]
            ),
            "GradientBoosting": GradientBoostingClassifier(
                **self.config["models"]["batch"]["gradient_boosting"]
            ),
        }

        self.scaler = StandardScaler()
        self.feature_columns = []
        self.is_trained = False
        self.training_history = []

    def get_audio_extensions(self):
        """Get supported audio file extensions."""
        return tuple(self.config["audio"]["supported_formats"])

    def extract_features_from_directory(
        self,
        audio_dir,
        is_ai_directory=True,
        max_workers=None,
        save_spectrograms=False,
        spectrogram_dir=None,
        spectrogram_type="mel",
    ):
        """Extract features from all audio files in a directory."""
        audio_path = Path(audio_dir)
        if not audio_path.exists():
            print(f"Directory {audio_dir} not found")
            return []

        # Use config for max_workers if not specified
        if max_workers is None:
            max_workers = self.config["processing"]["max_workers"]

        # Find audio files using config
        audio_extensions = self.get_audio_extensions()
        audio_files = [
            f for f in audio_path.iterdir() if f.suffix.lower() in audio_extensions
        ]

        if not audio_files:
            print(f"No audio files found in {audio_dir}")
            return []

        print(f"Processing {len(audio_files)} files from {audio_path.name}...")
        if save_spectrograms and spectrogram_dir:
            print(f"Saving {spectrogram_type} spectrograms to {spectrogram_dir}")

        # Prepare arguments for multiprocessing
        source_name = "AI_Generated" if is_ai_directory else "Human_Generated"
        if save_spectrograms and spectrogram_dir:
            args_list = [
                (
                    audio_file,
                    is_ai_directory,
                    source_name,
                    save_spectrograms,
                    spectrogram_dir,
                    spectrogram_type,
                )
                for audio_file in audio_files
            ]
        else:
            args_list = [
                (audio_file, is_ai_directory, source_name) for audio_file in audio_files
            ]

        results = []

        batch_threshold = self.config["processing"]["batch_threshold"]
        if len(audio_files) > batch_threshold:
            # Use multiprocessing for larger batches
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                future_to_file = {
                    executor.submit(process_single_audio_file, args): args[0]
                    for args in args_list
                }

                with tqdm(
                    total=len(audio_files), desc=f"Processing {source_name}"
                ) as pbar:
                    for future in as_completed(future_to_file):
                        try:
                            result = future.result()
                            if result:
                                results.append(result)
                        except Exception as e:
                            print(f"Error: {e}")
                        pbar.update(1)
        else:
            # Sequential processing for small batches
            for args in tqdm(args_list, desc=f"Processing {source_name}"):
                result = process_single_audio_file(args)
                if result:
                    results.append(result)

        return results

    def train_models(self, df_results):
        """Train the AI detection models."""
        print("TRAINING")
        print("=" * 50)

        # Prepare features
        metadata_columns = [
            "filename",
            "full_path",
            "source_directory",
            "file_extension",
            "is_ai",
            "spectrogram_path",
        ]
        self.feature_columns = [
            col for col in df_results.columns if col not in metadata_columns
        ]

        print(f"Using {len(self.feature_columns)} features")

        # Prepare data
        X = df_results[self.feature_columns].fillna(0)
        y = df_results["is_ai"].astype(int)

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )

        print(f"Training set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")

        # Train all models
        results = {}
        all_models = {**self.incremental_models, **self.batch_models}

        for name, model in all_models.items():
            print(f"\nTraining {name}...")

            if hasattr(model, "partial_fit"):
                # For incremental models
                model.partial_fit(X_train, y_train, classes=[0, 1])
            else:
                # For batch models
                model.fit(X_train, y_train)

            # Evaluate
            train_accuracy = model.score(X_train, y_train)
            test_accuracy = model.score(X_test, y_test)

            results[name] = {
                "train_accuracy": train_accuracy,
                "test_accuracy": test_accuracy,
            }

            print(f"  Train accuracy: {train_accuracy:.3f}")
            print(f"  Test accuracy: {test_accuracy:.3f}")

        self.is_trained = True

        # Save models
        self.save_models()

        # Record training session
        session_info = {
            "timestamp": datetime.now().isoformat(),
            "total_samples": len(df_results),
            "ai_samples": int(y.sum()),
            "human_samples": int((~y.astype(bool)).sum()),
            "features_used": len(self.feature_columns),
            "test_accuracies": {
                name: info["test_accuracy"] for name, info in results.items()
            },
            "training_type": "initial",
        }
        self.training_history.append(session_info)

        return results

    def update_with_new_data(self, df_new, retrain_batch_models=False):
        """Adaptively update models with new data - KEY ADAPTIVE FEATURE."""
        print("ADAPTIVE MODEL UPDATE")
        print("=" * 40)

        if not self.is_trained:
            print("No trained models found. Train initial models first.")
            return None

        # Handle new features
        metadata_columns = [
            "filename",
            "full_path",
            "source_directory",
            "file_extension",
            "is_ai",
            "spectrogram_path",
        ]
        new_feature_cols = [
            col for col in df_new.columns if col not in metadata_columns
        ]

        # Check for missing features in new data
        missing_features = set(self.feature_columns) - set(new_feature_cols)
        if missing_features:
            print(f"Adding {len(missing_features)} missing features with zero values")
            for feature in missing_features:
                df_new[feature] = 0

        # Prepare new data
        X_new = df_new[self.feature_columns].fillna(0)
        y_new = df_new["is_ai"].astype(int)

        # Validate data has both classes if retraining batch models
        unique_classes = y_new.unique()
        print(f"New data classes: {unique_classes}")

        if retrain_batch_models and len(unique_classes) < 2:
            print(
                f"WARNING: New data only contains {len(unique_classes)} class(es). Cannot retrain batch models."
            )
            print(f"Available classes: {unique_classes}")
            print(
                "TIP: For batch model retraining, you need both AI and Human samples in the new data."
            )
            retrain_batch_models = False

        # Transform with existing scaler
        X_new_scaled = self.scaler.transform(X_new)

        print(
            f"Updating with {len(y_new)} samples ({y_new.sum()} AI, {(~y_new.astype(bool)).sum()} Human)"
        )

        # Update incremental models (these can handle single-class updates)
        update_results = {}
        for name, model in self.incremental_models.items():
            try:
                if model is None:
                    print(f"WARNING: {name} model is None, skipping update")
                    continue

                # Use partial_fit for incremental learning
                model.partial_fit(X_new_scaled, y_new)

                # Evaluate performance on new data
                accuracy = model.score(X_new_scaled, y_new)
                update_results[name] = {"accuracy": accuracy}
                print(f"Updated {name} - Accuracy on new data: {accuracy:.3f}")

            except Exception as e:
                print(f"Failed to update {name}: {e}")

        # Only retrain batch models if we have both classes
        if retrain_batch_models:
            print("Retraining batch models...")

            # Double-check we still have both classes after any data processing
            if len(unique_classes) >= 2:
                for name, model in self.batch_models.items():
                    try:
                        if model is None:
                            print(f"WARNING: {name} model is None, reinitializing")
                            if name == "RandomForest":
                                model = RandomForestClassifier(
                                    n_estimators=200, random_state=42, n_jobs=-1
                                )
                            elif name == "GradientBoosting":
                                model = GradientBoostingClassifier(
                                    n_estimators=200, random_state=42
                                )
                            self.batch_models[name] = model

                        # Additional validation for batch models
                        if len(X_new_scaled) < 2:
                            print(
                                f"Not enough samples to retrain {name} (need at least 2)"
                            )
                            continue

                        # Check class distribution more thoroughly
                        class_counts = np.bincount(y_new)
                        available_classes = np.where(class_counts > 0)[0]

                        if len(available_classes) < 2:
                            print(
                                f"Only {len(available_classes)} class(es) available for {name}, skipping"
                            )
                            continue

                        min_class_size = np.min(class_counts[class_counts > 0])

                        # Special handling for GradientBoosting
                        if name == "GradientBoosting":
                            if len(X_new_scaled) < 10:
                                print(
                                    f"Warning: Small sample size ({len(X_new_scaled)}) for {name}"
                                )

                            if min_class_size < 2:
                                print(
                                    f"Insufficient samples in minority class for {name} (min: {min_class_size})"
                                )
                                print(
                                    "TIP: GradientBoosting needs at least 2 samples per class for retraining"
                                )
                                continue

                        # Fit the model
                        model.fit(X_new_scaled, y_new)
                        accuracy = model.score(X_new_scaled, y_new)
                        update_results[name] = {"accuracy": accuracy}
                        print(f"Retrained {name} - Accuracy: {accuracy:.3f}")

                    except Exception as e:
                        print(f"Failed to retrain {name}: {e}")
                        # Try to reinitialize the model if it got corrupted
                        try:
                            if name == "RandomForest":
                                self.batch_models[name] = RandomForestClassifier(
                                    n_estimators=200, random_state=42, n_jobs=-1
                                )
                            elif name == "GradientBoosting":
                                self.batch_models[name] = GradientBoostingClassifier(
                                    n_estimators=200, random_state=42
                                )
                            print(
                                f"Reinitialized {name} model (will use previous training)"
                            )
                        except Exception as reinit_error:
                            print(f"Failed to reinitialize {name}: {reinit_error}")
            else:
                print("Cannot retrain batch models: insufficient class diversity")

        # Show update summary
        if update_results:
            print(f"\nUPDATE SUMMARY:")
            for model_name, result in update_results.items():
                print(f"   {model_name}: accuracy = {result['accuracy']:.3f}")
        else:
            print("\nNo models were successfully updated")

        # Record update session
        session_info = {
            "timestamp": datetime.now().isoformat(),
            "total_samples": len(df_new),
            "ai_samples": int(y_new.sum()),
            "human_samples": int((~y_new.astype(bool)).sum()),
            "features_used": len(self.feature_columns),
            "update_accuracies": update_results,
            "training_type": "adaptive_update",
            "retrained_batch": retrain_batch_models and len(unique_classes) >= 2,
            "classes_in_new_data": unique_classes.tolist(),
        }
        self.training_history.append(session_info)

        # Save updated models
        self.save_models()

        return update_results

    def predict_file(self, audio_path, return_details=False):
        """Predict if a single audio file is AI-generated."""
        if not self.is_trained:
            if not self.load_models():
                print("No trained models found. Train first.")
                return None

        # Extract features
        analyzer = AudioAnalyzer()
        features = analyzer.analyze_audio_file(audio_path)
        if not features:
            return None

        # Prepare feature vector
        feature_dict = {}
        for col in self.feature_columns:
            # Map feature names
            base_name = col
            for prefix in ["benford_", "spectral_", "temporal_", "compression_"]:
                if col.startswith(prefix):
                    base_name = col[len(prefix) :]
                    break

            feature_dict[col] = features.get(base_name, features.get(col, 0))

        X = pd.DataFrame([feature_dict])
        X_scaled = self.scaler.transform(X.values)

        # Get predictions from all models
        predictions = {}
        all_models = {**self.incremental_models, **self.batch_models}

        for name, model in all_models.items():
            try:
                pred = model.predict(X_scaled)[0]
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(X_scaled)[0]
                    ai_probability = proba[1] if len(proba) > 1 else 0.5
                    confidence = max(proba)
                else:
                    ai_probability = float(pred)
                    confidence = 0.8

                predictions[name] = {
                    "prediction": bool(pred),
                    "ai_probability": ai_probability,
                    "confidence": confidence,
                }
            except Exception as e:
                print(f"{name} prediction failed: {e}")

        if not predictions:
            return None

        # Ensemble prediction
        avg_ai_prob = np.mean([p["ai_probability"] for p in predictions.values()])
        ensemble_prediction = avg_ai_prob > 0.5
        ensemble_confidence = np.mean([p["confidence"] for p in predictions.values()])

        result = {
            "filename": Path(audio_path).name,
            "prediction": ensemble_prediction,
            "ai_probability": avg_ai_prob,
            "confidence": ensemble_confidence,
            "is_ai": ensemble_prediction,
        }

        if return_details:
            result["individual_predictions"] = predictions
            result["audio_duration"] = features.get("audio_duration", 0)
            result["sample_rate"] = features.get("sample_rate", 0)

        return result

    def predict_batch(
        self, audio_dir, output_file=None, max_workers=None, use_multiprocessing=True
    ):
        """Predict AI vs Human for all files in a directory with multiprocessing."""
        audio_path = Path(audio_dir)
        if not audio_path.exists():
            print(f"Directory {audio_dir} not found")
            return None

        if not self.is_trained:
            if not self.load_models():
                print("No trained models found.")
                return None

        # Use config for max_workers if not specified
        if max_workers is None:
            max_workers = self.config["processing"]["max_workers"]

        # Find audio files using config
        audio_extensions = self.get_audio_extensions()
        audio_files = [
            f for f in audio_path.iterdir() if f.suffix.lower() in audio_extensions
        ]

        if not audio_files:
            print(f"No audio files found in {audio_dir}")
            return None

        print(f"Predicting {len(audio_files)} audio files...")

        batch_threshold = self.config["processing"]["batch_threshold"]
        if use_multiprocessing and len(audio_files) > batch_threshold:
            results = self._predict_batch_multiprocessing(audio_files, max_workers)
        else:
            results = self._predict_batch_sequential(audio_files)

        if results:
            df_results = pd.DataFrame(results)

            # Show summary
            ai_count = df_results["is_ai"].sum()
            human_count = len(df_results) - ai_count
            avg_confidence = df_results["confidence"].mean()

            print(f"PREDICTION SUMMARY:")
            print(f"  AI files: {ai_count} ({100*ai_count/len(df_results):.1f}%)")
            print(
                f"  Human files: {human_count} ({100*human_count/len(df_results):.1f}%)"
            )
            print(f"  Average confidence: {avg_confidence:.3f}")

            # Show uncertain predictions
            uncertain_threshold = 0.6
            uncertain_files = df_results[df_results["confidence"] < uncertain_threshold]
            if len(uncertain_files) > 0:
                print(
                    f"  Uncertain predictions: {len(uncertain_files)} ({100*len(uncertain_files)/len(df_results):.1f}%)"
                )

            # Save results
            if output_file:
                output_path = self.results_dir / output_file
                df_results.to_csv(output_path, index=False)
                print(f"\nResults saved to: {output_path}")

            return df_results

        return None

    def _predict_batch_sequential(self, audio_files):
        """Sequential batch prediction."""
        results = []
        for audio_file in tqdm(audio_files, desc="Predicting (sequential)"):
            result = self.predict_file(audio_file, return_details=True)
            if result:
                result["full_path"] = str(audio_file)
                result["file_size_mb"] = audio_file.stat().st_size / (1024 * 1024)
                results.append(result)
        return results

    def _predict_batch_multiprocessing(self, audio_files, max_workers):
        """Multiprocessing batch prediction - NOW IMPLEMENTED."""
        print(f"Using {max_workers} workers for parallel prediction...")

        # Prepare data for multiprocessing
        scaler_data = {
            "mean_": self.scaler.mean_,
            "scale_": self.scaler.scale_,
            "var_": self.scaler.var_,
            "n_features_in_": self.scaler.n_features_in_,
            "n_samples_seen_": self.scaler.n_samples_seen_,
        }

        all_models = {**self.incremental_models, **self.batch_models}

        # Prepare arguments for each file
        args_list = [
            (audio_file, self.feature_columns, scaler_data, all_models)
            for audio_file in audio_files
        ]

        results = []
        successful_files = 0
        failed_files = 0

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all jobs
            future_to_file = {
                executor.submit(process_single_prediction, args): args[0]
                for args in args_list
            }

            # Process completed jobs with progress bar
            with tqdm(total=len(audio_files), desc="Predicting (parallel)") as pbar:
                for future in as_completed(future_to_file):
                    audio_file = future_to_file[future]
                    try:
                        result = future.result()
                        if result is not None:
                            results.append(result)
                            successful_files += 1
                        else:
                            failed_files += 1
                    except Exception as e:
                        failed_files += 1
                        print(f"Error processing {audio_file.name}: {e}")

                    pbar.update(1)
                    pbar.set_postfix(
                        {"Success": successful_files, "Failed": failed_files}
                    )

        print(
            f"Parallel prediction complete: {successful_files} success, {failed_files} failed"
        )
        return results

    def save_models(self):
        """Save trained models."""
        model_data = {
            "incremental_models": self.incremental_models,
            "batch_models": self.batch_models,
            "feature_columns": self.feature_columns,
            "scaler": self.scaler,
            "training_history": self.training_history,
            "timestamp": datetime.now().isoformat(),
        }
        joblib.dump(model_data, self.model_dir / "ai_audio_detector.joblib")
        print(f"Models saved to {self.model_dir}")

    def load_models(self):
        """Load previously trained models."""
        model_file = self.model_dir / "ai_audio_detector.joblib"
        if model_file.exists():
            try:
                model_data = joblib.load(model_file)

                # Handle both old and new model formats
                if "incremental_models" in model_data:
                    self.incremental_models = model_data["incremental_models"]
                    self.batch_models = model_data["batch_models"]
                else:
                    # Legacy format - convert
                    old_models = model_data["models"]
                    self.incremental_models = {
                        k: v
                        for k, v in old_models.items()
                        if k in ["SGD", "PassiveAggressive"]
                    }
                    self.batch_models = {
                        k: v
                        for k, v in old_models.items()
                        if k in ["RandomForest", "GradientBoosting"]
                    }

                # Validate loaded models
                for name, model in self.incremental_models.items():
                    if model is None:
                        print(
                            f"WARNING: Incremental model {name} is None, reinitializing"
                        )
                        if name == "SGD":
                            self.incremental_models[name] = SGDClassifier(
                                random_state=42, loss="log_loss"
                            )
                        elif name == "PassiveAggressive":
                            self.incremental_models[name] = PassiveAggressiveClassifier(
                                random_state=42
                            )

                for name, model in self.batch_models.items():
                    if model is None:
                        print(f"WARNING: Batch model {name} is None, reinitializing")
                        if name == "RandomForest":
                            self.batch_models[name] = RandomForestClassifier(
                                n_estimators=200, random_state=42, n_jobs=-1
                            )
                        elif name == "GradientBoosting":
                            self.batch_models[name] = GradientBoostingClassifier(
                                n_estimators=200, random_state=42
                            )

                self.feature_columns = model_data["feature_columns"]
                self.scaler = model_data["scaler"]
                self.training_history = model_data.get("training_history", [])
                self.is_trained = True
                print(f"Loaded models with {len(self.feature_columns)} features")
                return True
            except Exception as e:
                print(f"Error loading models: {e}")
                return False
        return False

    def create_visualizations(self, df_results, save_dir=None):
        """Create visualization plots."""
        if save_dir is None:
            save_dir = self.base_dir
        else:
            save_dir = Path(save_dir)

        # Basic plot style
        plt.style.use("default")
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("AI Audio Detection Analysis", fontsize=16)

        # 1. Distribution of AI vs Human files
        ax1 = axes[0, 0]
        counts = df_results["is_ai"].value_counts()
        labels = ["Human", "AI"]
        colors = ["lightblue", "lightcoral"]
        ax1.bar(labels, [counts[False], counts[True]], color=colors)
        ax1.set_title("Distribution of Audio Files")
        ax1.set_ylabel("Count")

        # 2. Audio duration distribution
        ax2 = axes[0, 1]
        if "audio_duration" in df_results.columns:
            ai_durations = df_results[df_results["is_ai"]]["audio_duration"]
            human_durations = df_results[~df_results["is_ai"]]["audio_duration"]

            ax2.hist(
                human_durations, alpha=0.7, label="Human", bins=20, color="lightblue"
            )
            ax2.hist(ai_durations, alpha=0.7, label="AI", bins=20, color="lightcoral")
            ax2.set_title("Audio Duration Distribution")
            ax2.set_xlabel("Duration (seconds)")
            ax2.set_ylabel("Frequency")
            ax2.legend()

        # 3. Feature importance (if available)
        ax3 = axes[1, 0]
        if "RandomForest" in self.batch_models and hasattr(
            self.batch_models["RandomForest"], "feature_importances_"
        ):
            importances = self.batch_models["RandomForest"].feature_importances_
            indices = np.argsort(importances)[-10:]  # Top 10 features

            feature_names = [self.feature_columns[i] for i in indices]
            ax3.barh(range(len(indices)), importances[indices])
            ax3.set_yticks(range(len(indices)))
            ax3.set_yticklabels(
                [name[:20] for name in feature_names]
            )  # Truncate long names
            ax3.set_title("Top 10 Most Important Features (Random Forest Only)")
            ax3.set_xlabel("Importance")

        # 4. Training history
        ax4 = axes[1, 1]
        if self.training_history:
            sessions = []
            accuracies = []
            for i, session in enumerate(self.training_history):
                sessions.append(f"Session {i+1}")
                if "test_accuracies" in session:
                    best_acc = max(session["test_accuracies"].values())
                elif "update_accuracies" in session:
                    best_acc = max(
                        [v["accuracy"] for v in session["update_accuracies"].values()]
                        + [0]
                    )
                else:
                    best_acc = 0.5
                accuracies.append(best_acc)

            ax4.plot(sessions, accuracies, "o-", color="green")
            ax4.set_title("Model Performance Over Time")
            ax4.set_ylabel("Best Accuracy")
            ax4.set_ylim(0, 1)
            plt.setp(ax4.get_xticklabels(), rotation=45, ha="right")

        # This looks better in my opinion
        plt.tight_layout()

        # Save plot
        plot_file = save_dir / "ai_detection_analysis.png"
        plt.savefig(plot_file, dpi=300, bbox_inches="tight")
        print(f"Visualization saved to: {plot_file}")
        plt.show()

    def add_ai_data(self, ai_audio_dir, retrain_batch_models=False):
        """Add new AI-generated audio data to improve detection."""
        print("ADDING NEW AI DATA")
        print("=" * 40)

        if not self.is_trained:
            print("No trained models found. Train initial models first.")
            return None

        # Extract features from AI directory
        print(f"Processing AI data from: {ai_audio_dir}")
        ai_features = self.extract_features_from_directory(
            ai_audio_dir, is_ai_directory=True
        )

        if not ai_features:
            print("No AI features extracted")
            return None

        df_ai = pd.DataFrame(ai_features)
        print(f"Adding {len(df_ai)} new AI samples")

        # Update models with new AI data
        update_results = self.update_with_new_data(df_ai, retrain_batch_models)

        if update_results:
            print(f"\nAI DATA ADDED SUCCESSFULLY!")
            print(f"   Models updated with {len(df_ai)} AI samples")
            for model_name, result in update_results.items():
                print(f"   {model_name}: accuracy = {result['accuracy']:.3f}")

        return update_results

    def add_human_data(self, human_audio_dir, retrain_batch_models=False):
        """Add new human audio data to improve detection."""
        print("ADDING NEW HUMAN DATA")
        print("=" * 42)

        if not self.is_trained:
            print("No trained models found. Train initial models first.")
            return None

        # Extract features from human directory
        print(f"Processing human data from: {human_audio_dir}")
        human_features = self.extract_features_from_directory(
            human_audio_dir, is_ai_directory=False
        )

        if not human_features:
            print("No human features extracted")
            return None

        df_human = pd.DataFrame(human_features)
        print(f"Adding {len(df_human)} new human samples")

        # Update models with new human data
        update_results = self.update_with_new_data(df_human, retrain_batch_models)

        if update_results:
            print(f"\nHUMAN DATA ADDED SUCCESSFULLY!")
            print(f"   Models updated with {len(df_human)} human samples")
            for model_name, result in update_results.items():
                print(f"   {model_name}: accuracy = {result['accuracy']:.3f}")

        return update_results

    def add_mixed_data_batch(self, data_directories, retrain_batch_models=False):
        """Add multiple directories of mixed AI/Human data at once."""
        print("ADDING BATCH OF MIXED DATA")
        print("=" * 45)

        if not self.is_trained:
            print("No trained models found. Train initial models first.")
            return None

        all_new_features = []

        for dir_info in data_directories:
            dir_path = dir_info["path"]
            is_ai_dir = dir_info["is_ai"]
            label = "AI" if is_ai_dir else "Human"

            print(f"\nProcessing {label} data from: {dir_path}")

            if not Path(dir_path).exists():
                print(f"Directory {dir_path} not found, skipping...")
                continue

            features = self.extract_features_from_directory(
                dir_path, is_ai_directory=is_ai_dir
            )

            if features:
                all_new_features.extend(features)
                print(f"   Added {len(features)} {label} samples")
            else:
                print(f"   No features extracted from {dir_path}")

        if not all_new_features:
            print("No new features extracted from any directory")
            return None

        df_all_new = pd.DataFrame(all_new_features)
        ai_count = df_all_new["is_ai"].sum()
        human_count = len(df_all_new) - ai_count

        print(f"BATCH SUMMARY:")
        print(f"   Total new samples: {len(df_all_new)}")
        print(f"   AI samples: {ai_count}")
        print(f"   Human samples: {human_count}")

        # Update models
        update_results = self.update_with_new_data(df_all_new, retrain_batch_models)

        if update_results:
            print(f"\nBATCH DATA ADDED SUCCESSFULLY!")
            for model_name, result in update_results.items():
                print(f"   {model_name}: accuracy = {result['accuracy']:.3f}")

        return update_results

    def show_data_balance(self):
        """Show current balance of AI vs Human data in training history."""
        print("DATA BALANCE ANALYSIS")
        print("=" * 40)

        if not self.training_history:
            print("No training history available")
            return

        total_ai = 0
        total_human = 0

        print("Training sessions:")
        for i, session in enumerate(self.training_history):
            ai_samples = session.get("ai_samples", 0)
            human_samples = session.get("human_samples", 0)
            session_type = session.get("training_type", "unknown")

            total_ai += ai_samples
            total_human += human_samples

            print(
                f"  Session {i+1} ({session_type}): AI={ai_samples}, Human={human_samples}"
            )

        total_samples = total_ai + total_human
        if total_samples > 0:
            ai_percentage = (total_ai / total_samples) * 100
            human_percentage = (total_human / total_samples) * 100

            print(f"CUMULATIVE TOTALS:")
            print(f"   AI samples: {total_ai} ({ai_percentage:.1f}%)")
            print(f"   Human samples: {total_human} ({human_percentage:.1f}%)")
            print(f"   Total samples: {total_samples}")

            # Balance recommendations
            if ai_percentage < 40:
                print(
                    f"\nRECOMMENDATION: Add more AI data (currently only {ai_percentage:.1f}%)"
                )
            elif ai_percentage > 60:
                print(
                    f"\nRECOMMENDATION: Add more Human data (AI is {ai_percentage:.1f}%)"
                )
            else:
                print(f"\nGood balance between AI and Human data")

    def generate_spectrograms_batch(
        self,
        audio_dir,
        output_dir,
        spectrogram_type="mel",
        max_workers=4,
        figsize=(12, 8),
        dpi=300,
    ):
        """
        Generate spectrograms for all audio files in a directory.

        Args:
            audio_dir: Directory containing audio files
            output_dir: Directory to save spectrograms
            spectrogram_type: Type of spectrogram ('mel', 'stft', 'cqt', 'chroma', 'mfcc')
            max_workers: Number of workers for parallel processing
            figsize: Figure size for spectrograms
            dpi: DPI for saved images

        Returns:
            List of generated spectrogram paths
        """
        audio_path = Path(audio_dir)
        output_path = Path(output_dir)

        if not audio_path.exists():
            print(f"Audio directory {audio_dir} not found")
            return []

        output_path.mkdir(parents=True, exist_ok=True)

        # Find audio files
        audio_extensions = (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac")
        audio_files = [
            f for f in audio_path.iterdir() if f.suffix.lower() in audio_extensions
        ]

        if not audio_files:
            print(f"No audio files found in {audio_dir}")
            return []

        print(
            f"Generating {spectrogram_type} spectrograms for {len(audio_files)} files..."
        )

        analyzer = AudioAnalyzer()
        generated_paths = []

        if len(audio_files) > 3 and max_workers > 1:
            # Use parallel processing for large batches
            generate_func = functools.partial(
                self._generate_single_spectrogram,
                output_dir=output_path,
                spectrogram_type=spectrogram_type,
                figsize=figsize,
                dpi=dpi,
            )

            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                future_to_file = {
                    executor.submit(generate_func, audio_file): audio_file
                    for audio_file in audio_files
                }

                with tqdm(
                    total=len(audio_files),
                    desc=f"Generating {spectrogram_type} spectrograms",
                ) as pbar:
                    for future in as_completed(future_to_file):
                        try:
                            result_path = future.result()
                            if result_path:
                                generated_paths.append(result_path)
                        except Exception as e:
                            audio_file = future_to_file[future]
                            print(
                                f"Error generating spectrogram for {audio_file.name}: {e}"
                            )
                        pbar.update(1)
        else:
            # Sequential processing for small batches
            for audio_file in tqdm(
                audio_files, desc=f"Generating {spectrogram_type} spectrograms"
            ):
                result_path = analyzer.save_spectrogram(
                    audio_file, output_path, spectrogram_type, figsize, dpi
                )
                if result_path:
                    generated_paths.append(result_path)

        print(f"Generated {len(generated_paths)} spectrograms in {output_path}")
        return generated_paths

    @staticmethod
    def _generate_single_spectrogram(
        audio_file, output_dir, spectrogram_type, figsize, dpi
    ):
        """Helper function for parallel spectrogram generation."""
        try:
            analyzer = AudioAnalyzer()
            return analyzer.save_spectrogram(
                audio_file, output_dir, spectrogram_type, figsize, dpi
            )
        except Exception as e:
            print(
                f"Error in parallel spectrogram generation for {audio_file.name}: {e}"
            )
            return None

    def create_spectrogram_comparison(
        self, ai_dir, human_dir, output_dir, spectrogram_type="mel", num_samples=5
    ):
        """
        Create side-by-side comparison of AI vs Human spectrograms.

        Args:
            ai_dir: Directory containing AI-generated audio
            human_dir: Directory containing human-generated audio
            output_dir: Directory to save comparison images
            spectrogram_type: Type of spectrogram to generate
            num_samples: Number of sample pairs to compare
        """
        ai_path = Path(ai_dir)
        human_path = Path(human_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Find audio files
        audio_extensions = (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac")
        ai_files = [
            f for f in ai_path.iterdir() if f.suffix.lower() in audio_extensions
        ]
        human_files = [
            f for f in human_path.iterdir() if f.suffix.lower() in audio_extensions
        ]

        if not ai_files or not human_files:
            print("Need audio files in both directories for comparison")
            return

        # Sample files for comparison
        ai_sample = random.sample(ai_files, min(num_samples, len(ai_files)))
        human_sample = random.sample(human_files, min(num_samples, len(human_files)))

        print(f"Creating {spectrogram_type} spectrogram comparisons...")

        analyzer = AudioAnalyzer()

        for i, (ai_file, human_file) in enumerate(zip(ai_sample, human_sample)):
            try:
                # Create comparison figure
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

                # Load audio files
                y_ai, sr_ai = librosa.load(ai_file, sr=None)
                y_human, sr_human = librosa.load(human_file, sr=None)

                # Generate spectrograms
                if spectrogram_type.lower() == "mel":
                    S_ai = librosa.feature.melspectrogram(y=y_ai, sr=sr_ai, n_mels=128)
                    S_ai_dB = librosa.power_to_db(S_ai, ref=np.max)
                    S_human = librosa.feature.melspectrogram(
                        y=y_human, sr=sr_human, n_mels=128
                    )
                    S_human_dB = librosa.power_to_db(S_human, ref=np.max)

                    img1 = librosa.display.specshow(
                        S_ai_dB,
                        x_axis="time",
                        y_axis="mel",
                        sr=sr_ai,
                        fmax=8000,
                        ax=ax1,
                    )
                    img2 = librosa.display.specshow(
                        S_human_dB,
                        x_axis="time",
                        y_axis="mel",
                        sr=sr_human,
                        fmax=8000,
                        ax=ax2,
                    )

                elif spectrogram_type.lower() == "stft":
                    D_ai = librosa.stft(y_ai)
                    S_ai_dB = librosa.amplitude_to_db(np.abs(D_ai), ref=np.max)
                    D_human = librosa.stft(y_human)
                    S_human_dB = librosa.amplitude_to_db(np.abs(D_human), ref=np.max)

                    img1 = librosa.display.specshow(
                        S_ai_dB, x_axis="time", y_axis="hz", sr=sr_ai, ax=ax1
                    )
                    img2 = librosa.display.specshow(
                        S_human_dB, x_axis="time", y_axis="hz", sr=sr_human, ax=ax2
                    )

                # Set titles and labels
                ax1.set_title(f"AI Generated - {ai_file.name}", fontsize=14)
                ax2.set_title(f"Human Generated - {human_file.name}", fontsize=14)

                ax1.set_xlabel("Time (s)")
                ax2.set_xlabel("Time (s)")

                # Add colorbars
                plt.colorbar(img1, ax=ax1, format="%+2.0f dB")
                plt.colorbar(img2, ax=ax2, format="%+2.0f dB")

                plt.tight_layout()

                # Save comparison
                comparison_filename = f"comparison_{i+1}_{spectrogram_type}.png"
                comparison_path = output_path / comparison_filename
                plt.savefig(comparison_path, dpi=300, bbox_inches="tight")
                plt.close()

                print(f"Saved comparison {i+1}: {comparison_filename}")

            except Exception as e:
                print(f"Error creating comparison {i+1}: {e}")
                plt.close()

        print(f"Comparison spectrograms saved to {output_path}")


def main():
    """Main interface for the AI audio detector."""
    parser = argparse.ArgumentParser(
        description="AI Audio Detector - Detect AI-generated audio using machine learning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  ai-audio-detector --interactive
  ai-audio-detector --predict-file audio.wav
  ai-audio-detector --predict-batch /path/to/audio/files
  ai-audio-detector --train --human-dir /path/to/human --ai-dir /path/to/ai
        """,
    )

    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Run in interactive mode (default)",
    )
    parser.add_argument("--config", "-c", type=str, help="Path to configuration file")
    parser.add_argument(
        "--base-dir",
        "-d",
        type=str,
        default=None,
        help="Base directory for models and outputs",
    )

    # Prediction options
    parser.add_argument(
        "--predict-file", "-f", type=str, help="Predict a single audio file"
    )
    parser.add_argument(
        "--predict-batch", "-b", type=str, help="Predict all files in a directory"
    )
    parser.add_argument(
        "--output", "-o", type=str, help="Output CSV file for batch predictions"
    )

    # Training options
    parser.add_argument("--train", "-t", action="store_true", help="Train new models")
    parser.add_argument(
        "--human-dir", type=str, help="Directory containing human-generated audio"
    )
    parser.add_argument(
        "--ai-dir", type=str, help="Directory containing AI-generated audio"
    )

    # Processing options
    parser.add_argument(
        "--max-workers",
        "-w",
        type=int,
        default=None,
        help="Maximum number of worker processes",
    )
    parser.add_argument(
        "--no-multiprocessing", action="store_true", help="Disable multiprocessing"
    )

    args = parser.parse_args()

    try:
        # Initialize detector
        detector = AIAudioDetector(base_dir=args.base_dir, config_path=args.config)

        # Handle command line operations
        if args.predict_file:
            if detector.load_models():
                result = detector.predict_file(args.predict_file, return_details=True)
                if result:
                    print(f"File: {result['filename']}")
                    print(
                        f"Prediction: {'AI GENERATED' if result['is_ai'] else 'HUMAN GENERATED'}"
                    )
                    print(f"AI Probability: {result['ai_probability']:.3f}")
                    print(f"Confidence: {result['confidence']:.3f}")
                else:
                    print("Could not analyze file")
                    sys.exit(1)
            else:
                print("No trained models found. Train models first.")
                sys.exit(1)
            return

        elif args.predict_batch:
            if detector.load_models():
                max_workers = args.max_workers
                use_mp = not args.no_multiprocessing
                results = detector.predict_batch(
                    args.predict_batch, args.output, max_workers, use_mp
                )
                if not results:
                    sys.exit(1)
            else:
                print("No trained models found. Train models first.")
                sys.exit(1)
            return

        elif args.train:
            if not args.human_dir or not args.ai_dir:
                print("Error: Both --human-dir and --ai-dir are required for training")
                sys.exit(1)

            if not Path(args.human_dir).exists() or not Path(args.ai_dir).exists():
                print("Error: One or both directories not found")
                sys.exit(1)

            print("Extracting features...")
            human_features = detector.extract_features_from_directory(
                args.human_dir, is_ai_directory=False
            )
            ai_features = detector.extract_features_from_directory(
                args.ai_dir, is_ai_directory=True
            )

            if not human_features or not ai_features:
                print("Error: Could not extract features from directories")
                sys.exit(1)

            all_features = human_features + ai_features
            df_results = pd.DataFrame(all_features)

            print(
                f"Training with {len(df_results)} files ({len(human_features)} human, {len(ai_features)} AI)"
            )
            training_results = detector.train_models(df_results)

            best_model = max(
                training_results.items(), key=lambda x: x[1]["test_accuracy"]
            )
            print(
                f"Training complete! Best model: {best_model[0]} (accuracy: {best_model[1]['test_accuracy']:.3f})"
            )

            # Save results
            results_file = detector.results_dir / "training_results.csv"
            df_results.to_csv(results_file, index=False)
            print(f"Training data saved to: {results_file}")
            return

        # If no command line options, run interactive mode
        run_interactive_mode(detector)

    except KeyboardInterrupt:
        print("\nExiting...")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def run_interactive_mode(detector):
    """Run the interactive command-line interface."""
    print("AI AUDIO DETECTION SYSTEM")
    print("=" * 55)
    print("1. Train new models from directories")
    print("2. Predict single file")
    print("3. Predict batch")
    print("4. Update models with new data")
    print("5. Add new AI data")
    print("6. Add new Human data")
    print("7. Add batch of directories")
    print("8. Training history (Post init. only)")
    print("9. Data balance")
    print("10. Create visualizations")
    print("11. Generate spectrograms for directory")
    print("12. Create AI vs Human spectrogram comparison")

    choice = input("\nChoose option (1-12): ").strip()

    if choice == "1":
        # Train new models
        print("\nTRAINING FROM DIRECTORIES")

        human_dir = input("Enter path to human audio directory: ").strip().strip("\"'")
        ai_dir = input("Enter path to AI audio directory: ").strip().strip("\"'")

        if not Path(human_dir).exists() or not Path(ai_dir).exists():
            print("One or both directories not found")
            return

        # Ask about spectrogram generation
        save_spectrograms = (
            input("\nSave spectrograms during training? (y/n): ").strip().lower() == "y"
        )
        spectrogram_dir = None
        spectrogram_type = "mel"

        if save_spectrograms:
            spectrogram_dir = input(
                "Enter directory to save spectrograms (or Enter for default): "
            ).strip()
            if not spectrogram_dir:
                spectrogram_dir = detector.spectrograms_dir
            else:
                spectrogram_dir = Path(spectrogram_dir)

            print("Choose spectrogram type:")
            print("1. Mel spectrogram (default)")
            print("2. STFT spectrogram")
            print("3. CQT spectrogram")
            print("4. Chromagram")
            print("5. MFCC")
            spect_choice = input("Choose (1-5): ").strip()

            spect_types = {
                "1": "mel",
                "2": "stft",
                "3": "cqt",
                "4": "chroma",
                "5": "mfcc",
            }
            spectrogram_type = spect_types.get(spect_choice, "mel")

            print(f"Will save {spectrogram_type} spectrograms to {spectrogram_dir}")

        # Extract features
        print("\nExtracting features...")
        human_features = detector.extract_features_from_directory(
            human_dir,
            is_ai_directory=False,
            save_spectrograms=save_spectrograms,
            spectrogram_dir=spectrogram_dir,
            spectrogram_type=spectrogram_type,
        )
        ai_features = detector.extract_features_from_directory(
            ai_dir,
            is_ai_directory=True,
            save_spectrograms=save_spectrograms,
            spectrogram_dir=spectrogram_dir,
            spectrogram_type=spectrogram_type,
        )

        if not human_features or not ai_features:
            print("Could not extract features from directories")
            return

        # Combine data
        all_features = human_features + ai_features
        df_results = pd.DataFrame(all_features)

        print(f"Dataset Summary:")
        print(f"Total files: {len(df_results)}")
        print(f"Human files: {len(human_features)}")
        print(f"AI files: {len(ai_features)}")

        # Train models
        training_results = detector.train_models(df_results)

        # Show results
        print(f"\nTRAINING COMPLETE!")
        best_model = max(training_results.items(), key=lambda x: x[1]["test_accuracy"])
        print(
            f"Best model: {best_model[0]} (accuracy: {best_model[1]['test_accuracy']:.3f})"
        )

        # Save comprehensive results
        results_file = detector.results_dir / "training_results.csv"
        df_results.to_csv(results_file, index=False)
        print(f"Training data saved to: {results_file}")

        # Create visualizations
        detector.create_visualizations(df_results)

    elif choice == "2":
        # Predict single file
        if detector.load_models():
            file_path = input("Enter audio file path: ").strip().strip("\"'")
            result = detector.predict_file(file_path, return_details=True)

            if result:
                print(f"PREDICTION RESULT:")
                print(f"File: {result['filename']}")
                print(
                    f"Prediction: {'AI GENERATED' if result['is_ai'] else 'HUMAN GENERATED'}"
                )
                print(f"AI Probability: {result['ai_probability']:.3f}")
                print(f"Confidence: {result['confidence']:.3f}")
                print(f"Duration: {result.get('audio_duration', 0):.1f}s")

                # Show individual model predictions
                if "individual_predictions" in result:
                    print(f"\nIndividual model predictions:")
                    for model_name, pred_info in result[
                        "individual_predictions"
                    ].items():
                        indicator = "AI" if pred_info["prediction"] else "Human"
                        print(
                            f"  {indicator} {model_name}: {pred_info['ai_probability']:.3f} (conf: {pred_info['confidence']:.3f})"
                        )
            else:
                print("Could not analyze file")

    elif choice == "3":
        # Predict batch
        if detector.load_models():
            dir_path = input("Enter directory path: ").strip().strip("\"'")
            output_file = input("Output CSV file (or Enter for none): ").strip()
            max_workers = input(
                "Max workers for multiprocessing (or Enter for 4): "
            ).strip()

            if not output_file:
                output_file = None
            if max_workers.isdigit():
                max_workers = int(max_workers)
            else:
                max_workers = 4

            results = detector.predict_batch(dir_path, output_file, max_workers)

    elif choice == "4":
        # Adaptive learning - update with new data
        if detector.load_models():
            print("ADAPTIVE LEARNING UPDATE")

            new_dir = (
                input("Enter directory with new audio files: ").strip().strip("\"'")
            )
            if not Path(new_dir).exists():
                print("Directory not found")
                return

            # Ask user to label the new data
            print("\nHow should these files be labeled?")
            print("1. All AI-generated")
            print("2. All Human-generated")
            print("3. Mixed (determine from filename/directory)")
            label_choice = input("Choose (1-3): ").strip()

            if label_choice == "1":
                is_ai = True
            elif label_choice == "2":
                is_ai = False
            else:
                # Mixed - try to determine from filename patterns
                is_ai = None  # Will be determined per file

            # Extract features from new data
            if is_ai is not None:
                new_features = detector.extract_features_from_directory(
                    new_dir, is_ai_directory=is_ai
                )
            else:
                # For mixed data, analyze each file individually
                new_features = []
                audio_extensions = detector.get_audio_extensions()
                audio_files = [
                    f
                    for f in Path(new_dir).iterdir()
                    if f.suffix.lower() in audio_extensions
                ]

                for audio_file in tqdm(audio_files, desc="Processing mixed data"):
                    # Check filename for AI indicators
                    keywords = (
                        input("Enter keywords to identify AI files (comma-separated): ")
                        .strip()
                        .split(",")
                    )
                    keywords = [k.strip().lower() for k in keywords if k.strip()]
                    filename_lower = audio_file.name.lower()
                    is_ai_file = any(keyword in filename_lower for keyword in keywords)

                    args = (audio_file, is_ai_file, "Mixed_Data")
                    result = process_single_audio_file(args)
                    if result:
                        new_features.append(result)

            if new_features:
                df_new = pd.DataFrame(new_features)
                print(f"New data: {len(df_new)} files")
                if "is_ai" in df_new.columns:
                    ai_count = df_new["is_ai"].sum()
                    print(f"AI: {ai_count}, Human: {len(df_new) - ai_count}")

                # Ask about retraining batch models
                retrain_batch = (
                    input("\nRetrain batch models too? (y/n): ").strip().lower() == "y"
                )

                # Update models
                update_results = detector.update_with_new_data(
                    df_new, retrain_batch_models=retrain_batch
                )

                if update_results:
                    print(f"UPDATE RESULTS:")
                    for model_name, result in update_results.items():
                        print(f"  {model_name}: accuracy = {result['accuracy']:.3f}")
            else:
                print("No features extracted from new data")

    elif choice == "5":
        # Add AI data only
        if detector.load_models():
            ai_dir = (
                input("Enter path to NEW AI audio directory: ").strip().strip("\"'")
            )
            if Path(ai_dir).exists():
                retrain_batch = (
                    input("Retrain batch models too? (y/n): ").strip().lower() == "y"
                )
                detector.add_ai_data(ai_dir, retrain_batch_models=retrain_batch)
            else:
                print("Directory not found")

    elif choice == "6":
        # Add Human data only
        if detector.load_models():
            human_dir = (
                input("Enter path to NEW Human audio directory: ").strip().strip("\"'")
            )
            if Path(human_dir).exists():
                retrain_batch = (
                    input("Retrain batch models too? (y/n): ").strip().lower() == "y"
                )
                detector.add_human_data(human_dir, retrain_batch_models=retrain_batch)
            else:
                print("Directory not found")

    elif choice == "7":
        # Add batch of mixed directories
        if detector.load_models():
            print("BATCH DIRECTORY ADDITION")
            print(
                "Enter directory paths and labels. Press Enter with empty path to finish."
            )

            directories = []
            while True:
                dir_path = (
                    input(
                        f"\nDirectory path #{len(directories)+1} (or Enter to finish): "
                    )
                    .strip()
                    .strip("\"'")
                )
                if not dir_path:
                    break

                if not Path(dir_path).exists():
                    print("Directory not found, skipping...")
                    continue

                print("Is this directory:")
                print("1. AI-generated audio")
                print("2. Human-generated audio")
                choice_label = input("Choose (1-2): ").strip()

                is_ai_dir = choice_label == "1"
                directories.append({"path": dir_path, "is_ai": is_ai_dir})

                label = "AI" if is_ai_dir else "Human"
                print(f"Added {dir_path} as {label} data")

            if directories:
                print(f"Will process {len(directories)} directories:")
                for i, dir_info in enumerate(directories):
                    label = "AI" if dir_info["is_ai"] else "Human"
                    print(f"  {i+1}. {dir_info['path']} ({label})")

                confirm = input("\nProceed? (y/n): ").strip().lower() == "y"
                if confirm:
                    retrain_batch = (
                        input("Retrain batch models too? (y/n): ").strip().lower()
                        == "y"
                    )
                    detector.add_mixed_data_batch(
                        directories, retrain_batch_models=retrain_batch
                    )
            else:
                print("No directories specified")

    elif choice == "8":
        # Show training history
        if detector.load_models():
            print(f"\nTRAINING HISTORY:")
            if not detector.training_history:
                print("No training history found.")
            else:
                for i, session in enumerate(detector.training_history):
                    print(f"\nSession {i+1}: {session['timestamp'][: 19]}")
                    print(f"  Type: {session.get('training_type', 'unknown')}")
                    print(
                        f"  Samples: {session['total_samples']} (AI: {session['ai_samples']}, Human: {session['human_samples']})"
                    )
                    print(f"  Features: {session['features_used']}")

                    if "test_accuracies" in session:
                        best_acc = max(session["test_accuracies"].values())
                        print(f"  Best test accuracy: {best_acc:.3f}")
                    elif "update_accuracies" in session:
                        if session["update_accuracies"]:
                            best_acc = max(
                                [
                                    v["accuracy"]
                                    for v in session["update_accuracies"].values()
                                ]
                            )
                            print(f"  Best update accuracy: {best_acc:.3f}")

    elif choice == "9":
        # Show data balance
        if detector.load_models():
            detector.show_data_balance()

    elif choice == "10":
        # Create visualizations
        if detector.load_models():
            results_file = detector.results_dir / "training_results.csv"
            if results_file.exists():
                df_results = pd.read_csv(results_file)
                detector.create_visualizations(df_results)
            else:
                print("No training results found. Train models first.")

    elif choice == "11":
        # Generate spectrograms for directory
        print("\nSPECTROGRAM GENERATION")

        audio_dir = input("Enter path to audio directory: ").strip().strip("\"'")
        if not Path(audio_dir).exists():
            print("Directory not found")
            return detector

        output_dir = (
            input("Enter output directory for spectrograms: ").strip().strip("\"'")
        )
        if not output_dir:
            output_dir = detector.spectrograms_dir
        else:
            output_dir = Path(output_dir)

        print("\nChoose spectrogram type:")
        print("1. Mel spectrogram (default)")
        print("2. STFT spectrogram")
        print("3. CQT spectrogram")
        print("4. Chromagram")
        print("5. MFCC")
        spect_choice = input("Choose (1-5): ").strip()

        spect_types = {"1": "mel", "2": "stft", "3": "cqt", "4": "chroma", "5": "mfcc"}
        spectrogram_type = spect_types.get(spect_choice, "mel")

        max_workers = input(
            "Max workers for parallel processing (or Enter for 4): "
        ).strip()
        if max_workers.isdigit():
            max_workers = int(max_workers)
        else:
            max_workers = 4

        generated_paths = detector.generate_spectrograms_batch(
            audio_dir, output_dir, spectrogram_type, max_workers
        )

        if generated_paths:
            print(f"\nGenerated {len(generated_paths)} spectrograms")
            print(f"Saved to: {output_dir}")

    elif choice == "12":
        # Create AI vs Human spectrogram comparison
        print("\nSPECTROGRAM COMPARISON")

        ai_dir = input("Enter path to AI audio directory: ").strip().strip("\"'")
        human_dir = input("Enter path to Human audio directory: ").strip().strip("\"'")

        if not Path(ai_dir).exists() or not Path(human_dir).exists():
            print("One or both directories not found")
            return detector

        output_dir = (
            input("Enter output directory for comparisons: ").strip().strip("\"'")
        )
        if not output_dir:
            output_dir = (
                detector.base_dir / detector.config["output"]["comparisons_dir"]
            )
        else:
            output_dir = Path(output_dir)

        print("\nChoose spectrogram type:")
        print("1. Mel spectrogram (default)")
        print("2. STFT spectrogram")
        spect_choice = input("Choose (1-2): ").strip()

        spect_types = {"1": "mel", "2": "stft"}
        spectrogram_type = spect_types.get(spect_choice, "mel")

        num_samples = input("Number of comparison pairs (or Enter for 5): ").strip()
        if num_samples.isdigit():
            num_samples = int(num_samples)
        else:
            num_samples = 5

        detector.create_spectrogram_comparison(
            ai_dir, human_dir, output_dir, spectrogram_type, num_samples
        )


if __name__ == "__main__":
    main()
# Test comment
