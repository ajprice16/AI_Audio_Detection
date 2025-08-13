"""
Core AI Audio Detector class and related utilities.
"""

from typing import Dict, List, Any, Optional, Tuple, Union, cast
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from datetime import datetime
from tqdm import tqdm
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

# ML imports
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier, PassiveAggressiveClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

from .config import load_config
from .audio_analyzer import AudioAnalyzer

warnings.filterwarnings("ignore", category=UserWarning)


def process_single_audio_file(args: Tuple) -> Dict[str, Any]:
    """
    Process a single audio file for feature extraction.

    Args:
        args: Tuple containing (file_path, is_ai_directory, config)

    Returns:
        Dictionary of extracted features.
    """
    file_path, is_ai_directory, config = args

    analyzer = AudioAnalyzer()
    features = analyzer.analyze_audio_file(file_path)
    features["is_ai"] = is_ai_directory

    return features


def process_single_prediction(args: Tuple) -> Dict[str, Any]:
    """
    Process a single audio file for prediction.

    Args:
        args: Tuple containing (file_path, detector)

    Returns:
        Dictionary containing prediction results.
    """
    file_path, detector = args
    result = detector.predict_single_file(file_path)
    return result  # type: ignore


class AIAudioDetector:
    """
    Main AI Audio Detector class.

    This class provides functionality for training machine learning models
    to detect AI-generated audio using various audio features including
    Benford's Law analysis.
    """

    def __init__(self, base_dir: Optional[Union[str, Path]] = None, config_path: Optional[Path] = None):
        """
        Initialize the AI Audio Detector.

        Args:
            base_dir: Base directory for storing models and results.
            config_path: Path to configuration file.
        """
        self.base_dir = Path(base_dir) if base_dir else Path.cwd()
        self.config = load_config(config_path)

        # Create output directories
        self.models_dir = self.base_dir / self.config["output"]["models_dir"]
        self.results_dir = self.base_dir / self.config["output"]["results_dir"]
        self.spectrograms_dir = self.base_dir / self.config["output"]["spectrograms_dir"]
        self.comparisons_dir = self.base_dir / self.config["output"]["comparisons_dir"]

        for directory in [self.models_dir, self.results_dir, self.spectrograms_dir, self.comparisons_dir]:
            directory.mkdir(parents=True, exist_ok=True)

        # Initialize models
        self.models: Dict[str, Any] = {}
        self.scaler = StandardScaler()
        self.feature_columns: List[str] = []
        self.is_trained = False
        self.training_history: List[Dict[str, Any]] = []  # Add training history

        # Initialize audio analyzer
        self.analyzer = AudioAnalyzer()

        # Load existing models if available
        self.load_models()

    def get_audio_extensions(self) -> List[str]:
        """
        Get supported audio file extensions.

        Returns:
            List of supported audio file extensions.
        """
        formats = self.config["audio"]["supported_formats"]
        return list(formats) if isinstance(formats, (list, tuple)) else []

    def extract_features_from_directory(
        self, directory: Union[str, Path], is_ai_directory: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Extract features from all audio files in a directory.

        Args:
            directory: Path to directory containing audio files.
            is_ai_directory: Whether the directory contains AI-generated audio.

        Returns:
            List of feature dictionaries.
        """
        directory = Path(directory)
        if not directory.exists():
            print(f"Directory does not exist: {directory}")
            return []

        # Find audio files
        audio_extensions = self.get_audio_extensions()
        audio_files: List[Path] = []
        for ext in audio_extensions:
            # Use rglob to search recursively without duplicates
            audio_files.extend(directory.rglob(f"*{ext}"))

        if not audio_files:
            print(f"No audio files found in {directory}")
            return []

        print(f"Processing {len(audio_files)} files from {directory}")

        # Process files
        features_list = []
        batch_threshold = self.config["processing"]["batch_threshold"]

        if len(audio_files) > batch_threshold:
            # Use multiprocessing for large batches
            max_workers = self.config["processing"]["max_workers"]
            args_list = [(file_path, is_ai_directory, self.config) for file_path in audio_files]

            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                with tqdm(total=len(audio_files), desc="Extracting features") as pbar:
                    futures = {executor.submit(process_single_audio_file, args): args[0] for args in args_list}

                    for future in as_completed(futures):
                        try:
                            features = future.result()
                            if "error" not in features:
                                features_list.append(features)
                        except Exception as e:
                            file_path = futures[future]
                            print(f"Error processing {file_path}: {e}")
                        finally:
                            pbar.update(1)
        else:
            # Process sequentially for small batches
            for file_path in tqdm(audio_files, desc="Extracting features"):
                features = self.analyzer.analyze_audio_file(file_path)
                features["is_ai"] = is_ai_directory
                if "error" not in features:
                    features_list.append(features)

        print(f"Successfully processed {len(features_list)} files")
        return features_list

    def train_models(self, df_results: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """
        Train machine learning models on the provided data.

        Args:
            df_results: DataFrame containing features and labels.

        Returns:
            Dictionary containing training results for each model.
        """
        print("Training models...")

        # Prepare feature matrix
        feature_cols = [
            col for col in df_results.columns if col not in ["filename", "full_path", "is_ai", "error", "duration"]
        ]

        if not feature_cols:
            raise ValueError("No feature columns found for training")

        self.feature_columns = feature_cols
        X = df_results[feature_cols].fillna(0)
        y = df_results["is_ai"].astype(int)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Initialize models
        model_configs = self.config["models"]

        self.models = {
            "random_forest": RandomForestClassifier(**model_configs["batch"]["random_forest"]),
            "gradient_boosting": GradientBoostingClassifier(**model_configs["batch"]["gradient_boosting"]),
            "sgd": SGDClassifier(**model_configs["incremental"]["sgd"]),
            "passive_aggressive": PassiveAggressiveClassifier(**model_configs["incremental"]["passive_aggressive"]),
        }

        # Train models and collect results
        training_results = {}

        for name, model in self.models.items():
            print(f"Training {name}...")

            try:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                accuracy = accuracy_score(y_test, y_pred)

                training_results[name] = {
                    "train_accuracy": accuracy,  # Test accuracy (on held-out set)
                    "test_accuracy": accuracy,  # Same as train_accuracy for compatibility
                    "model_type": type(model).__name__,
                    "feature_count": len(feature_cols),
                    "training_samples": len(X_train),
                    "test_samples": len(X_test),
                }

                print(f"{name} accuracy: {accuracy:.4f}")

            except Exception as e:
                print(f"Error training {name}: {e}")
                training_results[name] = {"error": str(e)}

        self.is_trained = True

        # Add to training history
        ai_count = df_results["is_ai"].sum()
        human_count = len(df_results) - ai_count
        total_count = len(df_results)

        training_entry = {
            "timestamp": datetime.now().isoformat(),
            "data_balance": {
                "total": total_count,
                "ai": ai_count,
                "human": human_count,
                "ratio": min(ai_count, human_count) / max(ai_count, human_count) if max(ai_count, human_count) > 0 else 0,
            },
            "feature_count": len(feature_cols),
            "models_trained": list(training_results.keys()),
        }
        self.training_history.append(training_entry)

        # Save models
        self.save_models()

        # Save training results
        results_file = self.results_dir / "training_results.csv"
        results_df = pd.DataFrame(training_results).T
        results_df.to_csv(results_file)

        return training_results

    def predict_single_file(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Predict whether a single audio file is AI-generated.

        Args:
            file_path: Path to the audio file.

        Returns:
            Dictionary containing prediction results.
        """
        if not self.is_trained:
            return {"error": "Models not trained. Please train models first."}

        file_path = Path(file_path)
        if not file_path.exists():
            return {"error": f"File not found: {file_path}"}

        # Extract features
        features = self.analyzer.analyze_audio_file(file_path)

        if "error" in features:
            return features

        # Prepare feature vector
        feature_vector = []
        for col in self.feature_columns:
            value = features.get(col, 0)
            # Handle numpy arrays and ensure we get scalar values
            if hasattr(value, "__iter__") and not isinstance(value, str):
                # If it's an array-like object, take the mean
                try:
                    value = float(np.mean(value))
                except:
                    value = 0.0
            else:
                try:
                    value = float(value)
                except:
                    value = 0.0
            feature_vector.append(value)

        X = np.array(feature_vector).reshape(1, -1)
        X_scaled = self.scaler.transform(X)

        # Get predictions from all models
        predictions = {}
        confidences = {}

        for name, model in self.models.items():
            try:
                pred = model.predict(X_scaled)[0]
                predictions[name] = bool(pred)

                # Get confidence if available
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(X_scaled)[0]
                    confidences[name] = float(max(proba))
                elif hasattr(model, "decision_function"):
                    decision = model.decision_function(X_scaled)[0]
                    confidences[name] = float(abs(decision))
                else:
                    confidences[name] = 1.0 if pred else 0.0

            except Exception as e:
                print(f"Error predicting with {name}: {e}")
                predictions[name] = False
                confidences[name] = 0.0

        # Ensemble prediction (majority vote with confidence weighting)
        valid_predictions = {k: v for k, v in predictions.items() if v is not None}

        if not valid_predictions:
            return {"error": "No valid predictions from models"}

        # Weighted vote
        ai_score = sum(confidences[name] for name, pred in valid_predictions.items() if pred)
        human_score = sum(confidences[name] for name, pred in valid_predictions.items() if not pred)

        total_score = ai_score + human_score
        final_confidence = max(ai_score, human_score) / total_score if total_score > 0 else 0.5
        is_ai_generated = ai_score > human_score

        return {
            "filename": file_path.name,
            "prediction": "AI-generated" if is_ai_generated else "Human-generated",
            "confidence": final_confidence,
            "ai_probability": ai_score / total_score if total_score > 0 else 0.5,
            "is_ai_generated": is_ai_generated,
            "model_predictions": predictions,
            "model_confidences": confidences,
            "features": features,
        }

    def predict_batch(self, directory: Union[str, Path]) -> List[Dict[str, Any]]:
        """
        Predict AI generation for all audio files in a directory.

        Args:
            directory: Path to directory containing audio files.

        Returns:
            List of prediction dictionaries.
        """
        directory = Path(directory)
        if not directory.exists():
            print(f"Directory does not exist: {directory}")
            return []

        # Find audio files
        audio_extensions = self.get_audio_extensions()
        audio_files: List[Path] = []
        for ext in audio_extensions:
            # Use rglob to search recursively without duplicates
            audio_files.extend(directory.rglob(f"*{ext}"))

        if not audio_files:
            print(f"No audio files found in {directory}")
            return []

        print(f"Predicting for {len(audio_files)} files")

        # Process files
        results = []
        batch_threshold = self.config["processing"]["batch_threshold"]

        if len(audio_files) > batch_threshold:
            # Use multiprocessing for large batches
            max_workers = self.config["processing"]["max_workers"]
            args_list = [(file_path, self) for file_path in audio_files]

            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                with tqdm(total=len(audio_files), desc="Making predictions") as pbar:
                    futures = {executor.submit(process_single_prediction, args): args[0] for args in args_list}

                    for future in as_completed(futures):
                        try:
                            result = future.result()
                            results.append(result)
                        except Exception as e:
                            file_path = futures[future]
                            print(f"Error predicting {file_path}: {e}")
                            results.append({"filename": file_path.name, "error": str(e)})
                        finally:
                            pbar.update(1)
        else:
            # Process sequentially for small batches
            for file_path in tqdm(audio_files, desc="Making predictions"):
                result = self.predict_single_file(file_path)
                results.append(result)

        return results

    def save_models(self) -> bool:
        """
        Save trained models to disk.

        Returns:
            True if successful, False otherwise.
        """
        try:
            model_data = {
                "models": self.models,
                "scaler": self.scaler,
                "feature_columns": self.feature_columns,
                "is_trained": self.is_trained,
                "training_history": self.training_history,
                "config": self.config,
                "timestamp": datetime.now().isoformat(),
            }

            model_file = self.models_dir / "ai_audio_detector.joblib"
            joblib.dump(model_data, model_file)
            print(f"Models saved to {model_file}")
            return True

        except Exception as e:
            print(f"Error saving models: {e}")
            return False

    def load_models(self) -> bool:
        """
        Load trained models from disk.

        Returns:
            True if successful, False otherwise.
        """
        try:
            model_file = self.models_dir / "ai_audio_detector.joblib"
            if not model_file.exists():
                return False

            model_data = joblib.load(model_file)

            # Handle both old and new model file formats
            if "models" in model_data:
                # New format
                self.models = model_data["models"]
            else:
                # Old format with separate incremental and batch models
                self.models = {}
                if "incremental_models" in model_data:
                    self.models.update(model_data["incremental_models"])
                if "batch_models" in model_data:
                    self.models.update(model_data["batch_models"])

            self.scaler = model_data["scaler"]
            self.feature_columns = model_data["feature_columns"]
            self.is_trained = True  # Set to True if we loaded models successfully

            # Load training history if available
            if "training_history" in model_data:
                self.training_history = model_data["training_history"]
            else:
                self.training_history = []  # Initialize empty if not found

            return True

        except Exception as e:
            print(f"Error loading models: {e}")
            return False

    def update_with_new_data(
        self, new_data: Union[List[Dict[str, Any]], pd.DataFrame], retrain_batch_models: bool = False
    ) -> Dict[str, Any]:
        """
        Update models with new training data using incremental learning.

        Args:
            new_data: List of new feature dictionaries or DataFrame.
            retrain_batch_models: Whether to retrain batch models (ignored for now).

        Returns:
            Dictionary containing update results.
        """
        if not self.is_trained:
            raise ValueError("Models must be trained before updating")

        if isinstance(new_data, list):
            if not new_data:
                return {"error": "No new features provided"}
            # Convert to DataFrame
            df_new = pd.DataFrame(new_data)
        else:
            df_new = new_data
            if df_new.empty:
                return {"error": "No new features provided"}

        # Prepare features
        X_new = df_new[self.feature_columns].fillna(0)
        y_new = df_new["is_ai"].astype(int)

        # Scale features
        X_new_scaled = self.scaler.transform(X_new)

        # Update incremental models
        update_results = {}
        incremental_models = ["sgd", "passive_aggressive"]

        for name in incremental_models:
            if name in self.models:
                try:
                    self.models[name].partial_fit(X_new_scaled, y_new)
                    update_results[name] = {"status": "updated", "new_samples": len(X_new)}
                except Exception as e:
                    update_results[name] = {"status": "error", "error": str(e)}

        # Save updated models
        if self.save_models():
            update_results["models_saved"] = {"status": "success"}
        else:
            update_results["models_saved"] = {"status": "failed"}

        # Add to training history
        ai_count = df_new["is_ai"].sum()
        human_count = len(df_new) - ai_count
        total_count = len(df_new)

        update_entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "update",
            "data_balance": {
                "total": total_count,
                "ai": ai_count,
                "human": human_count,
                "ratio": min(ai_count, human_count) / max(ai_count, human_count) if max(ai_count, human_count) > 0 else 0,
            },
            "samples_added": len(df_new),
            "models_updated": list(update_results.keys()),
        }
        self.training_history.append(update_entry)

        return update_results

    def show_data_balance(self, df_results: Optional[pd.DataFrame] = None) -> None:
        """
        Display data balance information.

        Args:
            df_results: DataFrame containing the data. If None, uses last training data.
        """
        if df_results is None:
            if not self.training_history:
                print("No training data available to show balance.")
                return
            # Use the last training data from history
            last_training = self.training_history[-1]
            if "data_balance" in last_training:
                balance = last_training["data_balance"]
                print(f"\n=== Data Balance (from last training) ===")
                print(f"Total files: {balance['total']}")
                print(f"AI-generated: {balance['ai']} ({balance['ai']/balance['total']*100:.1f}%)")
                print(f"Human-generated: {balance['human']} ({balance['human']/balance['total']*100:.1f}%)")
                print(f"Balance ratio: {balance['ratio']:.2f}")
                return
            else:
                print("No data balance information available.")
                return

        ai_count = df_results["is_ai"].sum()
        human_count = len(df_results) - ai_count
        total_count = len(df_results)

        print(f"\n=== Data Balance ===")
        print(f"Total files: {total_count}")
        print(f"AI-generated: {ai_count} ({ai_count/total_count*100:.1f}%)")
        print(f"Human-generated: {human_count} ({human_count/total_count*100:.1f}%)")
        print(f"Balance ratio: {min(ai_count, human_count) / max(ai_count, human_count):.2f}")

    def generate_spectrogram(self, file_path: Union[str, Path], output_path: Optional[Path] = None) -> bool:
        """
        Generate spectrogram for an audio file.

        Args:
            file_path: Path to the audio file.
            output_path: Output path for the spectrogram image.

        Returns:
            True if successful, False otherwise.
        """
        file_path = Path(file_path)

        if output_path is None:
            output_path = self.spectrograms_dir / f"{file_path.stem}_spectrogram.png"

        return self.analyzer.generate_spectrogram(file_path, output_path, self.config)

    def compare_spectrograms(
        self, file1_path: Union[str, Path], file2_path: Union[str, Path], output_path: Optional[Path] = None
    ) -> bool:
        """
        Generate comparison of two spectrograms.

        Args:
            file1_path: Path to the first audio file.
            file2_path: Path to the second audio file.
            output_path: Output path for the comparison image.

        Returns:
            True if successful, False otherwise.
        """
        file1_path = Path(file1_path)
        file2_path = Path(file2_path)

        if output_path is None:
            output_path = self.comparisons_dir / f"{file1_path.stem}_vs_{file2_path.stem}.png"

        return self.analyzer.compare_spectrograms(file1_path, file2_path, output_path, self.config)
