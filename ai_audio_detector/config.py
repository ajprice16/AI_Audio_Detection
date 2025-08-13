"""
Configuration management for AI Audio Detector.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import yaml


def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to the configuration file. If None, uses default path.

    Returns:
        Configuration dictionary with defaults merged with user config.
    """
    if config_path is None:
        config_path = Path.cwd() / "config.yaml"

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

    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                user_config = yaml.safe_load(f)
            # Merge with defaults
            config = {**default_config, **user_config}
            return config
        except Exception as e:
            print(f"Warning: Could not load config file {config_path}: {e}")
            print("Using default configuration")

    return default_config
