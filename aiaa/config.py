"""
Configuration management for AIAA: AI Audio Authenticity.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import yaml
import logging

logger = logging.getLogger(__name__)


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate that configuration has all required keys.

    Args:
        config: Configuration dictionary

    Returns:
        True if valid, False otherwise

    Note:
        Logs warnings for missing optional keys
    """
    required_keys = {
        "models": {"batch", "incremental"},
        "audio": {"supported_formats"},
        "processing": {"max_workers", "batch_threshold"},
        "output": {"models_dir", "results_dir"},
        "visualization": {"figsize", "dpi"},
    }

    is_valid = True

    for section, keys in required_keys.items():
        if section not in config:
            logger.error(f"Missing required config section: {section}")
            is_valid = False
        else:
            for key in keys:
                if key not in config.get(section, {}):
                    logger.warning(f"Missing config key: {section}.{key}")

    return is_valid


def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to the configuration file. If None, uses default path.

    Returns:
        Configuration dictionary with defaults merged with user config.

    Note:
        Validates configuration and logs warnings for missing keys
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
            logger.info(f"Loaded configuration from {config_path}")
        except Exception as e:
            logger.warning(f"Could not load config file {config_path}: {e}")
            logger.warning("Using default configuration")
            config = default_config
    else:
        logger.debug(f"Config file not found at {config_path}, using defaults")
        config = default_config

    # Validate configuration
    if validate_config(config):
        logger.debug("Configuration validation passed")
    else:
        logger.warning("Configuration validation found issues - check logs above")

    return config
