"""
Logging configuration for AIAA: AI Audio Authenticity.
Provides centralized logging setup and utilities.
"""

import logging
import logging.handlers
from pathlib import Path
from typing import Optional


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[Path] = None,
    log_dir: Optional[Path] = None,
) -> None:
    """
    Configure logging for the AIAA package.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file. If None, only console logging.
        log_dir: Directory for log files. Created if doesn't exist.

    Raises:
        ValueError: If invalid log level provided.
    """
    valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
    if log_level.upper() not in valid_levels:
        raise ValueError(f"Invalid log level: {log_level}. Must be one of {valid_levels}")

    # Create logs directory if specified
    if log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))

    # Remove existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Console handler
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # File handler (if log file specified)
    if log_file or log_dir:
        file_path = Path(log_file) if log_file else (log_dir / "aiaa.log")
        file_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            file_handler = logging.handlers.RotatingFileHandler(
                file_path,
                maxBytes=10485760,  # 10MB
                backupCount=5,
            )
            file_formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            file_handler.setFormatter(file_formatter)
            root_logger.addHandler(file_handler)
        except Exception as e:
            root_logger.warning(f"Could not setup file logging: {e}")


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a module.

    Args:
        name: Logger name, typically __name__

    Returns:
        Configured Logger instance
    """
    return logging.getLogger(name)
