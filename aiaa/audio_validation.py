"""
Audio file validation utilities for AIAA.
Validates audio files for integrity and compatibility.
"""

import logging
from pathlib import Path
from typing import Tuple, Optional, List

from .runtime import configure_numba_cache

configure_numba_cache()

import librosa

logger = logging.getLogger(__name__)


class AudioValidationError(Exception):
    # Exception raised for audio validation failures....

    pass


def validate_audio_file(file_path: Path, max_duration: float = 3600.0) -> Tuple[bool, str]:
    """
    Validate that a file is valid audio and meets requirements.

    Args:
        file_path: Path to audio file
        max_duration: Maximum allowed duration in seconds (default 1 hour)

    Returns:
        Tuple of (is_valid: bool, message: str)

    Note:
        Validates:
        - File exists and is readable
        - File is recognized as audio format
        - Audio has reasonable duration
        - Audio is not corrupted
    """
    # Check existence
    if not file_path.exists():
        return False, f"File does not exist: {file_path}"

    if not file_path.is_file():
        return False, f"Path is not a file: {file_path}"

    # Check readability
    if not file_path.stat().st_size > 0:
        return False, f"File is empty: {file_path}"

    # Try to load minimal audio info
    try:
        duration = librosa.get_duration(path=str(file_path))

        if duration is None or duration <= 0:
            return False, f"Invalid audio duration: {duration}"

        if duration > max_duration:
            return False, f"Audio exceeds maximum duration ({duration}s > {max_duration}s)"

        return True, f"Valid audio file ({duration:.2f}s)"

    except Exception as e:
        return False, f"Failed to validate audio: {str(e)}"


def get_audio_sample_rate(file_path: Path) -> Optional[int]:
    """
    Get sample rate of audio file.

    Args:
        file_path: Path to audio file

    Returns:
        Sample rate in Hz, or None if error

    Note:
        Returns None gracefully on error rather than raising exception
    """
    try:
        sr = librosa.get_samplerate(str(file_path))
        return int(sr) if sr is not None else None
    except Exception as e:
        logger.warning(f"Could not get sample rate for {file_path}: {e}")
        return None


def validate_batch_audio_files(
    directory: Path,
    supported_formats: List[str],
    max_duration: float = 3600.0,
) -> Tuple[List[Path], List[Tuple[Optional[Path], str]]]:
    """
    Validate all audio files in a directory.

    Args:
        directory: Directory containing audio files
        supported_formats: List of supported file extensions (e.g., ['.wav', '.mp3'])
        max_duration: Maximum allowed duration for each file

    Returns:
        Tuple of (valid_files: List[Path], invalid_files: List[Tuple[Path|None, str]])
        - valid_files: List of Path objects for valid audio files
        - invalid_files: List of (Path, error_message) tuples for invalid files

    Note:
        Returns separate lists rather than raising on first error.
        Invalid files list contains (Path, message) tuples where message describes the issue.
    """
    valid_files: List[Path] = []
    invalid_files: List[Tuple[Optional[Path], str]] = []

    # Find audio files
    if not directory.exists():
        logger.warning(f"Directory does not exist: {directory}")
        return [], [(None, f"Directory does not exist: {directory}")]

    for ext in supported_formats:
        for file_path in directory.rglob(f"*{ext}"):
            is_valid, message = validate_audio_file(file_path, max_duration)

            if is_valid:
                valid_files.append(file_path)
                logger.debug(f"Validated: {file_path} - {message}")
            else:
                invalid_files.append((file_path, message))
                logger.warning(f"Invalid audio file {file_path}: {message}")

    return valid_files, invalid_files
