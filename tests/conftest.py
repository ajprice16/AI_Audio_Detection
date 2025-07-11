"""
Pytest configuration and shared fixtures
"""

import pytest
import tempfile
import shutil
import numpy as np
from pathlib import Path
import pandas as pd
import sys

# Add the parent directory to path to import the module
sys.path.insert(0, str(Path(__file__).parent.parent))

from ai_audio_detector import AIAudioDetector


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests"""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_audio_data():
    """Generate sample audio data for testing"""
    sample_rate = 22050
    duration = 1.0  # 1 second
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
    return audio, sample_rate


@pytest.fixture
def sample_features_data():
    """Generate sample feature data for testing"""
    return pd.DataFrame(
        {
            "filename": [
                "file1.wav",
                "file2.wav",
                "file3.wav",
                "file4.wav",
                "file5.wav",
                "file6.wav",
            ],
            "full_path": [
                "/path/file1.wav",
                "/path/file2.wav",
                "/path/file3.wav",
                "/path/file4.wav",
                "/path/file5.wav",
                "/path/file6.wav",
            ],
            "source_directory": ["AI", "AI", "AI", "Human", "Human", "Human"],
            "file_extension": [".wav", ".wav", ".wav", ".wav", ".wav", ".wav"],
            "is_ai": [True, True, True, False, False, False],
            "audio_duration": [2.5, 3.0, 2.8, 3.2, 2.9, 3.1],
            "sample_rate": [22050, 22050, 22050, 22050, 22050, 22050],
            "file_size_mb": [1.2, 1.5, 1.3, 1.6, 1.4, 1.7],
            "benford_chi2_p": [0.1, 0.05, 0.12, 0.8, 0.9, 0.85],
            "benford_chi2_stat": [15.2, 18.5, 16.1, 3.2, 2.1, 2.8],
            "spectral_centroid": [1500, 1600, 1550, 2000, 2100, 2050],
            "spectral_bandwidth": [800, 850, 825, 900, 950, 925],
            "temporal_rms_mean": [0.1, 0.12, 0.11, 0.15, 0.14, 0.145],
            "compression_clipping_ratio": [0.01, 0.02, 0.015, 0.005, 0.008, 0.006],
        }
    )


@pytest.fixture
def ai_detector(temp_dir):
    """Create an AI detector instance for testing"""
    return AIAudioDetector(base_dir=temp_dir)


@pytest.fixture
def trained_detector(ai_detector, sample_features_data):
    """Create a trained AI detector for testing"""
    ai_detector.train_models(sample_features_data)
    return ai_detector


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Set up the test environment"""
    # Suppress warnings during testing
    import warnings

    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    # Set random seeds for reproducible tests
    np.random.seed(42)
    import random

    random.seed(42)


def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "unit: mark test as unit test")
    config.addinivalue_line("markers", "benchmark: mark test as benchmark")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically"""
    for item in items:
        # Mark slow tests
        if "benchmark" in item.nodeid or "integration" in item.nodeid:
            item.add_marker(pytest.mark.slow)

        # Mark integration tests
        if "integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)

        # Mark unit tests
        if "test_" in item.nodeid and "integration" not in item.nodeid:
            item.add_marker(pytest.mark.unit)
