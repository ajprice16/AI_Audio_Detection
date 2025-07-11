#!/usr/bin/env python3
"""
Setup script for AI Audio Detector
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
readme_path = Path(__file__).parent / "README.md"
long_description = (
    readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""
)

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
if requirements_path.exists():
    with open(requirements_path, "r", encoding="utf-8") as f:
        requirements = [
            line.strip() for line in f if line.strip() and not line.startswith("#")
        ]
else:
    requirements = [
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "librosa>=0.9.0",
        "scikit-learn>=1.0.0",
        "scipy>=1.7.0",
        "matplotlib>=3.4.0",
        "tqdm>=4.62.0",
        "joblib>=1.1.0",
        "pyyaml>=6.0",
    ]

setup(
    name="ai-audio-detector",
    version="1.0.0",
    author="Alex Price",
    author_email="ajprice@mail.wli.edu",
    description="Machine learning system for detecting AI-generated audio using Benford's Law and advanced spectral features",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/ai-audio-detector",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "ai-audio-detector=ai_audio_detector:main",
        ],
    },
    keywords="ai detection audio machine-learning benford-law audio-analysis",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/ai-audio-detector/issues",
        "Source": "https://github.com/yourusername/ai-audio-detector",
        "Documentation": "https://github.com/yourusername/ai-audio-detector#readme",
    },
)
