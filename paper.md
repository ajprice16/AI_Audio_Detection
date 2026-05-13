---
title: 'AIAA: AI Audio Authenticity - A Machine Learning System for Detecting AI-Generated Audio Using Benford''s Law and Spectral Analysis'
tags:
  - Python
  - machine learning
  - audio analysis
  - AI detection
  - Benford's law
  - digital forensics
  - deep learning
  - signal processing
  - authenticity
authors:
  - firstname: Alexander
    surname: Price
    orcid: 0009-0004-8952-3074
    equal-contrib: true
    affiliation: "1"
  - firstname: Sybil
    surname: Prince Nelson
    equal-contrib: false
    affiliation: "1"
  - firstname: Aaliyah
    surname: Weber
    equal-contrib: false
    affiliation: "1"
  - firstname: Madi
    surname: Lwin
    equal-contrib: false
    affiliation: "1"

affiliations:
 - name: Washington and Lee University
   index: 1
date: 1 August 2025
bibliography: paper.bib
---

# Summary

AIAA: AI Audio Authenticity is a Python package that implements a machine learning system for detecting artificially generated audio content. As AI-generated audio becomes increasingly sophisticated and prevalent, there is a growing need for reliable detection methods to combat misinformation, deepfakes, and unauthorized voice synthesis. This package addresses this challenge by combining Benford's Law analysis—a statistical principle describing the frequency distribution of leading digits in naturally occurring datasets—with comprehensive spectral feature extraction and ensemble machine learning techniques.

The system employs multiple classification algorithms including Random Forest, Gradient Boosting, Stochastic Gradient Descent (SGD), and Passive Aggressive classifiers in an ensemble approach. The package supports incremental learning, batch processing, and works with common audio formats (WAV, MP3, FLAC, OGG, M4A, AAC).

The software package doesn't come with a library through public distribution for DMCA compliance purposes, but can be recreated relatively easily with datasets of a few hundred AI and human files each.  The model used and trained by the research team may be made available upon request at the discretion of the team.

# Statement of need

The rapid advancement of AI audio generation technologies has created an urgent need for robust detection systems. Current solutions often rely on proprietary algorithms or are limited to specific types of AI-generated content. AIAA: AI Audio Authenticity fills this gap by providing:

1. **Open-source accessibility**: This package is freely available for research and practical applications
2. **Multi-modal analysis**: Combines statistical analysis with traditional audio features
3. **Ensemble approach**: Uses multiple machine learning models to improve detection accuracy and robustness
4. **Incremental learning**: Supports model updates with new data without complete retraining
5. **Comprehensive format support**: Works with common audio formats (WAV, MP3, FLAC, OGG, M4A, AAC)

This tool is particularly valuable for researchers in digital forensics, media verification, and AI safety, as well as practitioners in journalism, content moderation, and cybersecurity.

# Implementation

## Core Architecture

The AIAA system implements a modular architecture with four main components:

**Audio Feature Extraction**: The `AudioFeatureExtractor` class extracts 46 comprehensive features including mel-frequency cepstral coefficients (MFCCs), spectral characteristics (centroid, bandwidth, rolloff), temporal features (RMS energy, zero-crossing rate), and compression artifacts that may indicate AI generation [@mcfee2015librosa].

**Benford's Law Analysis**: A novel application examining the frequency distribution of leading digits in audio signal representations. Novel research performed concurrently in the same lab indicates some features of audio follow Benford's law under certain circumstances.  Methods for characterization of benfords law via [@Barabesi2022benfordcharacterizatio].

**Ensemble Classification**: The system employs both incremental (SGD, Passive Aggressive) and batch learning models (Random Forest, Gradient Boosting) using scikit-learn [@pedregosa2011scikit], combining predictions through confidence-weighted voting.

**Adaptive Learning**: Supports incremental model updates, allowing continuous improvement as new AI generation techniques emerge without requiring complete model retraining.

## Usage

The package provides both programmatic and command-line interfaces:

```python
from aiaa import AIAudioDetector

detector = AIAudioDetector()
result = detector.predict_single_file("audio_file.wav")
print(f"AI Generated: {result['prediction']}")
print(f"Confidence: {result['confidence']}")
```

```bash
aiaa analyze path/to/audio/files/ --output results.csv
```

# Results and Validation

The system has been validated on diverse datasets containing both human-generated and AI-generated audio from various sources including:
- Generative transformer produced music
- AI synthetic speech
- Natural speech
- Human produced music


Initial testing demonstrates promising results with accuracy rates exceeding 94% on balanced datasets, though performance varies depending on the quality and type of AI generation used.

# Related Work

Several approaches exist for AI-generated content detection:

1. **Deep learning approaches** [@wang2020cnn] use convolutional neural networks to detect artifacts in generated audio
2. **Spectral analysis methods** [@yang2019exposing] focus on frequency-domain anomalies
3. **Temporal consistency analysis** [@li2020identification] examines temporal patterns in generated content

AIAA distinguishes itself by incorporating Benford's Law analysis, which provides a mathematically principled statistical foundation for detection that complements traditional signal processing approaches.
The package also uses unique and combined implementations of input data and processing for maximum detection capability

# Future Work

Planned enhancements include:
- Integration of transformer-based architectures for improved feature learning
- Real-time detection capabilities for streaming audio
- Enhanced robustness against adversarial examples
- Development of standardized benchmarks for AI audio detection

# Acknowledgements

We acknowledge the contributions of the open-source community and the researchers whose work in digital signal processing and machine learning made this project possible.

This work was completed with funding through the Washington and Lee University Summer Research Scholars program.

# References
