"""
Audio feature extraction utilities for AIAA: AI Audio Authenticity.
"""

from typing import Dict, List
import numpy as np
import librosa
from scipy import stats
from scipy.stats import ks_2samp


class AudioFeatureExtractor:
    """Audio feature extraction utilities."""

    @staticmethod
    def extract_benford_features(frequencies: List[float]) -> Dict[str, float]:
        """
        Extract Benford's Law features from frequency data.

        Args:
            frequencies: List of frequency values (Hz).

        Returns:
            Dictionary with keys:
            - chi2_p, chi2_stat: Chi-square test p-value and statistic
            - ks_p, ks_stat: Kolmogorov-Smirnov test p-value and statistic
            - mad: Mean absolute deviation from Benford's Law
            - max_deviation: Maximum single deviation
            - entropy: Information entropy of digit distribution

            Returns empty dict {} if fewer than 10 frequencies provided.

        Note:
            Implements Benford's Law analysis to distinguish AI from human audio.
            Tests if first-digit distribution of peak frequencies matches
            expected distribution from Benford's Law.
        """
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
            except Exception:
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
            except Exception:
                features["ks_p"] = 1.0
                features["ks_stat"] = 0.0

            # Mean absolute deviation from expected
            mad = sum(abs(obs - exp) for obs, exp in zip(observed_freq, expected_benford)) / len(expected_benford)
            features["mad"] = mad

            # Maximum deviation
            max_dev = max(abs(obs - exp) for obs, exp in zip(observed_freq, expected_benford))
            features["max_deviation"] = max_dev

            # Entropy
            entropy = -sum(p * np.log(p) for p in observed_freq if p > 0)
            features["entropy"] = entropy

            return features

        except Exception:
            return {}

    @staticmethod
    def extract_spectral_features(y: np.ndarray, sr: int) -> Dict[str, float]:
        """
        Extract spectral features from audio.

        Args:
            y: Audio signal (numpy array).
            sr: Sample rate in Hz.

        Returns:
            Dictionary with keys:
            - spectral_centroid, spectral_bandwidth, spectral_rolloff: Basic spectral features
            - mfcc_0 to mfcc_12: Mean Mel-frequency cepstral coefficients
            - mfcc_0_std to mfcc_12_std: Standard deviation of MFCCs
            - chroma_mean, chroma_std: Chroma feature statistics
            - spectral_contrast: Mean spectral contrast

            Returns empty dict {} on error.

        Note:
            All features are float values. Returns empty dict on extraction failure.
        """
        try:
            features: Dict[str, float] = {}

            # Basic spectral features
            features["spectral_centroid"] = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
            features["spectral_bandwidth"] = float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)))
            features["spectral_rolloff"] = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))

            # MFCCs
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            for i in range(13):
                features[f"mfcc_{i}"] = float(np.mean(mfccs[i]))
                features[f"mfcc_{i}_std"] = float(np.std(mfccs[i]))  # Add std features

            # Chroma
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            features["chroma_mean"] = float(np.mean(chroma))
            features["chroma_std"] = float(np.std(chroma))

            # Spectral contrast
            contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
            features["spectral_contrast"] = float(np.mean(contrast))

            return features

        except Exception:
            return {}

    @staticmethod
    def extract_temporal_features(y: np.ndarray, sr: int) -> Dict[str, float]:
        """
        Extract temporal features from audio.

        Args:
            y: Audio signal (numpy array).
            sr: Sample rate in Hz.

        Returns:
            Dictionary with keys:
            - temporal_rms_mean, temporal_rms_std: RMS energy statistics
            - temporal_zcr_mean, temporal_zcr_std: Zero-crossing rate statistics
            - temporal_tempo: Estimated tempo in BPM (0.0 on error)
            - temporal_spectral_flatness: Mean spectral flatness
            - temporal_dynamic_range: Max amplitude - min amplitude

            Returns empty dict {} on error.

        Note:
            All features are float values. Dynamic range is non-negative.
        """
        try:
            features: Dict[str, float] = {}

            # RMS Energy
            rms = librosa.feature.rms(y=y)
            features["temporal_rms_mean"] = float(np.mean(rms))
            features["temporal_rms_std"] = float(np.std(rms))

            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(y)
            features["temporal_zcr_mean"] = float(np.mean(zcr))
            features["temporal_zcr_std"] = float(np.std(zcr))

            # Tempo
            try:
                tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
                features["temporal_tempo"] = float(tempo.item() if hasattr(tempo, "item") else tempo)
            except Exception:
                features["temporal_tempo"] = 0.0

            # Spectral flatness
            flatness = librosa.feature.spectral_flatness(y=y)
            features["temporal_spectral_flatness"] = float(np.mean(flatness))

            # Dynamic range
            if len(y) > 0:
                abs_y = np.abs(y)
                max_val: float = float(np.max(abs_y))
                min_val: float = float(np.min(abs_y))
                features["temporal_dynamic_range"] = max_val - min_val
            else:
                features["temporal_dynamic_range"] = 0.0

            return features

        except Exception:
            return {}

    @staticmethod
    def extract_compression_features(y: np.ndarray, sr: int) -> Dict[str, float]:
        """
        Extract compression-related features from audio.

        Args:
            y: Audio signal (numpy array).
            sr: Sample rate in Hz.

        Returns:
            Dictionary with keys:
            - compression_estimated_bit_depth: Estimated bit depth (log2 scale)
            - compression_clipping_ratio: Ratio of clipped samples (0.0-1.0)
            - compression_dc_offset: Mean offset in signal
            - compression_high_freq_ratio: Power ratio above sr/4 (0.0-1.0)

            Returns empty dict {} on error.

        Note:
            Bit depth is estimated from unique values in signal.
            Clipping threshold is 0.95 of maximum amplitude.
            High frequency is defined as > sample_rate/4.
        """
        try:
            features: Dict[str, float] = {}

            # Bit depth estimation (based on quantization levels)
            unique_values = len(np.unique(y))
            max_possible = 2**16  # Assume 16-bit as reference
            features["compression_estimated_bit_depth"] = float(np.log2(unique_values) if unique_values > 1 else 1.0)

            # Clipping detection
            threshold = 0.95
            clipped_samples: int = int(np.sum(np.abs(y) > threshold))
            features["compression_clipping_ratio"] = float(clipped_samples / len(y))

            # DC offset
            features["compression_dc_offset"] = float(np.mean(y))

            # High frequency content
            fft = np.fft.fft(y)
            freqs = np.fft.fftfreq(len(fft), 1 / sr)
            high_freq_power: float = float(np.sum(np.abs(fft[freqs > sr / 4]) ** 2))
            total_power: float = float(np.sum(np.abs(fft) ** 2))
            features["compression_high_freq_ratio"] = float(high_freq_power / total_power if total_power > 0 else 0.0)

            return features

        except Exception:
            return {}
