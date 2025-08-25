"""
Deterministic audio feature extraction.
Extracts MFCC features with fixed parameters for consistent results.
"""

import numpy as np
import librosa
from typing import Dict, Any, Tuple
import logging

from .utils import get_env_var

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """Deterministic audio feature extraction"""
    
    def __init__(self):
        # Fixed MFCC parameters for deterministic extraction
        self.n_mfcc = 20
        self.n_fft = 1024
        self.hop_length = 256
        self.n_mels = 128
        self.fmin = 0
        self.fmax = None  # Will be set to sr/2
        
        # Feature vector configuration
        self.target_length = 512  # Fixed length for consistent model input
        
        self.version = get_env_var("FEATURES_VERSION", "mfcc_v1")
        
    def extract_features(self, audio_data: np.ndarray, sample_rate: int) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Extract deterministic MFCC features from audio.
        
        Args:
            audio_data: Mono audio signal as float32 array
            sample_rate: Audio sample rate
            
        Returns:
            Tuple of (feature_vector, metadata)
        """
        try:
            # Set fmax to Nyquist frequency
            fmax = sample_rate // 2
            
            # Extract MFCC features
            mfcc_features = librosa.feature.mfcc(
                y=audio_data,
                sr=sample_rate,
                n_mfcc=self.n_mfcc,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                n_mels=self.n_mels,
                fmin=self.fmin,
                fmax=fmax
            )
            
            # Extract additional spectral features
            spectral_features = self._extract_spectral_features(audio_data, sample_rate)
            
            # Extract rhythm features
            rhythm_features = self._extract_rhythm_features(audio_data, sample_rate)
            
            # Combine all features
            combined_features = np.concatenate([
                mfcc_features.flatten(),
                spectral_features,
                rhythm_features
            ])
            
            # Normalize to fixed length
            feature_vector = self._normalize_to_fixed_length(combined_features)
            
            metadata = {
                'version': self.version,
                'n_mfcc': self.n_mfcc,
                'n_fft': self.n_fft,
                'hop_length': self.hop_length,
                'sample_rate': sample_rate,
                'mfcc_shape': mfcc_features.shape,
                'feature_length': len(feature_vector),
                'spectral_features': len(spectral_features),
                'rhythm_features': len(rhythm_features)
            }
            
            logger.debug(f"Extracted features: {metadata}")
            return feature_vector, metadata
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            raise RuntimeError(f"Feature extraction failed: {str(e)}")
    
    def _extract_spectral_features(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        """Extract spectral features (centroid, bandwidth, etc.)"""
        try:
            # Spectral centroid
            spectral_centroid = librosa.feature.spectral_centroid(
                y=audio_data, sr=sample_rate, hop_length=self.hop_length
            )
            
            # Spectral bandwidth
            spectral_bandwidth = librosa.feature.spectral_bandwidth(
                y=audio_data, sr=sample_rate, hop_length=self.hop_length
            )
            
            # Spectral rolloff
            spectral_rolloff = librosa.feature.spectral_rolloff(
                y=audio_data, sr=sample_rate, hop_length=self.hop_length
            )
            
            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(
                audio_data, hop_length=self.hop_length
            )
            
            # Aggregate statistics for each feature
            features = []
            for feature in [spectral_centroid, spectral_bandwidth, spectral_rolloff, zcr]:
                features.extend([
                    np.mean(feature),
                    np.std(feature),
                    np.min(feature),
                    np.max(feature)
                ])
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            logger.warning(f"Error extracting spectral features: {e}")
            return np.zeros(16, dtype=np.float32)  # 4 features * 4 stats
    
    def _extract_rhythm_features(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        """Extract rhythm and tempo features"""
        try:
            # Tempo estimation
            tempo, beats = librosa.beat.beat_track(
                y=audio_data, sr=sample_rate, hop_length=self.hop_length
            )
            
            # RMS energy
            rms = librosa.feature.rms(
                y=audio_data, hop_length=self.hop_length
            )
            
            # Onset strength
            onset_strength = librosa.onset.onset_strength(
                y=audio_data, sr=sample_rate, hop_length=self.hop_length
            )
            
            features = [
                float(tempo),
                float(len(beats) / (len(audio_data) / sample_rate)),  # Beat density
                float(np.mean(rms)),
                float(np.std(rms)),
                float(np.mean(onset_strength)),
                float(np.std(onset_strength))
            ]
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            logger.warning(f"Error extracting rhythm features: {e}")
            return np.zeros(6, dtype=np.float32)
    
    def _normalize_to_fixed_length(self, features: np.ndarray) -> np.ndarray:
        """Normalize feature vector to fixed length"""
        if len(features) == self.target_length:
            return features
        elif len(features) > self.target_length:
            # Truncate
            return features[:self.target_length]
        else:
            # Pad with zeros
            padded = np.zeros(self.target_length, dtype=np.float32)
            padded[:len(features)] = features
            return padded
    
    def compute_feature_hash(self, feature_vector: np.ndarray) -> str:
        """Compute deterministic hash of feature vector"""
        import hashlib
        # Round to ensure deterministic results across platforms
        rounded = np.round(feature_vector, decimals=6)
        feature_bytes = rounded.tobytes()
        return hashlib.sha256(feature_bytes).hexdigest()


# Global feature extractor instance
feature_extractor = FeatureExtractor()
