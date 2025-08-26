#!/usr/bin/env python3
"""
Enhanced feature extraction combining multiple approaches
for personality and emotion analysis.
"""

import librosa
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def extract_enhanced_features(audio_file_path):
    """
    Extract comprehensive audio features for personality/emotion analysis
    """
    # Load audio
    y, sr = librosa.load(audio_file_path, sr=22050)
    
    # 1. MFCC features (current approach)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_stats = [np.mean(mfcc, axis=1), np.std(mfcc, axis=1), np.var(mfcc, axis=1)]
    mfcc_features = np.concatenate(mfcc_stats)
    
    # 2. Prosodic features (NEW)
    # Pitch/F0
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch_values = pitches[magnitudes > np.max(magnitudes) * 0.1]
    f0_mean = np.mean(pitch_values) if len(pitch_values) > 0 else 0
    f0_std = np.std(pitch_values) if len(pitch_values) > 0 else 0
    
    # Rhythm/Tempo
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    
    # Energy
    rms = librosa.feature.rms(y=y)[0]
    energy_mean = np.mean(rms)
    energy_std = np.std(rms)
    
    # 3. Spectral features (NEW)  
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr), axis=1)
    zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y))
    
    # 4. Voice quality features (NEW)
    # Jitter (pitch variation)
    jitter = np.std(pitch_values) / np.mean(pitch_values) if len(pitch_values) > 0 else 0
    
    # Shimmer (amplitude variation) 
    shimmer = np.std(rms) / np.mean(rms) if np.mean(rms) > 0 else 0
    
    # Harmonics-to-noise ratio
    harmonic = librosa.effects.harmonic(y)
    percussive = librosa.effects.percussive(y)
    hnr = np.mean(harmonic) / (np.mean(percussive) + 1e-10)
    
    # Combine all features
    prosodic_features = [f0_mean, f0_std, tempo, energy_mean, energy_std]
    spectral_features = [spectral_centroid, spectral_rolloff, zero_crossing_rate] + spectral_contrast.tolist()
    voice_quality_features = [jitter, shimmer, hnr]
    
    # Final feature vector
    all_features = np.concatenate([
        mfcc_features,
        prosodic_features,
        spectral_features, 
        voice_quality_features
    ])
    
    return all_features

if __name__ == "__main__":
    # Test feature extraction
    audio_path = "test_audio.wav"  # Replace with actual audio file
    features = extract_enhanced_features(audio_path)
    print(f"Extracted {len(features)} features")
    print(f"Feature vector shape: {features.shape}")
        