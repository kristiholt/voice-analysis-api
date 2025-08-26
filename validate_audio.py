#!/usr/bin/env python3
"""
Audio processing validation script.
Tests that our feature extraction is working correctly.
"""

import numpy as np
import librosa
from app.features import FeatureExtractor
from app.audio_io import AudioProcessor
import tempfile
import soundfile as sf

def create_test_audio():
    """Create test audio signals with known characteristics"""
    sr = 22050
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration))
    
    # Create different test signals
    signals = {
        "440hz_sine": np.sin(2 * np.pi * 440 * t),  # Pure A note
        "880hz_sine": np.sin(2 * np.pi * 880 * t),  # Higher A note  
        "white_noise": np.random.normal(0, 0.1, len(t)),
        "silent": np.zeros(len(t))
    }
    
    return signals, sr

def validate_mfcc_extraction():
    """Test that MFCC extraction produces expected results"""
    print("🔬 Testing MFCC Feature Extraction...")
    
    # Create test signals
    signals, sr = create_test_audio()
    extractor = FeatureExtractor()
    processor = AudioProcessor()
    
    results = {}
    
    for name, signal in signals.items():
        # Convert to bytes and back to simulate file processing
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            sf.write(tmp.name, signal, sr)
            
            # Read file as bytes
            with open(tmp.name, 'rb') as f:
                audio_bytes = f.read()
            
            # Process through our pipeline
            audio_data, sample_rate, metadata = processor.decode_audio_file(audio_bytes, 'test.wav')
            features, feat_metadata = extractor.extract_features(audio_data, sample_rate)
            
            results[name] = {
                'features': features,
                'metadata': metadata,
                'feature_shape': features.shape,
                'feature_range': (features.min(), features.max()),
                'feature_mean': features.mean()
            }
    
    # Validate results
    print(f"✅ Feature vector shape: {results['440hz_sine']['feature_shape']}")
    print(f"✅ 440Hz sine wave mean: {results['440hz_sine']['feature_mean']:.4f}")
    print(f"✅ 880Hz sine wave mean: {results['880hz_sine']['feature_mean']:.4f}")
    print(f"✅ White noise mean: {results['white_noise']['feature_mean']:.4f}")
    print(f"✅ Silent signal mean: {results['silent']['feature_mean']:.4f}")
    
    # Test consistency (same input = same output)
    features1, _ = extractor.extract_features(signals['440hz_sine'], sr)
    features2, _ = extractor.extract_features(signals['440hz_sine'], sr)
    
    if np.allclose(features1, features2):
        print("✅ DETERMINISTIC: Same input produces identical features")
    else:
        print("❌ ERROR: Same input produces different features")
    
    # Test frequency discrimination
    freq_diff = abs(results['440hz_sine']['feature_mean'] - results['880hz_sine']['feature_mean'])
    if freq_diff > 0.01:  # Should be meaningfully different
        print(f"✅ FREQUENCY SENSITIVE: 440Hz vs 880Hz shows {freq_diff:.4f} difference")
    else:
        print(f"❌ WARNING: 440Hz vs 880Hz too similar ({freq_diff:.4f})")
    
    return results

if __name__ == "__main__":
    print("🎵 Validating X Voice API Audio Processing...")
    validate_mfcc_extraction()
    print("✅ Validation complete!")