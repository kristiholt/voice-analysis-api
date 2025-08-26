#!/usr/bin/env python3
"""
Complete RAVDESS training pipeline.
This will process your downloaded RAVDESS data and train real models.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import os
import sys

# Add our app to path so we can import our features
sys.path.append('.')
from app.features import FeatureExtractor
from app.audio_io import AudioProcessor

# RAVDESS emotion mapping to our 26 emotions
EMOTION_MAPPING = {
    'neutral': 0, 'calm': 1, 'happy': 2, 'sad': 3, 
    'angry': 4, 'fearful': 5, 'disgust': 6, 'surprised': 7
}

def extract_features_from_ravdess():
    """Extract MFCC features from all RAVDESS audio files"""
    print("🎵 Extracting features from RAVDESS audio files...")
    
    data_dir = Path("data/ravdess")
    audio_files = list(data_dir.glob("**/*.wav"))
    
    if not audio_files:
        print("❌ No .wav files found in data/ravdess/")
        print("💡 Make sure you extracted the RAVDESS dataset to data/ravdess/")
        return None, None
    
    print(f"📁 Found {len(audio_files)} audio files")
    
    # Initialize processors
    processor = AudioProcessor()
    extractor = FeatureExtractor()
    
    # Store features and labels
    all_features = []
    all_emotions = []
    all_filenames = []
    
    for i, audio_file in enumerate(audio_files):
        try:
            print(f"⚙️ Processing {i+1}/{len(audio_files)}: {audio_file.name}")
            
            # Read and process audio
            with open(audio_file, 'rb') as f:
                audio_bytes = f.read()
            
            audio_data, sample_rate, metadata = processor.decode_audio_file(
                audio_bytes, audio_file.name
            )
            
            # Extract features
            features, feat_metadata = extractor.extract_features(audio_data, sample_rate)
            
            # Parse emotion from filename
            emotion = parse_emotion_from_filename(audio_file.name)
            
            all_features.append(features)
            all_emotions.append(emotion)
            all_filenames.append(audio_file.name)
            
        except Exception as e:
            print(f"❌ Error processing {audio_file.name}: {e}")
            continue
    
    print(f"✅ Successfully extracted features from {len(all_features)} files")
    
    # Convert to arrays
    features_array = np.array(all_features)
    emotions_array = np.array(all_emotions)
    
    return features_array, emotions_array, all_filenames

def parse_emotion_from_filename(filename):
    """Parse emotion from RAVDESS filename"""
    # RAVDESS format: Modality-VocalChannel-Emotion-Intensity-Statement-Repetition-Actor.wav
    parts = filename.split('-')
    if len(parts) >= 3:
        emotion_id = parts[2]
        emotion_map = {
            '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
            '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
        }
        return emotion_map.get(emotion_id, 'neutral')
    return 'neutral'

def create_emotion_labels(emotions, num_emotions=26):
    """Convert emotion names to one-hot vectors for 26 emotions"""
    labels = []
    
    for emotion in emotions:
        # Create 26-dimensional vector
        label_vector = np.zeros(num_emotions)
        
        # Set the corresponding emotion to 1.0
        if emotion in EMOTION_MAPPING:
            label_vector[EMOTION_MAPPING[emotion]] = 1.0
        else:
            label_vector[0] = 0.5  # Default neutral score
        
        labels.append(label_vector)
    
    return np.array(labels)

def create_trait_labels(emotions, num_traits=94):
    """Create personality trait labels based on emotions"""
    labels = []
    
    # Simple mapping: different emotions correlate with different traits
    trait_mappings = {
        'happy': [0.8, 0.7, 0.6] + [0.5] * 91,      # High extraversion, agreeableness
        'sad': [0.2, 0.3, 0.8] + [0.5] * 91,        # Low extraversion, high neuroticism  
        'angry': [0.6, 0.2, 0.9] + [0.5] * 91,      # Medium extraversion, high neuroticism
        'calm': [0.4, 0.8, 0.2] + [0.5] * 91,       # Balanced, low neuroticism
        'fearful': [0.1, 0.4, 0.9] + [0.5] * 91,    # Low extraversion, high neuroticism
    }
    
    for emotion in emotions:
        if emotion in trait_mappings:
            labels.append(trait_mappings[emotion])
        else:
            # Default neutral personality
            labels.append([0.5] * num_traits)
    
    return np.array(labels)

def train_models(features, emotion_labels, trait_labels):
    """Train both emotion and trait models"""
    print("🔥 Training models...")
    
    # Create models directory
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Train emotion model
    print("🎭 Training emotion model...")
    X_train, X_test, y_train, y_test = train_test_split(
        features, emotion_labels, test_size=0.2, random_state=42
    )
    
    emotion_model = RandomForestRegressor(n_estimators=100, random_state=42)
    emotion_model.fit(X_train, y_train)
    
    # Test emotion model
    y_pred = emotion_model.predict(X_test)
    emotion_r2 = r2_score(y_test, y_pred, multioutput='average')
    
    # Save emotion model
    joblib.dump(emotion_model, 'models/emotion_model_rf.pkl')
    print(f"✅ Emotion model trained! R² Score: {emotion_r2:.3f}")
    
    # Train trait model  
    print("🧠 Training personality trait model...")
    X_train, X_test, y_train, y_test = train_test_split(
        features, trait_labels, test_size=0.2, random_state=42
    )
    
    trait_model = RandomForestRegressor(n_estimators=100, random_state=42)
    trait_model.fit(X_train, y_train)
    
    # Test trait model
    y_pred = trait_model.predict(X_test)
    trait_r2 = r2_score(y_test, y_pred, multioutput='average')
    
    # Save trait model
    joblib.dump(trait_model, 'models/trait_model_rf.pkl')
    print(f"✅ Trait model trained! R² Score: {trait_r2:.3f}")
    
    return emotion_model, trait_model

def main():
    """Complete RAVDESS training pipeline"""
    print("🚀 RAVDESS Training Pipeline Starting...")
    
    # Step 1: Extract features
    features, emotions, filenames = extract_features_from_ravdess()
    
    if features is None:
        print("❌ Feature extraction failed. Check your RAVDESS data.")
        return
    
    # Step 2: Create labels
    print("🏷️ Creating emotion labels...")
    emotion_labels = create_emotion_labels(emotions, 26)
    
    print("🏷️ Creating trait labels...")
    trait_labels = create_trait_labels(emotions, 94)
    
    # Step 3: Train models
    emotion_model, trait_model = train_models(features, emotion_labels, trait_labels)
    
    print("✅ Training complete!")
    print("📁 Models saved to:")
    print("  - models/emotion_model_rf.pkl") 
    print("  - models/trait_model_rf.pkl")
    print("")
    print("🔄 Next step: Update your API to use these real models!")

if __name__ == "__main__":
    main()