#!/usr/bin/env python3
"""
Quick RAVDESS training - uses just 100 files for fast results.
This gets you working models in under 2 minutes!
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

# Add our app to path
sys.path.append('.')
from app.features import FeatureExtractor
from app.audio_io import AudioProcessor

# RAVDESS emotion mapping
EMOTION_MAPPING = {
    'neutral': 0, 'calm': 1, 'happy': 2, 'sad': 3, 
    'angry': 4, 'fearful': 5, 'disgust': 6, 'surprised': 7
}

def quick_training(max_files=100):
    """Train models on first 100 files for quick results"""
    print(f"🚀 Quick Training - Processing first {max_files} files...")
    
    data_dir = Path("data/ravdess")
    audio_files = list(data_dir.glob("**/*.wav"))[:max_files]  # Just first 100
    
    print(f"📁 Processing {len(audio_files)} audio files")
    
    # Initialize processors
    processor = AudioProcessor()
    extractor = FeatureExtractor()
    
    # Store data
    all_features = []
    all_emotions = []
    
    for i, audio_file in enumerate(audio_files):
        try:
            print(f"⚙️ {i+1}/{len(audio_files)}: {audio_file.name}")
            
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
            
        except Exception as e:
            print(f"❌ Error processing {audio_file.name}: {e}")
            continue
    
    print(f"✅ Feature extraction complete: {len(all_features)} samples")
    
    # Convert to arrays
    features_array = np.array(all_features)
    
    # Create labels
    print("🏷️ Creating labels...")
    emotion_labels = create_emotion_labels(all_emotions, 26)
    trait_labels = create_trait_labels(all_emotions, 94)
    
    # Train models
    print("🔥 Training models...")
    
    # Create models directory
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Train emotion model
    print("🎭 Training emotion model...")
    if len(features_array) < 10:
        print("❌ Not enough samples for training")
        return
    
    X_train, X_test, y_train, y_test = train_test_split(
        features_array, emotion_labels, test_size=0.2, random_state=42
    )
    
    emotion_model = RandomForestRegressor(n_estimators=50, random_state=42)  # Fewer trees for speed
    emotion_model.fit(X_train, y_train)
    
    # Test emotion model
    y_pred = emotion_model.predict(X_test)
    emotion_r2 = r2_score(y_test, y_pred, multioutput='uniform_average')
    
    # Save emotion model
    joblib.dump(emotion_model, 'models/emotion_model_rf.pkl')
    print(f"✅ Emotion model trained! R² Score: {emotion_r2:.3f}")
    
    # Train trait model  
    print("🧠 Training personality trait model...")
    X_train, X_test, y_train, y_test = train_test_split(
        features_array, trait_labels, test_size=0.2, random_state=42
    )
    
    trait_model = RandomForestRegressor(n_estimators=50, random_state=42)
    trait_model.fit(X_train, y_train)
    
    # Test trait model
    y_pred = trait_model.predict(X_test)
    trait_r2 = r2_score(y_test, y_pred, multioutput='uniform_average')
    
    # Save trait model
    joblib.dump(trait_model, 'models/trait_model_rf.pkl')
    print(f"✅ Trait model trained! R² Score: {trait_r2:.3f}")
    
    print("\n🎉 Quick training complete!")
    print("📁 Models saved to:")
    print("  - models/emotion_model_rf.pkl") 
    print("  - models/trait_model_rf.pkl")
    print(f"📊 Training data: {len(all_features)} audio samples")
    print("\n🔄 Next: Update your API to use these real models!")

def parse_emotion_from_filename(filename):
    """Parse emotion from RAVDESS filename"""
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
    """Convert emotion names to one-hot vectors"""
    labels = []
    for emotion in emotions:
        label_vector = np.zeros(num_emotions)
        if emotion in EMOTION_MAPPING:
            label_vector[EMOTION_MAPPING[emotion]] = 1.0
        else:
            label_vector[0] = 0.5  # Default neutral
        labels.append(label_vector)
    return np.array(labels)

def create_trait_labels(emotions, num_traits=94):
    """Create personality trait labels based on emotions"""
    labels = []
    trait_mappings = {
        'happy': [0.8, 0.7, 0.6] + [0.5] * 91,      
        'sad': [0.2, 0.3, 0.8] + [0.5] * 91,        
        'angry': [0.6, 0.2, 0.9] + [0.5] * 91,      
        'calm': [0.4, 0.8, 0.2] + [0.5] * 91,       
        'fearful': [0.1, 0.4, 0.9] + [0.5] * 91,    
    }
    
    for emotion in emotions:
        if emotion in trait_mappings:
            labels.append(trait_mappings[emotion])
        else:
            labels.append([0.5] * num_traits)
    
    return np.array(labels)

if __name__ == "__main__":
    quick_training(100)