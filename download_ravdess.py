#!/usr/bin/env python3
"""
Download and prepare RAVDESS dataset for emotion recognition training.
"""

import os
import requests
import zipfile
from pathlib import Path
import pandas as pd

# RAVDESS emotion mapping
RAVDESS_EMOTIONS = {
    '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
    '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
}

def download_ravdess():
    """Download RAVDESS dataset"""
    print("📥 Downloading RAVDESS dataset...")
    
    # RAVDESS is available via Kaggle or direct download
    # For this example, I'll show the structure you need
    
    # Create sample structure (you'll need to get actual files)
    data_dir = Path("data/ravdess")
    data_dir.mkdir(exist_ok=True)
    
    print("💡 RAVDESS Download Instructions:")
    print("1. Visit: https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio")
    print("2. Download the dataset zip file")
    print("3. Extract to data/ravdess/ directory")
    print("4. Run this script again to process the files")
    
    return data_dir

def parse_ravdess_filename(filename):
    """
    Parse RAVDESS filename to extract metadata.
    Format: Modality-VocalChannel-Emotion-EmotionalIntensity-Statement-Repetition-Actor.wav
    """
    parts = filename.stem.split('-')
    if len(parts) != 7:
        return None
    
    return {
        'modality': parts[0],
        'vocal_channel': parts[1], 
        'emotion': RAVDESS_EMOTIONS.get(parts[2], 'unknown'),
        'emotion_id': parts[2],
        'intensity': parts[3],
        'statement': parts[4],
        'repetition': parts[5],
        'actor': parts[6],
        'filename': filename.name
    }

def create_ravdess_labels():
    """Create emotion labels for RAVDESS data"""
    print("🏷️ Creating RAVDESS emotion labels...")
    
    data_dir = Path("data/ravdess")
    
    # Find all audio files
    audio_files = list(data_dir.glob("**/*.wav"))
    
    if not audio_files:
        print("❌ No audio files found. Please download RAVDESS dataset first.")
        return None
    
    # Parse metadata
    metadata = []
    for audio_file in audio_files:
        parsed = parse_ravdess_filename(audio_file)
        if parsed:
            parsed['filepath'] = str(audio_file)
            metadata.append(parsed)
    
    df = pd.DataFrame(metadata)
    
    # Create one-hot emotion labels (for your 26 emotion model)
    # RAVDESS has 8 emotions, you'll need to map to your 26
    emotion_mapping = {
        'neutral': [1.0] + [0.0] * 25,    # emo1 = neutral
        'happy': [0.0, 1.0] + [0.0] * 24, # emo2 = happy  
        'sad': [0.0, 0.0, 1.0] + [0.0] * 23, # emo3 = sad
        'angry': [0.0, 0.0, 0.0, 1.0] + [0.0] * 22, # emo4 = angry
        # Add mappings for all 8 RAVDESS emotions to your 26
    }
    
    # Create emotion labels
    emotion_labels = []
    for _, row in df.iterrows():
        emotion = row['emotion']
        if emotion in emotion_mapping:
            emotion_labels.append(emotion_mapping[emotion])
        else:
            emotion_labels.append([0.5] * 26)  # Default neutral scores
    
    df['emotion_labels'] = emotion_labels
    
    # Save metadata
    df.to_csv('data/ravdess/ravdess_metadata.csv', index=False)
    
    print(f"✅ Processed {len(df)} RAVDESS audio files")
    return df

if __name__ == "__main__":
    download_ravdess()
    create_ravdess_labels()