#!/usr/bin/env python3
"""
Download RAVDESS dataset using KaggleHub - much easier!
"""

import kagglehub
import shutil
import os
from pathlib import Path

def download_ravdess_with_kagglehub():
    """Download RAVDESS dataset using kagglehub"""
    print("🚀 Downloading RAVDESS dataset with KaggleHub...")
    
    try:
        # Download the RAVDESS emotional speech dataset
        path = kagglehub.dataset_download("uwrfkaggler/ravdess-emotional-speech-audio")
        
        print(f"✅ Downloaded to: {path}")
        
        # Move/copy to our expected location
        target_dir = Path("data/ravdess")
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy all files from downloaded path to our target
        source_path = Path(path)
        
        print("📂 Organizing files...")
        for file in source_path.glob("**/*"):
            if file.is_file():
                # Copy to target directory, preserving structure
                relative_path = file.relative_to(source_path)
                target_file = target_dir / relative_path
                target_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(file, target_file)
        
        # Count audio files
        audio_files = list(target_dir.glob("**/*.wav"))
        print(f"🎵 Found {len(audio_files)} audio files in data/ravdess/")
        
        if len(audio_files) > 0:
            print("✅ RAVDESS dataset ready!")
            print("🔄 Next step: python process_ravdess_training.py")
            return True
        else:
            print("❌ No audio files found. Checking directory structure...")
            print("📁 Contents of downloaded directory:")
            for item in source_path.glob("*"):
                print(f"  - {item.name}")
            return False
            
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        print("💡 You might need to authenticate with Kaggle first:")
        print("   - pip install kaggle")
        print("   - Get your API key from kaggle.com/settings")
        print("   - Set up authentication")
        return False

if __name__ == "__main__":
    print("🎵 RAVDESS Dataset Downloader")
    print("=" * 40)
    
    if download_ravdess_with_kagglehub():
        print("\n🎉 Success! Ready to train models.")
    else:
        print("\n❌ Download failed. Please check authentication.")