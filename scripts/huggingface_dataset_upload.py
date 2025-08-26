#!/usr/bin/env python3
"""
Upload and process CSVs using Hugging Face Datasets for emotion mapping and data cleaning.
"""

import pandas as pd
import os
from typing import Dict, Optional
import logging

# Note: Install these packages when needed:
# pip install datasets huggingface_hub

logger = logging.getLogger(__name__)

class HuggingFaceDatasetManager:
    """Manager for uploading and processing datasets on Hugging Face."""
    
    def __init__(self, hf_token: Optional[str] = None):
        """
        Initialize with Hugging Face token.
        
        Args:
            hf_token: Hugging Face API token (or set HF_TOKEN environment variable)
        """
        self.api = HfApi(token=hf_token)
        
        # Emotion mapping from emo codes to human-readable labels
        self.emotion_mapping = {
            'emo1': 'happiness',
            'emo2': 'sadness', 
            'emo3': 'anger',
            'emo4': 'fear',
            'emo5': 'neutral',
            'emo6': 'love',
            'emo7': 'surprise',
            'emo8': 'disgust',
            'emo9': 'grief',
            'emo10': 'anxiety',
            'emo11': 'loneliness',
            'emo12': 'contempt',
            'emo13': 'pride',
            'emo14': 'excitement',
            'emo15': 'confusion',
            'emo16': 'shame',
            'emo17': 'guilt',
            'emo18': 'jealousy',
            'emo19': 'envy',
            'emo20': 'anticipation',
            'emo21': 'hope',
            'emo22': 'despair',
            'emo23': 'curiosity',
            'emo24': 'boredom',
            'emo25': 'relief',
            'emo26': 'nostalgia',
            # Add emo27-emo38 mappings based on your documentation
            'emo27': 'empathy',
            'emo28': 'compassion',
            'emo29': 'gratitude',
            'emo30': 'forgiveness',
            'emo31': 'optimism',
            'emo32': 'pessimism',
            'emo33': 'confidence',
            'emo34': 'insecurity',
            'emo35': 'determination',
            'emo36': 'resignation',
            'emo37': 'serenity',
            'emo38': 'agitation'
        }

    def upload_csv_to_dataset(self, csv_path: str, dataset_name: str, description: str = "") -> str:
        """
        Upload CSV file to Hugging Face Datasets.
        
        Args:
            csv_path: Path to local CSV file
            dataset_name: Name for the dataset (format: username/dataset-name)
            description: Description of the dataset
            
        Returns:
            Dataset repository URL
        """
        try:
            # Load CSV into pandas DataFrame
            df = pd.read_csv(csv_path)
            logger.info(f"Loaded CSV with {len(df)} rows and {len(df.columns)} columns")
            
            # Convert to Hugging Face Dataset
            dataset = Dataset.from_pandas(df)
            
            # Create dataset repository
            repo_url = self.api.create_repo(
                repo_id=dataset_name,
                repo_type="dataset",
                exist_ok=True
            )
            
            # Push dataset to Hub
            dataset.push_to_hub(dataset_name)
            
            logger.info(f"✅ Dataset uploaded successfully: {repo_url}")
            return repo_url
            
        except Exception as e:
            logger.error(f"Error uploading dataset: {str(e)}")
            raise

    def apply_emotion_mapping(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply emotion code to human-readable label mapping.
        
        Args:
            df: DataFrame with emotion columns (emo1, emo2, etc.)
            
        Returns:
            DataFrame with additional human-readable emotion columns
        """
        mapped_df = df.copy()
        
        # Add human-readable emotion columns
        for emo_code, emotion_label in self.emotion_mapping.items():
            if emo_code in df.columns:
                # Create new column with human-readable name
                mapped_df[f"{emotion_label}_score"] = df[emo_code]
                
        logger.info(f"Applied emotion mapping for {len(self.emotion_mapping)} emotions")
        return mapped_df

    def create_emotion_labels_dataset(self, output_path: str = "emotion_labels.csv") -> str:
        """
        Create a reference dataset with emotion code mappings.
        
        Args:
            output_path: Path to save the emotion labels CSV
            
        Returns:
            Path to the created file
        """
        emotion_df = pd.DataFrame([
            {"emotion_code": code, "emotion_label": label, "category": self._categorize_emotion(label)}
            for code, label in self.emotion_mapping.items()
        ])
        
        emotion_df.to_csv(output_path, index=False)
        logger.info(f"Created emotion labels dataset: {output_path}")
        return output_path

    def _categorize_emotion(self, emotion: str) -> str:
        """Categorize emotions into basic groups for better organization."""
        positive_emotions = ['happiness', 'love', 'excitement', 'pride', 'hope', 'relief', 'gratitude', 'optimism', 'confidence', 'determination', 'serenity']
        negative_emotions = ['sadness', 'anger', 'fear', 'grief', 'anxiety', 'loneliness', 'shame', 'guilt', 'despair', 'pessimism', 'insecurity', 'resignation']
        social_emotions = ['contempt', 'jealousy', 'envy', 'empathy', 'compassion', 'forgiveness']
        
        if emotion in positive_emotions:
            return "positive"
        elif emotion in negative_emotions:
            return "negative" 
        elif emotion in social_emotions:
            return "social"
        else:
            return "neutral"


def main():
    """Example usage of HuggingFace dataset upload and processing."""
    
    # Initialize manager
    manager = HuggingFaceDatasetManager()
    
    # Example 1: Create emotion labels reference dataset
    emotion_labels_path = manager.create_emotion_labels_dataset()
    print(f"Created emotion labels: {emotion_labels_path}")
    
    # Example 2: Process a local CSV with emotion mapping
    if os.path.exists("voital.csv"):
        df = pd.read_csv("voital.csv")
        mapped_df = manager.apply_emotion_mapping(df)
        mapped_df.to_csv("voital_mapped.csv", index=False)
        print("Created mapped dataset: voital_mapped.csv")
        
        # Upload to Hugging Face (uncomment when ready)
        # manager.upload_csv_to_dataset("voital_mapped.csv", "your-username/voice-emotion-dataset")
    
    print("\n🚀 Next steps:")
    print("1. Set HF_TOKEN environment variable with your Hugging Face token")
    print("2. Upload your CSV: manager.upload_csv_to_dataset('your_file.csv', 'username/dataset-name')")
    print("3. Use Hugging Face Datasets viewer UI to explore and clean data")
    print("4. Download processed dataset back for training")


if __name__ == "__main__":
    main()