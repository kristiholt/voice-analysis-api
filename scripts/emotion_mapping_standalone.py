#!/usr/bin/env python3
"""
Standalone emotion mapping and CSV processing without external dependencies.
Use this to prepare data before uploading to Hugging Face manually.
"""

import pandas as pd
import os
import json
from typing import Dict
import logging

logger = logging.getLogger(__name__)

class EmotionMapper:
    """Standalone emotion mapping for CSV processing."""
    
    def __init__(self):
        """Initialize with comprehensive emotion mappings."""
        
        # Complete emotion mapping from emo1-emo38 to human-readable labels
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
        
        # Character trait mapping (char1-char94) - examples based on Big Five + others
        self.trait_mapping = {
            'char1': 'loss_of_selfworth',
            'char2': 'emotional_instability',
            'char3': 'social_anxiety',
            'char4': 'impulsiveness',
            'char5': 'perfectionism',
            # Add more based on your VoxCentai documentation
            # ... char6-char94 mappings
        }

    def process_csv_with_mapping(self, input_csv: str, output_csv: str) -> pd.DataFrame:
        """
        Process CSV with emotion and trait mapping for Hugging Face upload.
        
        Args:
            input_csv: Path to input CSV file
            output_csv: Path to save processed CSV
            
        Returns:
            Processed DataFrame
        """
        try:
            # Load original CSV
            df = pd.read_csv(input_csv)
            logger.info(f"Loaded CSV with {len(df)} rows and {len(df.columns)} columns")
            
            # Create a copy for processing
            processed_df = df.copy()
            
            # Add human-readable emotion columns
            for emo_code, emotion_label in self.emotion_mapping.items():
                if emo_code in df.columns:
                    # Create new column with human-readable name
                    processed_df[f"{emotion_label}_score"] = df[emo_code]
                    # Optionally keep original emo codes for reference
                    processed_df[f"{emo_code}_original"] = df[emo_code]
            
            # Add human-readable trait columns
            for trait_code, trait_label in self.trait_mapping.items():
                if trait_code in df.columns:
                    processed_df[f"{trait_label}_score"] = df[trait_code]
                    processed_df[f"{trait_code}_original"] = df[trait_code]
            
            # Add metadata columns for better organization
            processed_df['total_emotions'] = len([col for col in df.columns if col.startswith('emo')])
            processed_df['total_traits'] = len([col for col in df.columns if col.startswith('char')])
            
            # Save processed CSV
            processed_df.to_csv(output_csv, index=False)
            logger.info(f"✅ Processed CSV saved: {output_csv}")
            
            return processed_df
            
        except Exception as e:
            logger.error(f"Error processing CSV: {str(e)}")
            raise

    def create_emotion_reference(self, output_file: str = "emotion_reference.csv") -> str:
        """Create a reference file for emotion mappings."""
        
        emotion_data = []
        for code, label in self.emotion_mapping.items():
            emotion_data.append({
                'emotion_code': code,
                'emotion_label': label,
                'category': self._categorize_emotion(label),
                'description': f"Emotion score for {label}"
            })
        
        emotion_df = pd.DataFrame(emotion_data)
        emotion_df.to_csv(output_file, index=False)
        logger.info(f"Created emotion reference: {output_file}")
        return output_file

    def create_trait_reference(self, output_file: str = "trait_reference.csv") -> str:
        """Create a reference file for trait mappings."""
        
        trait_data = []
        for code, label in self.trait_mapping.items():
            trait_data.append({
                'trait_code': code,
                'trait_label': label,
                'category': 'personality_trait',
                'description': f"Personality trait score for {label}"
            })
        
        trait_df = pd.DataFrame(trait_data)
        trait_df.to_csv(output_file, index=False)
        logger.info(f"Created trait reference: {output_file}")
        return output_file

    def export_mapping_json(self, output_file: str = "emotion_trait_mapping.json") -> str:
        """Export mappings as JSON for easy reference."""
        
        mapping_data = {
            'emotions': self.emotion_mapping,
            'traits': self.trait_mapping,
            'metadata': {
                'total_emotions': len(self.emotion_mapping),
                'total_traits': len(self.trait_mapping),
                'version': 'v1.0'
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(mapping_data, f, indent=2)
        
        logger.info(f"Exported mappings to JSON: {output_file}")
        return output_file

    def _categorize_emotion(self, emotion: str) -> str:
        """Categorize emotions for better organization."""
        positive = ['happiness', 'love', 'excitement', 'pride', 'hope', 'relief', 'gratitude', 'optimism', 'confidence', 'determination', 'serenity']
        negative = ['sadness', 'anger', 'fear', 'grief', 'anxiety', 'loneliness', 'shame', 'guilt', 'despair', 'pessimism', 'insecurity', 'resignation']
        social = ['contempt', 'jealousy', 'envy', 'empathy', 'compassion', 'forgiveness']
        
        if emotion in positive:
            return "positive"
        elif emotion in negative:
            return "negative" 
        elif emotion in social:
            return "social"
        else:
            return "neutral"


def main():
    """Example usage for emotion mapping and CSV processing."""
    
    # Initialize mapper
    mapper = EmotionMapper()
    
    print("🎯 Emotion and Trait Mapping Tool")
    print("=" * 40)
    
    # Create reference files
    emotion_ref = mapper.create_emotion_reference()
    trait_ref = mapper.create_trait_reference()
    mapping_json = mapper.export_mapping_json()
    
    print(f"✅ Created reference files:")
    print(f"   - {emotion_ref}")
    print(f"   - {trait_ref}")
    print(f"   - {mapping_json}")
    
    # Process existing CSV if available
    if os.path.exists("voital.csv"):
        processed_df = mapper.process_csv_with_mapping("voital.csv", "voital_mapped_for_hf.csv")
        print(f"✅ Processed voital.csv -> voital_mapped_for_hf.csv")
        print(f"   Original columns: {len(processed_df.columns)}")
        print(f"   Rows: {len(processed_df)}")
    
    print("\n🚀 Next Steps for Hugging Face Upload:")
    print("1. Go to https://huggingface.co/new-dataset")
    print("2. Upload your processed CSV files")
    print("3. Use the HF dataset viewer to explore and clean data")
    print("4. Add description and tags for discoverability")
    print("5. Download cleaned dataset for training")
    
    print("\n📊 Files ready for upload:")
    print("   - voital_mapped_for_hf.csv (main dataset)")
    print("   - emotion_reference.csv (emotion mappings)")
    print("   - trait_reference.csv (trait mappings)")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()