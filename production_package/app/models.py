"""
Machine learning models for emotion and trait prediction.
Uses real trained Random Forest models on RAVDESS dataset.
"""

import numpy as np
from typing import Dict, Any, Tuple
import logging
import joblib
import os

from .utils import get_env_var

logger = logging.getLogger(__name__)


class EmotionModel:
    """Real emotion prediction model using trained Random Forest"""
    
    def __init__(self):
        self.version = "emotion_rf_v1"
        self.emotion_keys = [f"emo{i}" for i in range(1, 27)]  # emo1 to emo26
        self.meta_emotion_keys = ["emo29", "emo31", "emo32"]  # Customer pillars
        
        # Load trained model
        model_path = "models/emotion_model_rf.pkl"
        if os.path.exists(model_path):
            self.model = joblib.load(model_path)
            logger.info(f"✅ Loaded trained emotion model from {model_path}")
        else:
            logger.warning(f"❌ Model file {model_path} not found! Using stub model.")
            self.model = None
    
    def predict(self, features: np.ndarray, audio_metadata: Dict[str, Any]) -> Dict[str, float]:
        """
        Predict emotion scores using trained model.
        """
        if self.model is None:
            # Fallback to stub if no trained model
            logger.warning("Using stub emotion model - train real model first!")
            return self._stub_predict(features)
        
        try:
            # Reshape features for model input
            features_2d = features.reshape(1, -1)
            
            # Get predictions from trained model
            predictions = self.model.predict(features_2d)[0]  # Get first (and only) prediction
            
            # Ensure we have exactly 26 emotion scores
            if len(predictions) != 26:
                logger.error(f"Model returned {len(predictions)} scores, expected 26")
                return self._stub_predict(features)
            
            # Clip to valid range [0, 1]
            predictions = np.clip(predictions, 0.0, 1.0)
            
            # Create emotion dictionary
            emotion_scores = {
                key: float(score) for key, score in zip(self.emotion_keys, predictions)
            }
            
            # Add meta-emotions for customer pillars
            meta_emotions = self._calculate_meta_emotions(emotion_scores, audio_metadata)
            emotion_scores.update(meta_emotions)
            
            logger.debug(f"✅ Real emotion prediction complete: {len(emotion_scores)} emotions (includes {len(meta_emotions)} meta)")
            return emotion_scores
            
        except Exception as e:
            logger.error(f"Error in real emotion prediction: {e}")
            return self._stub_predict(features)
    
    def _stub_predict(self, features: np.ndarray) -> Dict[str, float]:
        """Fallback stub prediction if real model fails"""
        # Same stub logic as before
        feature_hash = hash(features.tobytes()) % 1000000
        np.random.seed(feature_hash)
        base_scores = np.random.beta(2, 2, len(self.emotion_keys))
        emotion_scores = {key: float(score) for key, score in zip(self.emotion_keys, base_scores)}
        
        # Add meta-emotions for customer pillars
        meta_emotions = self._calculate_meta_emotions(emotion_scores, {})
        emotion_scores.update(meta_emotions)
        
        return emotion_scores
    
    def _calculate_meta_emotions(self, base_emotions: Dict[str, float], audio_metadata: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate meta-emotions (emo29, emo31, emo32) from base emotions.
        These are the customer's required pillars.
        """
        
        # emo29: positive_overall (0-300 scale) - Customer calls this "empathy"
        # Calculate from positive emotions: happiness, calm, love, confidence
        positive_emotions = ['emo1', 'emo5', 'emo6', 'emo17']  # happiness, calm, love, confidence
        positive_score = sum(base_emotions.get(emo, 0.0) for emo in positive_emotions) / len(positive_emotions)
        emo29_raw = positive_score * 300.0  # Scale to 0-300 (original scale)
        emo29 = min(100.0, emo29_raw / 3.0)  # Normalize to 0-100 for customer compatibility
        
        # emo31: valence_overall (0-100 scale) - Customer calls this "self_expression"
        # Balance of positive vs negative emotions
        negative_emotions = ['emo4', 'emo9', 'emo11', 'emo19']  # anger, sadness, lonely, fear
        negative_score = sum(base_emotions.get(emo, 0.0) for emo in negative_emotions) / len(negative_emotions)
        emo31 = max(0.0, min(100.0, (positive_score - negative_score) * 100.0))  # Already 0-100 scale
        
        # emo32: control_overall (0-300 scale) - Customer calls this "self_management"
        # Emotional self-control based on calm/confidence minus stress/overwhelm
        control_indicators = ['emo5', 'emo17']  # calm, confidence  
        stress_indicators = ['emo8', 'emo21']   # stress, overwhelmed
        control_score = sum(base_emotions.get(emo, 0.0) for emo in control_indicators) / len(control_indicators)
        stress_score = sum(base_emotions.get(emo, 0.0) for emo in stress_indicators) / len(stress_indicators)
        emo32_raw = max(0.0, control_score - stress_score) * 300.0  # Scale to 0-300 (original scale)
        emo32 = min(100.0, emo32_raw / 3.0)  # Normalize to 0-100 for customer compatibility
        
        return {
            'emo29': emo29,  # empathy (normalized 0-100)
            'emo31': emo31,  # self_expression (0-100)
            'emo32': emo32   # self_management (normalized 0-100)
        }


class TraitModel:
    """Personality trait prediction model (char1-char94)"""
    
    def __init__(self):
        self.version = "trait_rf_v1"
        self.trait_keys = [f"char{i}" for i in range(1, 95)]  # char1 to char94
        
        # Load trained model
        model_path = "models/trait_model_rf.pkl"
        if os.path.exists(model_path):
            self.model = joblib.load(model_path)
            logger.info(f"✅ Loaded trained trait model from {model_path}")
        else:
            logger.warning(f"❌ Model file {model_path} not found! Using stub model.")
            self.model = None
        
    def predict(self, features: np.ndarray, audio_metadata: Dict[str, Any]) -> Dict[str, float]:
        """
        Predict trait scores using trained model.
        """
        if self.model is None:
            logger.warning("Using stub trait model - train real model first!")
            return self._stub_predict(features)
        
        try:
            features_2d = features.reshape(1, -1)
            predictions = self.model.predict(features_2d)[0]
            
            if len(predictions) != 94:
                logger.error(f"Model returned {len(predictions)} scores, expected 94")
                return self._stub_predict(features)
            
            predictions = np.clip(predictions, 0.0, 1.0)
            
            trait_scores = {
                key: float(score) for key, score in zip(self.trait_keys, predictions)
            }
            
            logger.debug(f"✅ Real trait prediction complete: {len(trait_scores)} traits")
            return trait_scores
            
        except Exception as e:
            logger.error(f"Error in real trait prediction: {e}")
            return self._stub_predict(features)
    
    def _stub_predict(self, features: np.ndarray) -> Dict[str, float]:
        """Fallback stub prediction"""
        feature_hash = hash(features.tobytes()) % 1000000
        np.random.seed(feature_hash + 42)
        base_scores = np.random.beta(1.5, 1.5, len(self.trait_keys))
        return {key: float(score) for key, score in zip(self.trait_keys, base_scores)}


class ModelManager:
    """Manages all prediction models"""
    
    def __init__(self):
        self.emotion_model = EmotionModel()
        self.trait_model = TraitModel()
        self.version = get_env_var("MODEL_VERSION", "emotion_v1;traits_v1")
        
    def predict_all(self, features: np.ndarray, audio_metadata: Dict[str, Any]) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Run predictions for both emotions and traits.
        
        Args:
            features: Extracted audio feature vector
            audio_metadata: Audio processing metadata
            
        Returns:
            Tuple of (emotion_scores, trait_scores)
        """
        try:
            # Predict emotions
            emotion_scores = self.emotion_model.predict(features, audio_metadata)
            
            # Predict traits
            trait_scores = self.trait_model.predict(features, audio_metadata)
            
            return emotion_scores, trait_scores
            
        except Exception as e:
            logger.error(f"Error in model predictions: {e}")
            # Return default scores on error
            emotion_scores = {f"emo{i}": 0.5 for i in range(1, 27)}
            trait_scores = {f"char{i}": 0.5 for i in range(1, 95)}
            return emotion_scores, trait_scores
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information for metadata"""
        return {
            'version': self.version,
            'emotion_model': self.emotion_model.version,
            'trait_model': self.trait_model.version,
            'emotion_outputs': len(self.emotion_model.emotion_keys),
            'trait_outputs': len(self.trait_model.trait_keys)
        }


# Keep original ModelManager for fallback compatibility  
class OriginalModelManager:
    """Original model manager for compatibility"""
    
    def __init__(self):
        self.emotion_model = EmotionModel()
        self.trait_model = TraitModel()
        self.version = get_env_var("MODEL_VERSION", "emotion_v1;traits_v1")
        
    def predict_all(self, features: np.ndarray, audio_metadata: Dict[str, Any]) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Run predictions for both emotions and traits."""
        try:
            emotion_scores = self.emotion_model.predict(features, audio_metadata)
            trait_scores = self.trait_model.predict(features, audio_metadata)
            return emotion_scores, trait_scores
        except Exception as e:
            logger.error(f"Error in model predictions: {e}")
            emotion_scores = {f"emo{i}": 0.5 for i in range(1, 27)}
            trait_scores = {f"char{i}": 0.5 for i in range(1, 95)}
            return emotion_scores, trait_scores
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information for metadata"""
        return {
            'version': self.version,
            'emotion_model': self.emotion_model.version,
            'trait_model': self.trait_model.version,
            'emotion_outputs': len(self.emotion_model.emotion_keys),
            'trait_outputs': len(self.trait_model.trait_keys)
        }

# Original model manager (available as fallback)
original_model_manager = OriginalModelManager()


# INSTRUCTIONS FOR REAL MODEL INTEGRATION:
# 
# To replace stub models with real models:
# 
# 1. For ONNX models:
#    - Install onnxruntime: pip install onnxruntime
#    - Load model: self.session = onnxruntime.InferenceSession("model.onnx")
#    - Run inference: outputs = self.session.run(None, {"input": features})
# 
# 2. For PyTorch models:
#    - Install torch: pip install torch
#    - Load model: self.model = torch.load("model.pt"); self.model.eval()
#    - Run inference: with torch.no_grad(): outputs = self.model(torch.tensor(features))
# 
# 3. For external API calls:
#    - Use httpx to call GPU inference endpoints
#    - Maintain same input/output signature
#    - Add proper error handling and timeouts
# 
# 4. Key requirements:
#    - Input: feature vector (float32 array)
#    - Output: Dict[str, float] with exact key names (emo1-emo26, char1-char94)
#    - Scores should be in range [0.0, 1.0]
#    - Handle errors gracefully with neutral scores
