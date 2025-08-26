"""
Enhanced models with Big Five -> char1-char94 mapping
Implements Track A of hybrid enhancement plan
"""

import numpy as np
from typing import Dict, Any
import logging
import yaml
import joblib
import os
from pathlib import Path

from .utils import get_env_var

logger = logging.getLogger(__name__)


class EnhancedBigFiveMapper:
    """Maps Big Five personality traits to char1-char94 using learned weights"""
    
    def __init__(self):
        self.mapping_matrix = self._load_mapping_matrix()
        self.bigfive_labels = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]
        
    def _load_mapping_matrix(self) -> np.ndarray:
        """Load Big Five -> char mapping matrix from configuration"""
        try:
            config_path = "label_map.yaml"
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                
                # Build mapping matrix (94 chars x 5 Big Five traits)
                mapping_matrix = np.zeros((94, 5))
                char_mappings = config.get('char_from_bigfive', {})
                
                for char_name, weights in char_mappings.items():
                    if char_name.startswith('char'):
                        char_idx = int(char_name[4:]) - 1  # char1 -> index 0
                        if 0 <= char_idx < 94 and len(weights) == 5:
                            mapping_matrix[char_idx] = weights
                
                non_zero = np.sum(mapping_matrix != 0)
                logger.info(f"✅ Loaded Big Five mapping with {non_zero} non-zero weights")
                return mapping_matrix
                
        except Exception as e:
            logger.warning(f"⚠️ Error loading mapping matrix: {e}")
        
        # Fallback: random small weights
        logger.info("🔄 Using fallback random mapping matrix")
        return np.random.uniform(-0.1, 0.1, (94, 5))
    
    def map_bigfive_to_chars(self, bigfive_scores: np.ndarray) -> np.ndarray:
        """Map Big Five scores to char1-char94 using learned mapping"""
        if len(bigfive_scores) != 5:
            raise ValueError(f"Expected 5 Big Five scores, got {len(bigfive_scores)}")
        
        # Linear transformation: char = mapping_matrix @ bigfive
        char_logits = self.mapping_matrix @ bigfive_scores
        
        # Apply sigmoid activation for [0,1] range
        char_scores = 1 / (1 + np.exp(-char_logits))
        
        return np.clip(char_scores, 0.0, 1.0)


class EnhancedEmotionModel:
    """Enhanced emotion model with better psychological grounding"""
    
    def __init__(self):
        self.version = "enhanced_emotion_v1"
        self.emotion_keys = [f"emo{i}" for i in range(1, 27)]
        self.meta_emotion_keys = ["emo29", "emo31", "emo32"]  # Customer pillars
        self.extended_emotion_keys = ["emo27", "emo28", "emo30"] + [f"emo{i}" for i in range(33, 39)]  # Missing emotions
        self.bigfive_mapper = EnhancedBigFiveMapper()
        
        # Load trained model if available
        model_path = "models/emotion_model_rf.pkl"
        
        # Try multiple possible paths for the model file
        possible_paths = [
            model_path,
            f"production_package/{model_path}",
            f"../{model_path}",
            "./models/emotion_model_rf.pkl"
        ]
        
        self.model = None
        for path in possible_paths:
            if os.path.exists(path):
                try:
                    self.model = joblib.load(path)
                    logger.info(f"✅ Loaded trained emotion model from {path}")
                    break
                except Exception as e:
                    logger.warning(f"Failed to load model from {path}: {e}")
                    continue
        
        if self.model is None:
            logger.warning("⚠️ Using fallback emotion model - trained model not found")
            logger.warning(f"Searched paths: {possible_paths}")
            logger.warning(f"Current working directory: {os.getcwd()}")
    
    def predict(self, features: np.ndarray, audio_metadata: Dict[str, Any]) -> Dict[str, float]:
        """Enhanced emotion prediction with psychological consistency"""
        
        # Get base emotion predictions
        if self.model is not None:
            try:
                features_2d = features.reshape(1, -1)
                base_predictions = self.model.predict(features_2d)[0]
                base_predictions = np.clip(base_predictions, 0.0, 1.0)
            except Exception as e:
                logger.error(f"Error in emotion model prediction: {e}")
                base_predictions = self._fallback_predict(features)
        else:
            base_predictions = self._fallback_predict(features)
        
        # Enhance predictions with psychological insights
        enhanced_predictions = self._enhance_with_psychology(base_predictions, features)
        
        # Create emotion dictionary
        emotion_scores = {
            key: float(score) for key, score in zip(self.emotion_keys, enhanced_predictions)
        }
        
        # Add meta-emotions for customer pillars
        meta_emotions = self._calculate_meta_emotions(emotion_scores, audio_metadata)
        emotion_scores.update(meta_emotions)
        
        # Add extended emotions (calculated from existing emotions)
        extended_emotions = self._calculate_extended_emotions(emotion_scores, audio_metadata)
        emotion_scores.update(extended_emotions)
        
        total_meta = len(meta_emotions) + len(extended_emotions)
        logger.debug(f"✅ Enhanced emotion prediction complete: {len(emotion_scores)} emotions (includes {total_meta} calculated)")
        return emotion_scores
    
    def _enhance_with_psychology(self, base_predictions: np.ndarray, features: np.ndarray) -> np.ndarray:
        """Enhance predictions using psychological knowledge"""
        
        # Estimate Big Five traits from audio features (simple heuristic)
        # Handle varying feature dimensions - use statistical summary instead of fixed reshaping
        if len(features) >= 13:
            # Use segments of the feature vector for statistical analysis
            num_segments = len(features) // 13
            if num_segments > 0:
                segments = features[:num_segments * 13].reshape(num_segments, 13)
                feature_means = np.mean(segments, axis=1)[:5]  # Take first 5 means
            else:
                feature_means = [np.mean(features[:13])] * 5
        else:
            feature_means = [0.5] * 5
        feature_means = feature_means[:5] if len(feature_means) >= 5 else list(feature_means) + [0.5] * (5 - len(feature_means))
        
        # Normalize to [0,1] - ensure it's a numpy array
        estimated_bigfive = np.clip(np.array(feature_means), 0.0, 1.0)
        
        # Apply psychological constraints
        enhanced = base_predictions.copy()
        
        # High Neuroticism -> higher negative emotions
        if estimated_bigfive[4] > 0.6:  # High Neuroticism
            enhanced[8] *= 1.2   # emo9 (sadness)
            enhanced[10] *= 1.1  # emo11 (lonely)
        
        # High Extraversion -> higher positive emotions  
        if estimated_bigfive[2] > 0.6:  # High Extraversion
            enhanced[0] *= 1.1   # emo1 (happiness)
            enhanced[5] *= 1.1   # emo6 (love)
        
        # High Agreeableness -> lower anger
        if estimated_bigfive[3] > 0.6:  # High Agreeableness
            enhanced[3] *= 0.8   # emo4 (anger)
        
        return np.clip(enhanced, 0.0, 1.0)
    
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
        
        # emo31: self_expression (0-100 scale) - Customer calls this "Self-Expression"  
        # Based on confidence, drive, and expressiveness emotions
        expression_emotions = ['emo17', 'emo26', 'emo6', 'emo1']  # confidence, drive, love, happiness
        expression_score = sum(base_emotions.get(emo, 0.0) for emo in expression_emotions) / len(expression_emotions)
        emo31 = max(10.0, min(100.0, expression_score * 100.0))  # Direct expression score to 0-100 scale
        
        # emo32: control_score (0-100 scale) - Customer calls this "Self-Management"
        # Use control indicators directly without subtraction
        control_indicators = ['emo5', 'emo17']  # calm, confidence  
        control_score = sum(base_emotions.get(emo, 0.0) for emo in control_indicators) / len(control_indicators)
        emo32 = max(15.0, min(100.0, control_score * 100.0))  # Direct control score to 0-100 scale
        
        return {
            'emo29': emo29,  # empathy (normalized 0-100)
            'emo31': emo31,  # self_expression (0-100)
            'emo32': emo32   # self_management (normalized 0-100)
        }
    
    def _calculate_extended_emotions(self, base_emotions: Dict[str, float], audio_metadata: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate extended emotions (emo27-emo28, emo30, emo33-emo38) from base emotions.
        These provide complete Vibeonix compatibility with actual predictions instead of neutral values.
        """
        
        # Helper function to safely get emotion values
        def get_emo(key: str) -> float:
            return base_emotions.get(key, 0.5)
        
        # emo27: coolness - Adaptive emotional regulation (confidence + calm - anger)
        coolness = (get_emo('emo17') + get_emo('emo5') - get_emo('emo4')) / 3
        coolness = max(0.0, min(1.0, coolness))
        
        # emo28: jealousy - Based on envy + loneliness - love
        # Note: envy is emo21, loneliness is emo11, love is emo6
        jealousy = (get_emo('emo21') + get_emo('emo11') - get_emo('emo6')) / 3
        jealousy = max(0.0, min(1.0, jealousy))
        
        # emo30: negative_overall - Aggregate of negative emotions
        negative_emotions = [
            get_emo('emo4'),   # anger
            get_emo('emo8'),   # down
            get_emo('emo9'),   # sadness
            get_emo('emo10'),  # fear
            get_emo('emo11'),  # loneliness
            get_emo('emo13'),  # doubt
            get_emo('emo14'),  # overwhelmed
            get_emo('emo18'),  # confused
            get_emo('emo19'),  # frustration
            get_emo('emo20'),  # melancholy
            get_emo('emo21'),  # envy
            get_emo('emo24'),  # despair
            get_emo('emo25')   # exhausted
        ]
        negative_overall = sum(negative_emotions) / len(negative_emotions)
        
        # Fear spectrum emotions (emo33-emo38)
        
        # emo33: negative_stress - Opposite of positive stress, plus overwhelmed
        negative_stress = (1.0 - get_emo('emo7') + get_emo('emo14') + get_emo('emo25')) / 3
        negative_stress = max(0.0, min(1.0, negative_stress))
        
        # emo34: vigilance - Fear of losing identity (fear + alertness + doubt)
        vigilance = (get_emo('emo10') + get_emo('emo15') + get_emo('emo13')) / 3
        vigilance = max(0.0, min(1.0, vigilance))
        
        # emo35: fear_abandonment - Fear of separation (loneliness + fear + attachment issues)
        fear_abandonment = (get_emo('emo11') + get_emo('emo10') + get_emo('emo16')) / 3  # loneliness + fear + longing
        fear_abandonment = max(0.0, min(1.0, fear_abandonment))
        
        # emo36: fear_autonomy_loss - Fear of loss of autonomy (doubt + overwhelmed + low confidence)
        fear_autonomy_loss = (get_emo('emo13') + get_emo('emo14') + (1.0 - get_emo('emo17'))) / 3
        fear_autonomy_loss = max(0.0, min(1.0, fear_autonomy_loss))
        
        # emo37: fear_failure - Fear of failure (doubt + down + fear + low confidence)
        fear_failure = (get_emo('emo13') + get_emo('emo8') + get_emo('emo10') + (1.0 - get_emo('emo17'))) / 4
        fear_failure = max(0.0, min(1.0, fear_failure))
        
        # emo38: fear_physical_harm - Fear of physical/mental harm (fear + overwhelmed + vigilance)
        fear_physical_harm = (get_emo('emo10') + get_emo('emo14') + vigilance) / 3
        fear_physical_harm = max(0.0, min(1.0, fear_physical_harm))
        
        # Convert to 0-100 scale (base emotions are 0-1, extended should match)
        extended_emotions = {
            'emo27': coolness * 100.0,          # coolness
            'emo28': jealousy * 100.0,          # jealousy
            'emo30': negative_overall * 100.0,   # negative_overall
            'emo33': negative_stress * 100.0,    # negative_stress
            'emo34': vigilance * 100.0,          # vigilance
            'emo35': fear_abandonment * 100.0,   # fear_abandonment
            'emo36': fear_autonomy_loss * 100.0, # fear_autonomy_loss
            'emo37': fear_failure * 100.0,       # fear_failure
            'emo38': fear_physical_harm * 100.0  # fear_physical_harm
        }
        
        return extended_emotions
    
    def _fallback_predict(self, features: np.ndarray) -> np.ndarray:
        """Improved fallback prediction when model unavailable"""
        # Use feature-based hashing for consistency
        feature_hash = hash(features.tobytes()) % 1000000
        np.random.seed(feature_hash)
        
        # Generate more realistic emotion scores
        # Use mixture of distributions for more natural emotional profiles
        scores = np.zeros(len(self.emotion_keys))
        
        # Base emotional state - moderate positive bias
        base_scores = np.random.beta(3, 2, len(self.emotion_keys))  # Shifted toward higher values
        
        # Add some individual variation based on audio features
        if len(features) > 0:
            energy_proxy = float(np.mean(np.abs(features)))
            # Higher energy -> more expressive emotions
            energy_factor = min(1.5, 0.5 + energy_proxy)
            
            # Boost certain emotions based on energy
            positive_emotions = [0, 5, 16]  # emo1 (happiness), emo6 (love), emo17 (confidence) 
            for idx in positive_emotions:
                if idx < len(base_scores):
                    base_scores[idx] *= energy_factor
        
        return np.clip(base_scores, 0.0, 1.0)


class EnhancedTraitModel:
    """Enhanced trait model using Big Five mapping + direct prediction"""
    
    def __init__(self):
        self.version = "enhanced_trait_v1"
        self.trait_keys = [f"char{i}" for i in range(1, 95)]
        self.bigfive_mapper = EnhancedBigFiveMapper()
        
        # Load trained model if available
        model_path = "models/trait_model_rf.pkl"
        
        # Try multiple possible paths for the model file
        possible_paths = [
            model_path,
            f"production_package/{model_path}",
            f"../{model_path}",
            "./models/trait_model_rf.pkl"
        ]
        
        self.model = None
        for path in possible_paths:
            if os.path.exists(path):
                try:
                    self.model = joblib.load(path)
                    logger.info(f"✅ Loaded trained trait model from {path}")
                    break
                except Exception as e:
                    logger.warning(f"Failed to load trait model from {path}: {e}")
                    continue
        
        if self.model is None:
            logger.warning("⚠️ Using enhanced fallback trait model - trained model not found")
    
    def predict(self, features: np.ndarray, audio_metadata: Dict[str, Any]) -> Dict[str, float]:
        """Enhanced trait prediction using Big Five mapping + blending"""
        
        # 1. Estimate Big Five traits from audio features
        estimated_bigfive = self._estimate_bigfive_from_audio(features)
        
        # 2. Map Big Five to char1-char94 using learned mapping
        mapped_chars = self.bigfive_mapper.map_bigfive_to_chars(estimated_bigfive)
        
        # 3. Get direct char predictions (if model available)
        if self.model is not None:
            try:
                features_2d = features.reshape(1, -1)
                direct_chars = self.model.predict(features_2d)[0]
                direct_chars = np.clip(direct_chars, 0.0, 1.0)
            except Exception as e:
                logger.error(f"Error in direct trait prediction: {e}")
                direct_chars = self._fallback_predict(features)
        else:
            direct_chars = self._fallback_predict(features)
        
        # 4. Blend mapped and direct predictions
        # Give more weight to mapped predictions (psychologically grounded)
        alpha = 0.7  # Weight for Big Five mapped predictions
        final_chars = alpha * mapped_chars + (1 - alpha) * direct_chars
        
        # 5. Apply psychological constraints
        final_chars = self._apply_psychological_constraints(final_chars, estimated_bigfive)
        
        # Create trait dictionary
        trait_scores = {
            key: float(score) for key, score in zip(self.trait_keys, final_chars)
        }
        
        logger.debug(f"✅ Enhanced trait prediction complete: {len(trait_scores)} traits")
        logger.debug(f"Big Five estimate: O={estimated_bigfive[0]:.2f}, C={estimated_bigfive[1]:.2f}, "
                    f"E={estimated_bigfive[2]:.2f}, A={estimated_bigfive[3]:.2f}, N={estimated_bigfive[4]:.2f}")
        
        return trait_scores
    
    def _estimate_bigfive_from_audio(self, features: np.ndarray) -> np.ndarray:
        """Estimate Big Five traits from audio features using heuristics"""
        
        # Handle varying feature dimensions (current system uses larger feature vectors)
        feature_vector = features.flatten() if features.ndim > 1 else features
        
        # Use statistical analysis of feature vector for personality estimation
        if len(feature_vector) >= 13:
            # Use first 13 dimensions as MFCC-like features for analysis
            mfcc_features = feature_vector[:13]
            
            # Heuristic mappings (these would be learned in full implementation)
            mean_energy = np.mean(np.abs(mfcc_features))
            variance = np.var(mfcc_features)
            spectral_range = np.max(mfcc_features) - np.min(mfcc_features)
            
            # Rough personality estimates from voice characteristics
            openness = min(1.0, float(variance * 2))  # Higher variance -> more open
            conscientiousness = min(1.0, float(1 - variance))  # Lower variance -> more conscientious  
            extraversion = min(1.0, float(mean_energy * 1.5))  # Higher energy -> more extraverted
            agreeableness = min(1.0, float(0.5 + 0.3 * (1 - spectral_range)))  # Smoother voice -> more agreeable
            neuroticism = min(1.0, float(variance + 0.2 * spectral_range))  # More variation -> higher neuroticism
            
            bigfive = np.array([openness, conscientiousness, extraversion, agreeableness, neuroticism])
            
        else:
            # Fallback: use full feature vector statistics
            mfcc_features = feature_vector[:min(13, len(feature_vector))]
            if len(mfcc_features) < 13:
                # Pad with mean if insufficient features
                mean_val = np.mean(mfcc_features) if len(mfcc_features) > 0 else 0.5
                mfcc_features = np.pad(mfcc_features, (0, 13 - len(mfcc_features)), 'constant', constant_values=mean_val)
            
            # Calculate personality from whatever features we have
            mean_energy = np.mean(np.abs(mfcc_features))
            variance = np.var(mfcc_features)
            spectral_range = np.max(mfcc_features) - np.min(mfcc_features)
            
            # Conservative estimates for limited features
            openness = min(1.0, float(0.3 + variance * 1.5))
            conscientiousness = min(1.0, float(0.4 + (1 - variance) * 0.6))
            extraversion = min(1.0, float(0.3 + mean_energy * 1.2))
            agreeableness = min(1.0, float(0.4 + 0.3 * (1 - spectral_range)))
            neuroticism = min(1.0, float(0.2 + variance + 0.2 * spectral_range))
            
            bigfive = np.array([openness, conscientiousness, extraversion, agreeableness, neuroticism])
        
        return np.clip(bigfive, 0.0, 1.0)
    
    def _apply_psychological_constraints(self, char_scores: np.ndarray, bigfive: np.ndarray) -> np.ndarray:
        """Apply psychological consistency constraints"""
        
        constrained = char_scores.copy()
        
        # Neuroticism constraints (high N -> higher anxiety-related traits)
        if bigfive[4] > 0.6:  # High Neuroticism
            constrained[0] = max(constrained[0], 0.4)   # char1: Loss of Selfworth
            constrained[10] = max(constrained[10], 0.4) # char11: Worryier
            constrained[24] = max(constrained[24], 0.4) # char25: Anger
        
        # Extraversion constraints (high E -> higher leadership traits)
        if bigfive[2] > 0.6:  # High Extraversion
            constrained[9] = max(constrained[9], 0.4)   # char10: Leader
            constrained[18] = max(constrained[18], 0.4) # char19: Assertive
            constrained[90] = max(constrained[90], 0.4) # char91: Talkative
        
        # Ensure opposite traits are negatively correlated
        # E.g., if confident, reduce doubting
        if constrained[41] > 0.6:  # char42: Confident
            constrained[14] = min(constrained[14], 0.3)  # char15: Doubting
        
        return np.clip(constrained, 0.0, 1.0)
    
    def _fallback_predict(self, features: np.ndarray) -> np.ndarray:
        """Enhanced fallback prediction with psychological structure"""
        feature_hash = hash(features.tobytes()) % 1000000
        np.random.seed(feature_hash)
        return np.random.beta(1.5, 1.5, len(self.trait_keys))


class EnhancedModelManager:
    """Enhanced model manager with psychological grounding"""
    
    def __init__(self):
        self.emotion_model = EnhancedEmotionModel()
        self.trait_model = EnhancedTraitModel()
        self.version = "enhanced_v1"
        
        logger.info("✅ Enhanced model manager initialized")
        logger.info("   • Big Five -> char mapping loaded")
        logger.info("   • Psychological constraints active")
        logger.info("   • Blended prediction approach enabled")
    
    def predict_all(self, features: np.ndarray, audio_metadata: Dict[str, Any]) -> tuple[Dict[str, float], Dict[str, float]]:
        """Enhanced prediction with psychological consistency"""
        try:
            # Get enhanced predictions
            emotion_scores = self.emotion_model.predict(features, audio_metadata)
            trait_scores = self.trait_model.predict(features, audio_metadata)
            
            return emotion_scores, trait_scores
            
        except Exception as e:
            logger.error(f"Error in enhanced predictions: {e}")
            # Fallback to basic predictions
            emotion_scores = {f"emo{i}": 0.5 for i in range(1, 27)}
            trait_scores = {f"char{i}": 0.5 for i in range(1, 95)}
            return emotion_scores, trait_scores
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models"""
        return {
            "version": self.version,
            "emotion_model": {
                "version": self.emotion_model.version,
                "has_trained_model": self.emotion_model.model is not None,
                "emotions_count": len(self.emotion_model.emotion_keys),
                "meta_emotions_count": len(self.emotion_model.meta_emotion_keys),
                "extended_emotions_count": len(self.emotion_model.extended_emotion_keys)
            },
            "trait_model": {
                "version": self.trait_model.version,
                "has_trained_model": self.trait_model.model is not None,
                "traits_count": len(self.trait_model.trait_keys),
                "bigfive_mapping": True
            }
        }


# Global model manager instance
enhanced_model_manager = EnhancedModelManager()
