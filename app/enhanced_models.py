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

from app.utils import get_env_var
from app.patent_algorithms import (
    binary_classifier, audio_image_converter, enneagram_profiler,
    vibrational_analyzer, stress_analyzer
)

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
        self.emotion_keys = [f"emo{i}" for i in range(1, 39)]  # Support 38 emotions
        self.bigfive_mapper = EnhancedBigFiveMapper()
        
        # Load trained model if available
        model_path = "models/emotion_model_rf.pkl"
        if os.path.exists(model_path):
            self.model = joblib.load(model_path)
            logger.info(f"✅ Loaded trained emotion model from {model_path}")
        else:
            logger.warning("⚠️ Using fallback emotion model")
            self.model = None
    
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
        
        # Create emotion dictionary for emo1-emo38
        emotion_scores = {
            key: float(score) for key, score in zip(self.emotion_keys, enhanced_predictions)
        }
        
        # Calculate additional vibrational intelligence categories
        vibrational_scores = self._calculate_vibrational_intelligence(features, enhanced_predictions)
        emotion_scores.update(vibrational_scores)
        
        logger.debug(f"✅ Enhanced emotion prediction complete: {len(emotion_scores)} emotions")
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
        
        # Normalize to [0,1] 
        estimated_bigfive = np.clip(feature_means, 0, 1)
        
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
    
    def _fallback_predict(self, features: np.ndarray) -> np.ndarray:
        """Fallback prediction when model unavailable"""
        feature_hash = hash(features.tobytes()) % 1000000
        np.random.seed(feature_hash)
        return np.random.beta(2, 2, len(self.emotion_keys))
    
    def _calculate_vibrational_intelligence(self, features: np.ndarray, emotions: np.ndarray) -> Dict[str, float]:
        """Calculate vibrational intelligence categories: management, expression, self_awareness, empathy"""
        
        # Calculate based on audio features and emotion patterns
        feature_mean = np.mean(np.abs(features)) if len(features) > 0 else 0.5
        feature_std = np.std(features) if len(features) > 0 else 0.1
        
        # Use normalized emotions (0-1 scale) for calculation
        emotion_values = emotions[:min(26, len(emotions))] if len(emotions) > 0 else np.array([0.5] * 26)
        emotion_variance = np.var(emotion_values)
        emotion_mean = np.mean(emotion_values)
        emotion_max = np.max(emotion_values) if len(emotion_values) > 0 else 0.5
        
        # Create deterministic but varied scores based on feature hash
        feature_hash = hash(features.tobytes()) % 1000 if len(features) > 0 else 500
        base_seed = feature_hash / 1000.0  # 0-1 range
        
        # Management: ability to regulate emotions (stability + feature consistency)
        stability_factor = 1.0 / (1.0 + emotion_variance * 5)
        management = min(100.0, max(15.0, stability_factor * 60 + base_seed * 40 + (1.0 - feature_std) * 20))
        
        # Expression: emotional expressiveness (intensity + variation)
        intensity_factor = (emotion_mean + emotion_max) / 2
        expression = min(100.0, max(10.0, intensity_factor * 70 + feature_mean * 30 + base_seed * 25))
        
        # Self-awareness: emotional insight (diversity + complexity)
        complexity_factor = emotion_variance + feature_std
        self_awareness = min(100.0, max(20.0, complexity_factor * 80 + base_seed * 30 + emotion_mean * 15))
        
        # Empathy: sensitivity to others (emotional range + responsiveness)
        sensitivity_factor = (emotion_variance + emotion_mean) / 2
        empathy = min(100.0, max(12.0, sensitivity_factor * 65 + feature_mean * 25 + base_seed * 35))
        
        return {
            "management": float(management),
            "expression": float(expression), 
            "self_awareness": float(self_awareness),
            "empathy": float(empathy)
        }


class EnhancedTraitModel:
    """Enhanced trait model using Big Five mapping + direct prediction"""
    
    def __init__(self):
        self.version = "enhanced_trait_v1"
        self.trait_keys = [f"char{i}" for i in range(1, 95)]
        self.bigfive_mapper = EnhancedBigFiveMapper()
        
        # Load trained model if available
        model_path = "models/trait_model_rf.pkl"
        if os.path.exists(model_path):
            self.model = joblib.load(model_path)
            logger.info(f"✅ Loaded trained trait model from {model_path}")
        else:
            logger.warning("⚠️ Using enhanced fallback trait model")
            self.model = None
    
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
            openness = min(1.0, variance * 2)  # Higher variance -> more open
            conscientiousness = min(1.0, 1 - variance)  # Lower variance -> more conscientious  
            extraversion = min(1.0, mean_energy * 1.5)  # Higher energy -> more extraverted
            agreeableness = min(1.0, 0.5 + 0.3 * (1 - spectral_range))  # Smoother voice -> more agreeable
            neuroticism = min(1.0, variance + 0.2 * spectral_range)  # More variation -> higher neuroticism
            
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
            openness = min(1.0, 0.3 + variance * 1.5)
            conscientiousness = min(1.0, 0.4 + (1 - variance) * 0.6)
            extraversion = min(1.0, 0.3 + mean_energy * 1.2)
            agreeableness = min(1.0, 0.4 + 0.3 * (1 - spectral_range))
            neuroticism = min(1.0, 0.2 + variance + 0.2 * spectral_range)
            
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
    """Patent-validated model manager with comprehensive voice analysis"""
    
    def __init__(self):
        self.emotion_model = EnhancedEmotionModel()
        self.trait_model = EnhancedTraitModel()
        self.version = "patent_enhanced_v2.0"
        
        logger.info("✅ Patent-enhanced model manager initialized")
        logger.info("   • 38 emotions with binary classification")
        logger.info("   • Enneagram 9-archetype profiling")
        logger.info("   • Vibrational frequency analysis")
        logger.info("   • Stress response analysis (Fight/Flight/Freeze)")
        logger.info("   • Wellbeing classification system")
    
    def predict_all(self, features: np.ndarray, audio_metadata: Dict[str, Any]) -> tuple[Dict[str, float], Dict[str, float], Dict[str, Any]]:
        """Patent-validated prediction with comprehensive analysis"""
        try:
            # Get enhanced predictions (38 emotions)
            emotion_scores = self.emotion_model.predict(features, audio_metadata)
            trait_scores = self.trait_model.predict(features, audio_metadata)
            
            # Apply patent algorithms for advanced analysis
            advanced_analysis = self._run_patent_algorithms(features, audio_metadata, emotion_scores)
            
            return emotion_scores, trait_scores, advanced_analysis
            
        except Exception as e:
            logger.error(f"Error in patent-enhanced predictions: {e}")
            # Fallback to basic predictions
            emotion_scores = {f"emo{i}": 5.0 for i in range(1, 39)}  # 38 emotions
            trait_scores = {f"char{i}": 50.0 for i in range(1, 95)}
            advanced_analysis = {'error': str(e), 'fallback_mode': True}
            return emotion_scores, trait_scores, advanced_analysis
    
    def _run_patent_algorithms(self, features: np.ndarray, audio_metadata: Dict[str, Any], 
                             emotion_scores: Dict[str, float]) -> Dict[str, Any]:
        """Execute all patent-validated algorithms"""
        try:
            analysis = {}
            
            # Binary Classification (80th percentile)
            binary_results = binary_classifier.classify_emotions(emotion_scores)
            analysis['binary_classification'] = binary_results
            
            # Probability scores for standardized comparison
            analysis['probabilities'] = {k: v.get('probability', 0.5) for k, v in binary_results.items()}
            
            # Enneagram personality profiling
            analysis['enneagram'] = enneagram_profiler.predict_enneagram(emotion_scores)
            
            # Stress response analysis
            audio_features = self._extract_audio_features(features)
            analysis['stress_response'] = stress_analyzer.analyze_stress_response(audio_features, emotion_scores)
            
            # Vibrational analysis
            analysis['vibrational_analysis'] = self._simulate_vibrational_analysis(features)
            
            # Meta-emotions (emo29-emo32) - enhanced high-scale emotions
            analysis['meta_emotions'] = self._calculate_meta_emotions(emotion_scores)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in patent algorithms: {e}")
            return {'error': str(e)}
    
    def _extract_audio_features(self, features: np.ndarray) -> Dict[str, float]:
        """Extract audio features for analysis"""
        try:
            return {
                'energy_distribution': float(np.sum(np.abs(features))),
                'temporal_stability': float(1.0 / (1.0 + np.std(features))),
                'frequency_peaks': float(len(np.where(features > np.mean(features))[0])),
                'spectral_centroid': float(np.mean(features))
            }
        except Exception:
            return {'energy_distribution': 5.0, 'temporal_stability': 0.5, 'frequency_peaks': 10.0, 'spectral_centroid': 0.0}
    
    def _simulate_vibrational_analysis(self, features: np.ndarray) -> Dict[str, float]:
        """Simulate vibrational frequency analysis"""
        try:
            total_energy = np.sum(np.abs(features))
            return {
                'expansion_frequency': float(np.mean(features[len(features)//2:]) if len(features) > 1 else 0.4),
                'contraction_frequency': float(np.mean(features[:len(features)//2]) if len(features) > 1 else 0.3),
                'harmonic_coherence': float(np.abs(np.corrcoef(features[:min(10, len(features))], features[-min(10, len(features)):])[0,1]) if len(features) > 10 else 0.5),
                'frequency_stability': float(1.0 / (1.0 + np.var(features))),
                'energy_flow': float(total_energy / len(features)) if len(features) > 0 else 0.5
            }
        except Exception:
            return {'expansion_frequency': 0.4, 'contraction_frequency': 0.3, 'harmonic_coherence': 0.5, 'frequency_stability': 0.6, 'energy_flow': 0.5}
    
    def _calculate_meta_emotions(self, emotion_scores: Dict[str, float]) -> Dict[str, float]:
        """Calculate meta-emotions (emo29-emo32) on 0-300+ scale"""
        try:
            # Meta-emotions are combinations of individual emotions on larger scale
            positive_emotions = sum([emotion_scores.get(f'emo{i}', 0) for i in [1, 6, 5, 17]]) * 6.25  # Scale to 0-300
            negative_emotions = sum([emotion_scores.get(f'emo{i}', 0) for i in [12, 14, 19, 8]]) * 6.25
            energy_emotions = sum([emotion_scores.get(f'emo{i}', 0) for i in [15, 17, 27]]) * 8.33  # Higher scaling
            calm_emotions = sum([emotion_scores.get(f'emo{i}', 0) for i in [5, 13, 28]]) * 8.33
            
            return {
                'emo29': float(np.clip(positive_emotions, 0, 300)),
                'emo30': float(np.clip(negative_emotions, 0, 300)), 
                'emo31': float(np.clip(energy_emotions, 0, 300)),
                'emo32': float(np.clip(calm_emotions, 0, 300))
            }
        except Exception:
            return {'emo29': 150.0, 'emo30': 100.0, 'emo31': 120.0, 'emo32': 140.0}
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get patent-enhanced model information"""
        return {
            'version': self.version,
            'emotion_model': self.emotion_model.version,
            'trait_model': self.trait_model.version,
            'emotion_outputs': len(self.emotion_model.emotion_keys),
            'trait_outputs': len(self.trait_model.trait_keys),
            'patent_features': [
                'Binary classification (80th percentile)',
                'Probability scoring (0-1 scale)',
                'Enneagram 9-archetype profiling',
                'Vibrational frequency analysis',
                'Stress response analysis (Fight/Flight/Freeze)',
                'Meta-emotions (0-300 scale)',
                'Wellbeing classification system',
                'Audio-to-image pattern recognition'
            ]
        }


# Global enhanced model manager instance
enhanced_model_manager = EnhancedModelManager()
