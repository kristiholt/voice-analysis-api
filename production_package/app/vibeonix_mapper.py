"""
Wellness Score Mapper

Converts complex emotion and trait analysis (emo1-emo26 + char1-char94) 
into simple wellness indicators.
"""

import logging
from typing import Dict, Any
import numpy as np

logger = logging.getLogger(__name__)

class WellnessMapper:
    """Maps complex voice analysis results to simple wellness scores"""
    
    def __init__(self):
        self.version = "vibeonix_mapper_v1.0"
        
        # Emotion mappings from Vibeonix specification (corrected)
        self.emotion_mappings = {
            # Core emotions (emo1-emo28)
            "happiness": ["emo1"],
            "pondering": ["emo2"],  # Feeling stuck
            "involvement": ["emo3"],  # Engaged
            "anger": ["emo4"],
            "relax": ["emo5"],  # Peace/Calm
            "love": ["emo6"],
            "positive_stress": ["emo7"],
            "down": ["emo8"],  # Discouraged
            "sadness": ["emo9"],
            "fear": ["emo10"],  # FIXED: was incorrectly emo19
            "loneliness": ["emo11"],
            "attention": ["emo12"],  # Present
            "doubt": ["emo13"],  # FIXED: was incorrectly emo20
            "overwhelmed": ["emo14"],  # FIXED: was incorrectly emo21
            "alertness": ["emo15"],  # self_awareness pillar
            "longing": ["emo16"],
            "confidence": ["emo17"],
            "confused": ["emo18"],
            "frustration": ["emo19"],  # Was incorrectly mapped to fear
            "melancholy": ["emo20"],  # Was incorrectly mapped to doubt
            "envy": ["emo21"],  # Was incorrectly mapped to overwhelmed
            "bored": ["emo22"],
            "shy": ["emo23"],
            "despair": ["emo24"],
            "exhausted": ["emo25"],
            "drive": ["emo26"],  # Motivation
            "coolness": ["emo27"],  # Adaptive - NEW
            "jealousy": ["emo28"],  # NEW
            
            # Pillar/overall emotions
            "empathy_overall": ["emo29"],  # positive_overall
            "negative_overall": ["emo30"],  # NEW
            "self_expression_overall": ["emo31"],  # valence_overall
            "self_management_overall": ["emo32"],  # control_overall
            
            # Fear-spectrum emotions (emo33-emo38) - ALL NEW
            "negative_stress": ["emo33"],
            "vigilance": ["emo34"],  # Fear of losing identity
            "fear_abandonment": ["emo35"],  # Fear of separation
            "fear_autonomy_loss": ["emo36"],  # Fear of loss of autonomy
            "fear_failure": ["emo37"],  # Fear of failure
            "fear_physical_harm": ["emo38"],  # Fear of physical/mental harm
        }
        
        # Trait mappings from label_map.yaml  
        self.trait_mappings = {
            "anger_trait": ["char25"],
            "calm_trait": ["char34"], 
            "confidence_trait": ["char42"],
            "loneliness_trait": ["char47"],
            "drive_trait": ["char77"],
            "worry_trait": ["char11"],
            "doubting_trait": ["char15"],
            "restless_trait": ["char24"],
            "leadership_trait": ["char10"],
            "resilience_traits": ["char26", "char37", "char85"], # Hard worker, skilled, picking up pieces
        }
    
    def map_to_wellness_scores(self, emotion_scores: Dict[str, float], 
                             trait_scores: Dict[str, float]) -> Dict[str, int]:
        """
        Convert complex analysis to wellness scores (1-100 scale)
        
        Args:
            emotion_scores: emo1-emo26 scores (0.0-1.0)
            trait_scores: char1-char94 scores (0.0-1.0)
            
        Returns:
            Dict with 7 wellness scores (1-100 scale)
        """
        
        try:
            # Extract relevant scores with safety checks
            emotions = self._safe_extract_scores(emotion_scores, self.emotion_mappings)
            traits = self._safe_extract_scores(trait_scores, self.trait_mappings)
            
            # Calculate wellness scores (1-100 scale)
            wellness_scores = {
                "mood_score": self._calculate_mood_score(emotions, traits),
                "anxiety_score": self._calculate_anxiety_score(emotions, traits),
                "stress_score": self._calculate_stress_score(emotions, traits),
                "happiness_score": self._calculate_happiness_score(emotions, traits),
                "loneliness_score": self._calculate_loneliness_score(emotions, traits),
                "resilience_score": self._calculate_resilience_score(emotions, traits),
                "energy_score": self._calculate_energy_score(emotions, traits),
            }
            
            # Ensure all scores are integers in 1-100 range
            wellness_scores = {k: max(1, min(100, int(v))) for k, v in wellness_scores.items()}
            
            logger.info(f"✅ Wellness scores calculated: {wellness_scores}")
            return wellness_scores
            
        except Exception as e:
            logger.error(f"Error in wellness score mapping: {e}")
            # Return neutral scores as fallback
            return {
                "mood_score": 50,
                "anxiety_score": 50,
                "stress_score": 50,
                "happiness_score": 50,
                "loneliness_score": 50,
                "resilience_score": 50,
                "energy_score": 50,
            }
    
    def _safe_extract_scores(self, scores: Dict[str, float], 
                           mappings: Dict[str, list]) -> Dict[str, float]:
        """Safely extract and average scores from mappings"""
        extracted = {}
        
        for key, score_keys in mappings.items():
            values = []
            for score_key in score_keys:
                if score_key in scores and scores[score_key] is not None:
                    raw_value = float(scores[score_key])
                    # Normalize to 0-1 range regardless of input scale
                    if raw_value > 1.0:
                        # Assume 0-100 scale, convert to 0-1
                        normalized_value = raw_value / 100.0
                    else:
                        # Already in 0-1 scale
                        normalized_value = raw_value
                    
                    # Ensure value is within bounds
                    normalized_value = max(0.0, min(1.0, normalized_value))
                    values.append(normalized_value)
            
            if values:
                extracted[key] = np.mean(values)
            else:
                extracted[key] = 0.5  # Neutral fallback
                
        return extracted
    
    def _calculate_mood_score(self, emotions: Dict, traits: Dict) -> float:
        """Calculate mood score (1-100, higher = better mood)"""
        # Primary: happiness vs sadness/despair balance
        happiness = emotions.get("happiness", 0.5)
        sadness = emotions.get("sadness", 0.5)
        despair = emotions.get("despair", 0.5)  # emo24 (new)
        
        # Secondary: relax vs anger balance (updated name)
        relax = emotions.get("relax", 0.5)  # emo5 (peace/calm)
        anger = emotions.get("anger", 0.5)
        
        # Positive emotions
        love = emotions.get("love", 0.5)  # emo6 (new)
        involvement = emotions.get("involvement", 0.5)  # emo3 engaged (new)
        
        # Weighted average with expanded emotion range
        mood_balance = (
            happiness * 0.3 - sadness * 0.25 - despair * 0.15 +
            relax * 0.1 - anger * 0.1 +
            love * 0.05 + involvement * 0.05
        )
        
        # Convert to 1-100 scale (neutral = 50)
        return 50 + (mood_balance * 50)
    
    def _calculate_anxiety_score(self, emotions: Dict, traits: Dict) -> float:
        """Calculate anxiety score (1-100, higher = more anxious)"""
        # Primary: fear, doubt, overwhelmed (now correctly mapped)
        fear = emotions.get("fear", 0.5)
        doubt = emotions.get("doubt", 0.5)
        overwhelmed = emotions.get("overwhelmed", 0.5)
        
        # Fear-spectrum emotions (new additions)
        fear_abandonment = emotions.get("fear_abandonment", 0.5)
        fear_failure = emotions.get("fear_failure", 0.5)
        vigilance = emotions.get("vigilance", 0.5)
        
        # Inverse of calm (less calm = more anxious)
        relax_inverse = 1.0 - emotions.get("relax", 0.5)
        
        # Trait indicators
        worry_trait = traits.get("worry_trait", 0.5)
        restless_trait = traits.get("restless_trait", 0.5)
        
        # Weighted average with expanded fear spectrum
        anxiety_level = (
            fear * 0.2 + 
            doubt * 0.15 + 
            overwhelmed * 0.15 + 
            fear_abandonment * 0.1 +
            fear_failure * 0.1 +
            vigilance * 0.1 +
            relax_inverse * 0.1 + 
            worry_trait * 0.05 + 
            restless_trait * 0.05
        )
        
        # Convert to 1-100 scale
        return max(1, min(100, anxiety_level * 100))
    
    def _calculate_stress_score(self, emotions: Dict, traits: Dict) -> float:
        """Calculate stress score (1-100, higher = more stressed)"""
        # Primary: positive and negative stress
        positive_stress = emotions.get("positive_stress", 0.5)  # emo7
        negative_stress = emotions.get("negative_stress", 0.5)  # emo33 (new)
        down = emotions.get("down", 0.5)  # emo8 (discouraged)
        
        # Secondary: overwhelmed, exhausted (new)
        overwhelmed = emotions.get("overwhelmed", 0.5)
        exhausted = emotions.get("exhausted", 0.5)  # emo25
        restless_trait = traits.get("restless_trait", 0.5)
        
        # Weighted average with expanded stress indicators
        stress_level = (
            positive_stress * 0.2 +
            negative_stress * 0.3 +
            down * 0.15 +
            overwhelmed * 0.15 + 
            exhausted * 0.1 +
            restless_trait * 0.1
        )
        
        # Convert to 1-100 scale  
        return max(1, min(100, stress_level * 100))
    
    def _calculate_happiness_score(self, emotions: Dict, traits: Dict) -> float:
        """Calculate happiness score (1-100, higher = happier)"""
        # Primary: direct happiness emotion and love
        happiness = emotions.get("happiness", 0.5)
        love = emotions.get("love", 0.5)  # emo6 (new)
        
        # Secondary: relax, confidence, involvement
        relax = emotions.get("relax", 0.5)  # emo5 (peace/calm)
        confidence = emotions.get("confidence", 0.5)
        involvement = emotions.get("involvement", 0.5)  # emo3 engaged (new)
        
        # Inverse of negative emotions
        sadness_inverse = 1.0 - emotions.get("sadness", 0.5)
        despair_inverse = 1.0 - emotions.get("despair", 0.5)  # emo24 (new)
        melancholy_inverse = 1.0 - emotions.get("melancholy", 0.5)  # emo20 (new)
        
        # Weighted average with expanded emotion range
        happiness_level = (
            happiness * 0.35 + 
            love * 0.15 +
            relax * 0.15 + 
            confidence * 0.1 + 
            involvement * 0.05 +
            sadness_inverse * 0.1 +
            despair_inverse * 0.05 +
            melancholy_inverse * 0.05
        )
        
        # Convert to 1-100 scale
        return max(1, min(100, happiness_level * 100))
    
    def _calculate_loneliness_score(self, emotions: Dict, traits: Dict) -> float:
        """Calculate loneliness score (1-100, higher = more lonely)"""
        # Primary: direct loneliness emotion and trait
        loneliness_emo = emotions.get("loneliness", 0.5)
        loneliness_trait = traits.get("loneliness_trait", 0.5)
        
        # Secondary: inverse of social confidence/leadership
        leadership_inverse = 1.0 - traits.get("leadership_trait", 0.5)
        
        # Weighted average
        loneliness_level = (
            loneliness_emo * 0.5 + 
            loneliness_trait * 0.35 + 
            leadership_inverse * 0.15
        )
        
        # Convert to 1-100 scale
        return max(1, min(100, loneliness_level * 100))
    
    def _calculate_resilience_score(self, emotions: Dict, traits: Dict) -> float:
        """Calculate resilience score (1-100, higher = more resilient)"""
        # Primary: confidence, drive
        confidence = emotions.get("confidence", 0.5)
        drive = emotions.get("drive", 0.5)
        
        # Secondary: resilience traits (hard worker, skilled, picking up pieces)
        resilience_traits_avg = np.mean([
            traits.get("resilience_traits", 0.5)  # Average of char26, char37, char85
        ])
        
        # Inverse of doubt, fear
        doubt_inverse = 1.0 - emotions.get("doubt", 0.5)
        fear_inverse = 1.0 - emotions.get("fear", 0.5)
        
        # Weighted average
        resilience_level = (
            confidence * 0.25 + 
            drive * 0.25 + 
            resilience_traits_avg * 0.2 + 
            doubt_inverse * 0.15 + 
            fear_inverse * 0.15
        )
        
        # Convert to 1-100 scale
        return max(1, min(100, resilience_level * 100))
    
    def _calculate_energy_score(self, emotions: Dict, traits: Dict) -> float:
        """Calculate energy score (1-100, higher = more energetic)"""
        # Primary: drive emotion and trait
        drive_emo = emotions.get("drive", 0.5)
        drive_trait = traits.get("drive_trait", 0.5)
        
        # Secondary: confidence, leadership
        confidence = emotions.get("confidence", 0.5)
        leadership = traits.get("leadership_trait", 0.5)
        
        # Inverse of overwhelmed, sadness
        overwhelmed_inverse = 1.0 - emotions.get("overwhelmed", 0.5)
        sadness_inverse = 1.0 - emotions.get("sadness", 0.5)
        
        # Weighted average
        energy_level = (
            drive_emo * 0.3 + 
            drive_trait * 0.25 + 
            confidence * 0.15 + 
            leadership * 0.1 + 
            overwhelmed_inverse * 0.1 + 
            sadness_inverse * 0.1
        )
        
        # Convert to 1-100 scale
        return max(1, min(100, energy_level * 100))


# Global mapper instance
wellness_mapper = WellnessMapper()