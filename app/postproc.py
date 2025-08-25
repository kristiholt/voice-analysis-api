"""
Post-processing and normalization of model outputs.
Handles z-score normalization using rolling baselines and legacy scale mapping.
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
from datetime import datetime, timezone
import logging
import json

from .storage import storage
from .utils import get_env_var

logger = logging.getLogger(__name__)


class PostProcessor:
    """Post-processing and normalization of prediction scores"""
    
    def __init__(self):
        self.normalization_scheme = get_env_var("NORMALIZATION_SCHEME", "zscore_rolling30d")
        self.window_days = int(get_env_var("NORMALIZATION_WINDOW_DAYS", "30"))
        
    async def normalize_scores(self, emotion_scores: Dict[str, float], trait_scores: Dict[str, float]) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, Any]]:
        """
        Apply normalization to emotion and trait scores.
        
        Args:
            emotion_scores: Raw emotion predictions
            trait_scores: Raw trait predictions
            
        Returns:
            Tuple of (normalized_emotions, normalized_traits, normalization_info)
        """
        try:
            # Get normalization baselines
            baselines = await storage.get_normalization_baselines(self.normalization_scheme)
            
            if baselines:
                # Apply z-score normalization
                normalized_emotions = self._apply_zscore_normalization(
                    emotion_scores, baselines.get('emotion_stats', {})
                )
                normalized_traits = self._apply_zscore_normalization(
                    trait_scores, baselines.get('trait_stats', {})
                )
                
                normalization_info = {
                    'scheme': self.normalization_scheme,
                    'window_days': self.window_days,
                    'baseline_date': baselines.get('updated_at', '')[:10] if baselines.get('updated_at') else None
                }
            else:
                # No baselines available - use raw scores
                logger.warning("No normalization baselines found, using raw scores")
                normalized_emotions = emotion_scores.copy()
                normalized_traits = trait_scores.copy()
                
                normalization_info = {
                    'scheme': 'none',
                    'window_days': 0,
                    'baseline_date': None
                }
            
            # Apply legacy scale mapping (0-100)
            final_emotions = self._apply_legacy_scale_mapping(normalized_emotions)
            final_traits = self._apply_legacy_scale_mapping(normalized_traits)
            
            return final_emotions, final_traits, normalization_info
            
        except Exception as e:
            logger.error(f"Error in normalization: {e}")
            # Return original scores on error
            return emotion_scores, trait_scores, {
                'scheme': 'error',
                'window_days': 0,
                'baseline_date': None
            }
    
    def _apply_zscore_normalization(self, scores: Dict[str, float], stats: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Apply z-score normalization using baseline statistics"""
        normalized = {}
        
        for key, value in scores.items():
            if key in stats:
                mean = stats[key].get('mean', 0.5)
                std = stats[key].get('std', 0.1)
                
                # Apply z-score: (value - mean) / std
                if std > 0:
                    z_score = (value - mean) / std
                    # Convert to probability using CDF approximation
                    normalized_value = self._z_score_to_probability(z_score)
                else:
                    normalized_value = value
                
                normalized[key] = float(np.clip(normalized_value, 0.0, 1.0))
            else:
                # No baseline for this key, use raw value
                normalized[key] = value
        
        return normalized
    
    def _z_score_to_probability(self, z_score: float) -> float:
        """Convert z-score to probability using error function approximation"""
        # Approximate normal CDF using error function
        # This maps z-scores to [0, 1] range
        try:
            from scipy.stats import norm
            return float(norm.cdf(z_score))
        except ImportError:
            # Fallback approximation if scipy not available
            return float(0.5 * (1 + np.tanh(z_score / np.sqrt(2))))
    
    def _apply_legacy_scale_mapping(self, scores: Dict[str, float]) -> Dict[str, float]:
        """
        Apply legacy scale mapping to convert [0,1] scores to [0,100] range.
        
        This function can be replaced with actual legacy mapping when available.
        """
        # PLACEHOLDER: Replace with actual legacy mapping function
        # For now, simple linear scaling to 0-100
        legacy_scores = {}
        
        for key, value in scores.items():
            # Ensure value is in [0, 1] range
            clamped_value = max(0.0, min(1.0, value))
            
            # Scale to 0-100 range
            legacy_value = clamped_value * 100.0
            
            legacy_scores[key] = float(np.round(legacy_value, 2))
        
        return legacy_scores
    
    async def compute_normalization_baselines(self, days: Optional[int] = None) -> Dict[str, Any]:
        """
        Compute normalization baselines from recent results.
        Used by the nightly normalization job.
        
        Args:
            days: Number of days to look back (default uses configured window)
            
        Returns:
            Baseline statistics for emotions and traits
        """
        if days is None:
            days = self.window_days
        
        try:
            # Get recent results
            recent_results = await storage.get_recent_results_for_normalization(days)
            
            if len(recent_results) < 10:  # Minimum samples needed
                logger.warning(f"Insufficient data for baseline computation: {len(recent_results)} samples")
                return {}
            
            # Aggregate scores by type
            emotion_data = {}
            trait_data = {}
            
            for result in recent_results:
                scores = result.get('scores', {})
                if isinstance(scores, str):
                    scores = json.loads(scores)
                
                # Collect emotion scores
                emotions = scores.get('emotions', {})
                for key, value in emotions.items():
                    if key not in emotion_data:
                        emotion_data[key] = []
                    emotion_data[key].append(float(value))
                
                # Collect trait scores
                traits = scores.get('traits', {})
                for key, value in traits.items():
                    if key not in trait_data:
                        trait_data[key] = []
                    trait_data[key].append(float(value))
            
            # Compute statistics
            emotion_stats = self._compute_statistics(emotion_data)
            trait_stats = self._compute_statistics(trait_data)
            
            baseline_data = {
                'scheme': self.normalization_scheme,
                'window_days': days,
                'sample_count': len(recent_results),
                'emotion_stats': emotion_stats,
                'trait_stats': trait_stats,
                'computed_at': datetime.now(timezone.utc).isoformat(),
                'updated_at': datetime.now(timezone.utc).isoformat()
            }
            
            logger.info(f"Computed baselines from {len(recent_results)} samples over {days} days")
            return baseline_data
            
        except Exception as e:
            logger.error(f"Error computing normalization baselines: {e}")
            return {}
    
    def _compute_statistics(self, data: Dict[str, list]) -> Dict[str, Dict[str, float]]:
        """Compute mean and standard deviation for each key"""
        stats = {}
        
        for key, values in data.items():
            if len(values) > 0:
                values_array = np.array(values)
                stats[key] = {
                    'mean': float(np.mean(values_array)),
                    'std': float(np.std(values_array)),
                    'min': float(np.min(values_array)),
                    'max': float(np.max(values_array)),
                    'count': len(values)
                }
            else:
                stats[key] = {
                    'mean': 0.5,
                    'std': 0.1,
                    'min': 0.0,
                    'max': 1.0,
                    'count': 0
                }
        
        return stats


# Global post-processor instance
post_processor = PostProcessor()
