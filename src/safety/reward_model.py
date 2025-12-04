"""
Safety reward model for scoring template safety.
"""
import numpy as np
from typing import List, Optional
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

from .classifier import SafetyFlags
from ..utils.logging import get_logger

logger = get_logger(__name__)


class SafetyRewardModel:
    """
    Safety reward model using gradient boosting.
    
    Combines rule-based features with text embeddings to predict safety scores.
    """
    
    def __init__(self):
        """Initialize safety reward model."""
        self.vectorizer = TfidfVectorizer(
            max_features=100,
            ngram_range=(1, 2),
            strip_accents='unicode',
            lowercase=True
        )
        self.model = HistGradientBoostingClassifier(
            max_iter=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        )
        self.is_trained = False
        
        logger.info("SafetyRewardModel initialized")
    
    def _extract_features(
        self,
        templates: List[str],
        safety_flags_list: Optional[List[SafetyFlags]] = None
    ) -> np.ndarray:
        """
        Extract features from templates.
        
        Args:
            templates: List of template texts
            safety_flags_list: Optional list of safety flags for rule-based features
            
        Returns:
            Feature matrix
        """
        # Text features (TF-IDF)
        if not self.is_trained:
            text_features = self.vectorizer.fit_transform(templates).toarray()
        else:
            text_features = self.vectorizer.transform(templates).toarray()
        
        # If safety flags provided, add rule-based features
        if safety_flags_list:
            rule_features = np.array([
                [
                    float(flags.contains_disallowed_content),
                    float(flags.over_refuses_safe_request),
                    float(flags.meets_refusal_quality_standards),
                    len(flags.flagged_keywords)
                ]
                for flags in safety_flags_list
            ])
            
            features = np.concatenate([text_features, rule_features], axis=1)
        else:
            features = text_features
        
        return features
    
    def train(
        self,
        templates: List[str],
        safety_labels: List[bool],
        safety_flags_list: Optional[List[SafetyFlags]] = None
    ) -> None:
        """
        Train safety reward model.
        
        Args:
            templates: List of template texts
            safety_labels: Binary labels (True = safe, False = unsafe)
            safety_flags_list: Optional safety flags for additional features
        """
        X = self._extract_features(templates, safety_flags_list)
        y = np.array(safety_labels).astype(int)
        
        self.model.fit(X, y)
        self.is_trained = True
        
        logger.info("Safety reward model trained", n_samples=len(templates))
    
    def predict_proba(
        self,
        templates: List[str],
        safety_flags_list: Optional[List[SafetyFlags]] = None
    ) -> np.ndarray:
        """
        Predict safety probabilities.
        
        Args:
            templates: List of template texts
            safety_flags_list: Optional safety flags
            
        Returns:
            Array of probabilities (probability of being safe)
        """
        if not self.is_trained:
            logger.warning("Model not trained, returning default probabilities")
            return np.full(len(templates), 0.8)
        
        X = self._extract_features(templates, safety_flags_list)
        probs = self.model.predict_proba(X)
        
        # Return probability of safe class (class 1)
        return probs[:, 1]
    
    def compute_safety_score(
        self,
        template: str,
        safety_flags: Optional[SafetyFlags] = None
    ) -> float:
        """
        Compute safety score (0-100) for a single template.
        
        Args:
            template: Template text
            safety_flags: Optional safety flags
            
        Returns:
            Safety score between 0 and 100
        """
        flags_list = [safety_flags] if safety_flags else None
        prob = self.predict_proba([template], flags_list)[0]
        
        # Convert probability to 0-100 scale
        score = float(prob * 100)
        
        logger.debug("Safety score computed", template_length=len(template), score=score)
        
        return score
    
    def batch_compute_scores(
        self,
        templates: List[str],
        safety_flags_list: Optional[List[SafetyFlags]] = None
    ) -> List[float]:
        """
        Compute safety scores for multiple templates.
        
        Args:
            templates: List of template texts
            safety_flags_list: Optional list of safety flags
            
        Returns:
            List of safety scores (0-100)
        """
        probs = self.predict_proba(templates, safety_flags_list)
        scores = (probs * 100).tolist()
        
        logger.info(f"Computed safety scores for {len(templates)} templates")
        
        return scores
