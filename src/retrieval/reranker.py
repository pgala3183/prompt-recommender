"""
LightGBM re-ranking model for template scoring.
"""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np
import pandas as pd
try:
    import lightgbm as lgb
except ImportError:
    lgb = None

from .embeddings import TemplateCandidate
from .pricing import calculate_cost
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ScoredTemplate:
    """Template with quality, cost, and safety scores."""
    
    template_id: str
    template_text: str
    domain: Optional[str]
    description: Optional[str]
    predicted_quality: float
    estimated_token_cost: float
    estimated_cost_usd: float
    safety_score: float
    combined_score: float
    model_recommendation: str
    itemized_cost: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'template_id': self.template_id,
            'template_text': self.template_text,
            'domain': self.domain,
            'description': self.description,
            'predicted_quality': self.predicted_quality,
            'estimated_cost_usd': self.estimated_cost_usd,
            'safety_score': self.safety_score,
            'combined_score': self.combined_score,
            'model_recommendation': self.model_recommendation,
            'itemized_cost': self.itemized_cost,
        }


class LightGBMReranker:
    """
    Re-rank candidates using LightGBM with LambdaRank objective.
    
    Scores templates on quality, cost, and safety dimensions.
    """
    
    def __init__(
        self,
        quality_weight: float = 0.5,
        cost_weight: float = 0.3,
        safety_weight: float = 0.2
    ):
        """
        Initialize reranker.
        
        Args:
            quality_weight: Weight for quality score
            cost_weight: Weight for cost score (inverse)
            safety_weight: Weight for safety score
        """
        self.quality_weight = quality_weight
        self.cost_weight = cost_weight
        self.safety_weight = safety_weight
        
        self.quality_model: Optional[lgb.Booster] = None
        self.cost_model: Optional[lgb.Booster] = None
        
        logger.info(
            "LightGBMReranker initialized",
            quality_weight=quality_weight,
            cost_weight=cost_weight,
            safety_weight=safety_weight
        )
    
    def _prepare_features(
        self,
        candidate: TemplateCandidate,
        query: str
    ) -> Dict[str, float]:
        """
        Prepare features for a candidate.
        
        Args:
            candidate: Template candidate
            query: Query text
            
        Returns:
            Feature dictionary
        """
        # Basic features
        features = {
            'similarity_score': candidate.similarity_score,
            'template_length': len(candidate.template_text),
            'query_length': len(query),
            'domain_match': 1.0 if candidate.domain and candidate.domain in query.lower() else 0.0,
        }
        
        # Text overlap features
        query_words = set(query.lower().split())
        template_words = set(candidate.template_text.lower().split())
        
        if query_words:
            overlap = len(query_words & template_words) / len(query_words)
            features['word_overlap'] = overlap
        else:
            features['word_overlap'] = 0.0
        
        # Template structure features
        features['has_examples'] = float('example' in candidate.template_text.lower())
        features['has_constraints'] = float('must' in candidate.template_text.lower() or 
                                           'should' in candidate.template_text.lower())
        
        return features
    
    def train(
        self,
        training_data: pd.DataFrame,
        quality_column: str = 'quality_score',
        cost_column: str = 'total_cost_usd'
    ) -> None:
        """
        Train quality and cost models.
        
        Args:
            training_data: DataFrame with features and targets
            quality_column: Column name for quality scores
            cost_column: Column name for cost values
        """
        if lgb is None:
            raise ImportError("lightgbm not installed")
        
        # Separate features and targets
        feature_cols = [col for col in training_data.columns 
                       if col not in [quality_column, cost_column, 'template_id', 'query']]
        
        X = training_data[feature_cols]
        y_quality = training_data[quality_column]
        y_cost = training_data[cost_column]
        
        # Train quality model
        logger.info("Training quality model...")
        quality_params = {
            'objective': 'regression',
            'metric': 'rmse',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1
        }
        
        quality_train = lgb.Dataset(X, label=y_quality)
        self.quality_model = lgb.train(
            quality_params,
            quality_train,
            num_boost_round=100
        )
        
        # Train cost model
        logger.info("Training cost model...")
        cost_params = quality_params.copy()
        cost_train = lgb.Dataset(X, label=y_cost)
        self.cost_model = lgb.train(
            cost_params,
            cost_train,
            num_boost_round=100
        )
        
        logger.info("Training complete")
    
    def rerank(
        self,
        candidates: List[TemplateCandidate],
        query: str,
        safety_scores: Optional[Dict[str, float]] = None,
        default_safety_score: float = 80.0,
        model_name: str = "gpt-4",
        avg_input_tokens: int = 500,
        avg_output_tokens: int = 300
    ) -> List[ScoredTemplate]:
        """
        Re-rank candidates with quality, cost, and safety scores.
        
        Args:
            candidates: List of template candidates
            query: Query text
            safety_scores: Optional dict mapping template_id to safety score
            default_safety_score: Default safety score if not provided
            model_name: Model to use for cost estimation
            avg_input_tokens: Average input token count for cost estimation
            avg_output_tokens: Average output token count for cost estimation
            
        Returns:
            List of scored templates, sorted by combined score
        """
        if not candidates:
            return []
        
        safety_scores = safety_scores or {}
        scored_templates = []
        
        for candidate in candidates:
            # Prepare features
            features = self._prepare_features(candidate, query)
            feature_values = np.array([list(features.values())])
            
            # Predict quality
            if self.quality_model:
                predicted_quality = float(self.quality_model.predict(feature_values)[0])
                # Clip to [0, 1]
                predicted_quality = max(0.0, min(1.0, predicted_quality))
            else:
                # Fallback: use similarity score
                predicted_quality = float(candidate.similarity_score)
            
            # Estimate cost
            template_tokens = len(candidate.template_text) // 4  # Approximate
            total_input_tokens = avg_input_tokens + template_tokens
            
            if self.cost_model:
                estimated_cost = float(self.cost_model.predict(feature_values)[0])
            else:
                # Fallback: use model pricing
                estimated_cost = calculate_cost(model_name, total_input_tokens, avg_output_tokens)
            
            # Get safety score
            safety_score = safety_scores.get(candidate.template_id, default_safety_score)
            
            # Compute combined score
            # Normalize cost (inverse, so lower cost = higher score)
            max_cost = 1.0  # Assume $1 is max reasonable cost
            cost_score = 1.0 - min(estimated_cost / max_cost, 1.0)
            
            combined_score = (
                self.quality_weight * predicted_quality +
                self.cost_weight * cost_score +
                self.safety_weight * (safety_score / 100.0)
            )
            
            # Create scored template
            scored = ScoredTemplate(
                template_id=candidate.template_id,
                template_text=candidate.template_text,
                domain=candidate.domain,
                description=candidate.description,
                predicted_quality=predicted_quality,
                estimated_token_cost=total_input_tokens + avg_output_tokens,
                estimated_cost_usd=estimated_cost,
                safety_score=safety_score,
                combined_score=combined_score,
                model_recommendation=model_name,
                itemized_cost={
                    'input_tokens': total_input_tokens,
                    'output_tokens': avg_output_tokens,
                    'cost_usd': estimated_cost
                }
            )
            scored_templates.append(scored)
        
        # Sort by combined score (descending)
        scored_templates.sort(key=lambda x: x.combined_score, reverse=True)
        
        logger.info(f"Re-ranked {len(scored_templates)} templates")
        return scored_templates
