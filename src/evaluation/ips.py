"""
Inverse Propensity Scoring (IPS) estimator for off-policy evaluation.
"""
import numpy as np
from typing import Optional
from ..utils.logging import get_logger

logger = get_logger(__name__)


class IPSEstimator:
    """
    Inverse Propensity Scoring with self-normalization.
    
    Implements the formula:
    IPS = Σ(r_i × π(a_i|x_i) / π₀(a_i|x_i)) / Σ(π(a_i|x_i) / π₀(a_i|x_i))
    
    Where:
    - r_i: reward for action i
    - π(a_i|x_i): new policy probability
    - π₀(a_i|x_i): logging policy probability (propensity)
    """
    
    def __init__(self, propensity_clip_min: float = 0.01):
        """
        Initialize IPS estimator.
        
        Args:
            propensity_clip_min: Minimum propensity value to avoid extreme weights
        """
        self.propensity_clip_min = propensity_clip_min
        logger.info("IPSEstimator initialized", clip_min=propensity_clip_min)
    
    def compute_ips(
        self,
        rewards: np.ndarray,
        propensities: np.ndarray,
        policy_probs: np.ndarray
    ) -> float:
        """
        Compute self-normalized IPS estimate.
        
        Args:
            rewards: Array of observed rewards
            propensities: Array of logging policy probabilities π₀(a|x)
            policy_probs: Array of new policy probabilities π(a|x)
            
        Returns:
            IPS estimate of expected reward
        """
        # Clip propensities to avoid extreme weights
        clipped_propensities = np.clip(
            propensities,
            self.propensity_clip_min,
            1.0
        )
        
        # Compute importance weights
        importance_weights = policy_probs / clipped_propensities
        
        # Self-normalized IPS
        numerator = np.sum(rewards * importance_weights)
        denominator = np.sum(importance_weights)
        
        if denominator == 0:
            logger.warning("Zero denominator in IPS, returning 0")
            return 0.0
        
        ips_estimate = numerator / denominator
        
        logger.debug(
            "IPS computed",
            estimate=ips_estimate,
            num_samples=len(rewards),
            avg_weight=np.mean(importance_weights)
        )
        
        return float(ips_estimate)
    
    def compute_ips_variance(
        self,
        rewards: np.ndarray,
        propensities: np.ndarray,
        policy_probs: np.ndarray,
        ips_estimate: Optional[float] = None
    ) -> float:
        """
        Compute variance of IPS estimate.
        
        Args:
            rewards: Array of observed rewards
            propensities: Array of logging policy probabilities
            policy_probs: Array of new policy probabilities
            ips_estimate: Pre-computed IPS estimate (optional)
            
        Returns:
            Estimated variance
        """
        if ips_estimate is None:
            ips_estimate = self.compute_ips(rewards, propensities, policy_probs)
        
        # Clip propensities
        clipped_propensities = np.clip(
            propensities,
            self.propensity_clip_min,
            1.0
        )
        
        # Compute importance weights
        importance_weights = policy_probs / clipped_propensities
        
        # Compute weighted rewards
        weighted_rewards = rewards * importance_weights
        
        # Self-normalized variance formula
        n = len(rewards)
        variance = np.sum(
            (weighted_rewards - ips_estimate) ** 2 * importance_weights
        ) / (n * np.sum(importance_weights))
        
        return float(variance)
    
    def compute_effective_sample_size(
        self,
        propensities: np.ndarray,
        policy_probs: np.ndarray
    ) -> float:
        """
        Compute effective sample size to assess weight concentration.
        
        Args:
            propensities: Array of logging policy probabilities
            policy_probs: Array of new policy probabilities
            
        Returns:
            Effective sample size
        """
        clipped_propensities = np.clip(
            propensities,
            self.propensity_clip_min,
            1.0
        )
        
        importance_weights = policy_probs / clipped_propensities
        
        # ESS = (Σw_i)² / Σ(w_i²)
        sum_weights = np.sum(importance_weights)
        sum_squared_weights = np.sum(importance_weights ** 2)
        
        if sum_squared_weights == 0:
            return 0.0
        
        ess = (sum_weights ** 2) / sum_squared_weights
        
        logger.debug("Effective sample size", ess=ess, actual_n=len(propensities))
        
        return float(ess)
