"""
Doubly Robust (DR) estimator for off-policy evaluation.
"""
import numpy as np
import pandas as pd
from typing import Callable, Optional
from sklearn.ensemble import GradientBoostingRegressor
from ..utils.logging import get_logger

logger = get_logger(__name__)


class RewardModel:
    """Learned reward model for DR estimation."""
    
    def __init__(self):
        """Initialize reward model."""
        self.model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        )
        self.is_trained = False
    
    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train reward model.
        
        Args:
            X: Feature matrix (states and actions)
            y: Observed rewards
        """
        self.model.fit(X, y)
        self.is_trained = True
        logger.info("Reward model trained", n_samples=len(X))
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict rewards.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predicted rewards
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        return self.model.predict(X)


class DoublyRobustEstimator:
    """
    Doubly Robust estimator combining IPS with learned reward model.
    
    Formula:
    DR = Σ[q̂(x,π(x)) + (π(a|x)/π₀(a|x))(r - q̂(x,a))]
    
    Where q̂ is the learned reward model.
    """
    
    def __init__(self, propensity_clip_min: float = 0.01):
        """
        Initialize DR estimator.
        
        Args:
            propensity_clip_min: Minimum propensity value
        """
        self.propensity_clip_min = propensity_clip_min
        self.reward_model = RewardModel()
        logger.info("DoublyRobustEstimator initialized", clip_min=propensity_clip_min)
    
    def train_reward_model(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray
    ) -> None:
        """
        Train the reward model on logged data.
        
        Args:
            states: State features
            actions: Action features
            rewards: Observed rewards
        """
        # Combine states and actions
        X = np.concatenate([states, actions], axis=1)
        self.reward_model.train(X, rewards)
    
    def compute_dr(
        self,
        states: np.ndarray,
        logged_actions: np.ndarray,
        policy_actions: np.ndarray,
        rewards: np.ndarray,
        propensities: np.ndarray,
        policy_probs: np.ndarray
    ) -> float:
        """
        Compute Doubly Robust estimate.
        
        Args:
            states: State features for each sample
            logged_actions: Actions taken by logging policy
            policy_actions: Actions recommended by new policy
            rewards: Observed rewards
            propensities: Logging policy probabilities
            policy_probs: New policy probabilities
            
        Returns:
            DR estimate of expected reward
        """
        if not self.reward_model.is_trained:
            raise ValueError("Reward model must be trained first")
        
        n = len(rewards)
        
        # Clip propensities
        clipped_propensities = np.clip(
            propensities,
            self.propensity_clip_min,
            1.0
        )
        
        # Predict rewards under policy actions
        policy_action_features = np.concatenate([states, policy_actions], axis=1)
        q_policy = self.reward_model.predict(policy_action_features)
        
        # Predict rewards under logged actions
        logged_action_features = np.concatenate([states, logged_actions], axis=1)
        q_logged = self.reward_model.predict(logged_action_features)
        
        # Compute importance weights
        importance_weights = policy_probs / clipped_propensities
        
        # DR formula
        dr_values = q_policy + importance_weights * (rewards - q_logged)
        dr_estimate = np.mean(dr_values)
        
        logger.debug(
            "DR computed",
            estimate=dr_estimate,
            num_samples=n,
            avg_weight=np.mean(importance_weights)
        )
        
        return float(dr_estimate)
    
    def compute_dr_variance(
        self,
        states: np.ndarray,
        logged_actions: np.ndarray,
        policy_actions: np.ndarray,
        rewards: np.ndarray,
        propensities: np.ndarray,
        policy_probs: np.ndarray,
        dr_estimate: Optional[float] = None
    ) -> float:
        """
        Compute variance of DR estimate.
        
        Args:
            states: State features
            logged_actions: Logged actions
            policy_actions: Policy actions
            rewards: Observed rewards
            propensities: Logging policy probabilities
            policy_probs: New policy probabilities
            dr_estimate: Pre-computed DR estimate (optional)
            
        Returns:
            Estimated variance
        """
        if dr_estimate is None:
            dr_estimate = self.compute_dr(
                states,
                logged_actions,
                policy_actions,
                rewards,
                propensities,
                policy_probs
            )
        
        # Clip propensities
        clipped_propensities = np.clip(
            propensities,
            self.propensity_clip_min,
            1.0
        )
        
        # Predict rewards
        policy_action_features = np.concatenate([states, policy_actions], axis=1)
        q_policy = self.reward_model.predict(policy_action_features)
        
        logged_action_features = np.concatenate([states, logged_actions], axis=1)
        q_logged = self.reward_model.predict(logged_action_features)
        
        # Compute importance weights
        importance_weights = policy_probs / clipped_propensities
        
        # DR values
        dr_values = q_policy + importance_weights * (rewards - q_logged)
        
        # Variance
        variance = np.var(dr_values) / len(dr_values)
        
        return float(variance)
