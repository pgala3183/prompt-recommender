"""
Policy comparison framework for evaluating candidate policies.
"""
from dataclasses import dataclass
from typing import Callable, Dict, Any, Tuple
import numpy as np
import pandas as pd

from .ips import IPSEstimator
from .doubly_robust import DoublyRobustEstimator
from .bootstrap import BootstrapCI
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PolicyEvaluationResult:
    """Results from evaluating a single policy."""
    
    policy_name: str
    ips_estimate: float
    ips_ci_lower: float
    ips_ci_upper: float
    dr_estimate: float
    dr_ci_lower: float
    dr_ci_upper: float
    effective_sample_size: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'policy_name': self.policy_name,
            'ips_estimate': self.ips_estimate,
            'ips_ci': [self.ips_ci_lower, self.ips_ci_upper],
            'dr_estimate': self.dr_estimate,
            'dr_ci': [self.dr_ci_lower, self.dr_ci_upper],
            'effective_sample_size': self.effective_sample_size,
        }


@dataclass
class ComparisonResult:
    """Results from comparing two policies."""
    
    policy_a_result: PolicyEvaluationResult
    policy_b_result: PolicyEvaluationResult
    ips_difference: float
    ips_diff_ci_lower: float
    ips_diff_ci_upper: float
    dr_difference: float
    dr_diff_ci_lower: float
    dr_diff_ci_upper: float
    is_significant: bool
    p_value: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'policy_a': self.policy_a_result.to_dict(),
            'policy_b': self.policy_b_result.to_dict(),
            'ips_difference': self.ips_difference,
            'ips_diff_ci': [self.ips_diff_ci_lower, self.ips_diff_ci_upper],
            'dr_difference': self.dr_difference,
            'dr_diff_ci': [self.dr_diff_ci_lower, self.dr_diff_ci_upper],
            'is_significant': self.is_significant,
            'p_value': self.p_value,
        }


class PolicyComparator:
    """Framework for comparing and evaluating policies on logged data."""
    
    def __init__(
        self,
        propensity_clip_min: float = 0.01,
        bootstrap_samples: int = 1000,
        alpha: float = 0.05
    ):
        """
        Initialize policy comparator.
        
        Args:
            propensity_clip_min: Minimum propensity value
            bootstrap_samples: Number of bootstrap samples
            alpha: Significance level for confidence intervals
        """
        self.ips_estimator = IPSEstimator(propensity_clip_min)
        self.dr_estimator = DoublyRobustEstimator(propensity_clip_min)
        self.bootstrap = BootstrapCI(n_samples=bootstrap_samples)
        self.alpha = alpha
        
        logger.info("PolicyComparator initialized")
    
    def evaluate_policy(
        self,
        policy_name: str,
        logged_data: pd.DataFrame,
        policy_fn: Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]]
    ) -> PolicyEvaluationResult:
        """
        Evaluate a single policy on logged data.
        
        Args:
            policy_name: Name of the policy
            logged_data: DataFrame with columns: states, actions, rewards, propensities
            policy_fn: Function that takes states and returns (actions, probabilities)
            
        Returns:
            PolicyEvaluationResult
        """
        # Extract data
        states = np.array(logged_data['states'].tolist())
        logged_actions = np.array(logged_data['actions'].tolist())
        rewards = logged_data['rewards'].values
        propensities = logged_data['propensities'].values
        
        # Get policy actions and probabilities
        policy_actions, policy_probs = policy_fn(states)
        
        # Compute IPS estimate with bootstrap CI
        def ips_func(data_indices):
            idx = data_indices.astype(int)
            return self.ips_estimator.compute_ips(
                rewards[idx],
                propensities[idx],
                policy_probs[idx]
            )
        
        ips_est, ips_lower, ips_upper = self.bootstrap.compute_ci(
            np.arange(len(rewards)),
            ips_func,
            alpha=self.alpha
        )
        
        # Train reward model for DR
        self.dr_estimator.train_reward_model(states, logged_actions, rewards)
        
        # Compute DR estimate with bootstrap CI
        def dr_func(data_indices):
            idx = data_indices.astype(int)
            return self.dr_estimator.compute_dr(
                states[idx],
                logged_actions[idx],
                policy_actions[idx],
                rewards[idx],
                propensities[idx],
                policy_probs[idx]
            )
        
        dr_est, dr_lower, dr_upper = self.bootstrap.compute_ci(
            np.arange(len(rewards)),
            dr_func,
            alpha=self.alpha
        )
        
        # Compute effective sample size
        ess = self.ips_estimator.compute_effective_sample_size(
            propensities,
            policy_probs
        )
        
        result = PolicyEvaluationResult(
            policy_name=policy_name,
            ips_estimate=ips_est,
            ips_ci_lower=ips_lower,
            ips_ci_upper=ips_upper,
            dr_estimate=dr_est,
            dr_ci_lower=dr_lower,
            dr_ci_upper=dr_upper,
            effective_sample_size=ess
        )
        
        logger.info(f"Policy {policy_name} evaluated", result=result.to_dict())
        return result
    
    def compare_policies(
        self,
        policy_a_name: str,
        policy_a_fn: Callable,
        policy_b_name: str,
        policy_b_fn: Callable,
        logged_data: pd.DataFrame
    ) -> ComparisonResult:
        """
        Compare two policies on logged data.
        
        Args:
            policy_a_name: Name of first policy
            policy_a_fn: First policy function
            policy_b_name: Name of second policy
            policy_b_fn: Second policy function
            logged_data: Logged data
            
        Returns:
            ComparisonResult with statistical comparison
        """
        # Evaluate both policies
        result_a = self.evaluate_policy(policy_a_name, logged_data, policy_a_fn)
        result_b = self.evaluate_policy(policy_b_name, logged_data, policy_b_fn)
        
        # Compute differences
        ips_diff = result_a.ips_estimate - result_b.ips_estimate
        dr_diff = result_a.dr_estimate - result_b.dr_estimate
        
        # Check significance (if 0 is not in confidence interval of difference)
        ips_diff_lower = result_a.ips_ci_lower - result_b.ips_ci_upper
        ips_diff_upper = result_a.ips_ci_upper - result_b.ips_ci_lower
        
        dr_diff_lower = result_a.dr_ci_lower - result_b.dr_ci_upper
        dr_diff_upper = result_a.dr_ci_upper - result_b.dr_ci_lower
        
        # Conservative: significant if both IPS and DR intervals exclude 0
        ips_significant = ips_diff_lower > 0 or ips_diff_upper < 0
        dr_significant = dr_diff_lower > 0 or dr_diff_upper < 0
        is_significant = ips_significant and dr_significant
        
        # Approximate p-value (conservative)
        p_value = self.alpha if is_significant else 0.5
        
        comparison = ComparisonResult(
            policy_a_result=result_a,
            policy_b_result=result_b,
            ips_difference=ips_diff,
            ips_diff_ci_lower=ips_diff_lower,
            ips_diff_ci_upper=ips_diff_upper,
            dr_difference=dr_diff,
            dr_diff_ci_lower=dr_diff_lower,
            dr_diff_ci_upper=dr_diff_upper,
            is_significant=is_significant,
            p_value=p_value
        )
        
        logger.info("Policy comparison complete", comparison=comparison.to_dict())
        return comparison
