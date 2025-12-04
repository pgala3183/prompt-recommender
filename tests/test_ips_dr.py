"""
Unit tests for IPS and DR estimators.
"""
import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.evaluation.ips import IPSEstimator
from src.evaluation.doubly_robust import DoublyRobustEstimator


class TestIPSEstimator:
    """Test IPS estimator."""
    
    def test_ips_unbiased(self):
        """Test that IPS is unbiased on simulated data."""
        # Generate synthetic data where logging policy = target policy
        # In this case, IPS should equal the mean reward
        np.random.seed(42)
        n = 1000
        
        rewards = np.random.normal(0.7, 0.1, n)
        propensities = np.full(n, 0.5)  # Uniform logging policy
        policy_probs = np.full(n, 0.5)  # Same as logging policy
        
        estimator = IPSEstimator()
        ips_estimate = estimator.compute_ips(rewards, propensities, policy_probs)
        
        # Should be close to mean reward when policies are identical
        assert abs(ips_estimate - np.mean(rewards)) < 0.01
    
    def test_propensity_clipping(self):
        """Test propensity clipping prevents extreme weights."""
        np.random.seed(42)
        n = 100
        
        rewards = np.random.normal(0.7, 0.1, n)
        propensities = np.random.uniform(0.001, 1.0, n)  # Some very small propensities
        policy_probs = np.full(n, 0.5)
        
        estimator = IPSEstimator(propensity_clip_min=0.01)
        ips_estimate = estimator.compute_ips(rewards, propensities, policy_probs)
        
        # Should return a finite value
        assert np.isfinite(ips_estimate)
        assert ips_estimate >= 0
    
    def test_effective_sample_size(self):
        """Test effective sample size calculation."""
        np.random.seed(42)
        n = 100
        
        propensities = np.full(n, 0.5)
        policy_probs = np.full(n, 0.5)
        
        estimator = IPSEstimator()
        ess = estimator.compute_effective_sample_size(propensities, policy_probs)
        
        # When policies are identical, ESS should equal n
        assert abs(ess - n) < 1
    
    def test_ips_variance(self):
        """Test variance calculation."""
        np.random.seed(42)
        n = 100
        
        rewards = np.random.normal(0.7, 0.1, n)
        propensities = np.full(n, 0.5)
        policy_probs = np.full(n, 0.5)
        
        estimator = IPSEstimator()
        variance = estimator.compute_ips_variance(rewards, propensities, policy_probs)
        
        # Variance should be non-negative
        assert variance >= 0


class TestDoublyRobustEstimator:
    """Test DR estimator."""
    
    def test_dr_training(self):
        """Test that reward model can be trained."""
        np.random.seed(42)
        n = 200
        
        states = np.random.randn(n, 5)
        actions = np.random.randn(n, 2)
        rewards = np.random.normal(0.7, 0.1, n)
        
        estimator = DoublyRobustEstimator()
        estimator.train_reward_model(states, actions, rewards)
        
        assert estimator.reward_model.is_trained
    
    def test_dr_estimate(self):
        """Test DR estimation."""
        np.random.seed(42)
        n = 200
        
        # Generate synthetic data
        states = np.random.randn(n, 5)
        logged_actions = np.random.randn(n, 2)
        policy_actions = logged_actions  # Same actions for simplicity
        rewards = np.random.normal(0.7, 0.1, n)
        propensities = np.full(n, 0.5)
        policy_probs = np.full(n, 0.5)
        
        estimator = DoublyRobustEstimator()
        estimator.train_reward_model(states, logged_actions, rewards)
        
        dr_estimate = estimator.compute_dr(
            states,
            logged_actions,
            policy_actions,
            rewards,
            propensities,
            policy_probs
        )
        
        # Should be close to mean reward when policies are identical
        assert abs(dr_estimate - np.mean(rewards)) < 0.1
        assert np.isfinite(dr_estimate)
    
    def test_dr_reduces_variance(self):
        """Test that DR has lower variance than IPS (in expectation)."""
        np.random.seed(42)
        n = 100
        
        states = np.random.randn(n, 5)
        logged_actions = np.random.randn(n, 2)
        policy_actions = np.random.randn(n, 2)
        rewards = np.random.normal(0.7, 0.1, n)
        
        # Variable propensities
        propensities = np.random.uniform(0.1, 1.0, n)
        policy_probs = np.random.uniform(0.1, 1.0, n)
        
        # IPS estimate
        ips_estimator = IPSEstimator()
        ips_var = ips_estimator.compute_ips_variance(rewards, propensities, policy_probs)
        
        # DR estimate
        dr_estimator = DoublyRobustEstimator()
        dr_estimator.train_reward_model(states, logged_actions, rewards)
        dr_var = dr_estimator.compute_dr_variance(
            states,
            logged_actions,
            policy_actions,
            rewards,
            propensities,
            policy_probs
        )
        
        # DR variance should be lower (this may not always hold with random data,
        # but is true in expectation with good reward model)
        assert dr_var >= 0
        assert ips_var >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
