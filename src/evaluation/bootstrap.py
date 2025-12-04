"""
Bootstrap confidence interval calculator.
"""
import numpy as np
from typing import Callable, Tuple
from ..utils.logging import get_logger

logger = get_logger(__name__)


class BootstrapCI:
    """Bootstrap confidence interval calculator."""
    
    def __init__(self, n_samples: int = 1000, random_state: int = 42):
        """
        Initialize bootstrap calculator.
        
        Args:
            n_samples: Number of bootstrap samples
            random_state: Random seed for reproducibility
        """
        self.n_samples = n_samples
        self.random_state = random_state
        np.random.seed(random_state)
        logger.info("BootstrapCI initialized", n_samples=n_samples)
    
    def compute_ci(
        self,
        data: np.ndarray,
        estimator: Callable[[np.ndarray], float],
        alpha: float = 0.05
    ) -> Tuple[float, float, float]:
        """
        Compute bootstrap confidence interval using percentile method.
        
        Args:
            data: Input data for bootstrapping
            estimator: Function that computes estimate from data
            alpha: Significance level (default 0.05 for 95% CI)
            
        Returns:
            Tuple of (point_estimate, lower_bound, upper_bound)
        """
        n = len(data)
        bootstrap_estimates = []
        
        logger.info(f"Computing bootstrap CI with {self.n_samples} samples...")
        
        for i in range(self.n_samples):
            # Resample with replacement
            indices = np.random.choice(n, size=n, replace=True)
            bootstrap_sample = data[indices]
            
            # Compute estimate
            estimate = estimator(bootstrap_sample)
            bootstrap_estimates.append(estimate)
        
        bootstrap_estimates = np.array(bootstrap_estimates)
        
        # Compute point estimate on original data
        point_estimate = estimator(data)
        
        # Compute percentile confidence bounds
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        lower_bound = np.percentile(bootstrap_estimates, lower_percentile)
        upper_bound = np.percentile(bootstrap_estimates, upper_percentile)
        
        logger.info(
            "Bootstrap CI computed",
            point_estimate=point_estimate,
            ci=f"[{lower_bound:.4f}, {upper_bound:.4f}]",
            alpha=alpha
        )
        
        return float(point_estimate), float(lower_bound), float(upper_bound)
    
    def compute_ci_advanced(
        self,
        data: np.ndarray,
        estimator: Callable[[np.ndarray], float],
        alpha: float = 0.05,
        method: str = 'percentile'
    ) -> Tuple[float, float, float]:
        """
        Compute bootstrap CI with different methods.
        
        Args:
            data: Input data
            estimator: Estimator function
            alpha: Significance level
            method: 'percentile' or 'bca' (bias-corrected and accelerated)
            
        Returns:
            Tuple of (point_estimate, lower_bound, upper_bound)
        """
        if method == 'percentile':
            return self.compute_ci(data, estimator, alpha)
        elif method == 'bca':
            # BCA method (more complex, not implemented for brevity)
            logger.warning("BCA method not implemented, using percentile")
            return self.compute_ci(data, estimator, alpha)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def compute_ci_for_difference(
        self,
        data_a: np.ndarray,
        data_b: np.ndarray,
        estimator: Callable[[np.ndarray], float],
        alpha: float = 0.05
    ) -> Tuple[float, float, float]:
        """
        Compute bootstrap CI for difference between two datasets.
        
        Args:
            data_a: First dataset
            data_b: Second dataset
            estimator: Estimator function
            alpha: Significance level
            
        Returns:
            Tuple of (difference, lower_bound, upper_bound)
        """
        n_a = len(data_a)
        n_b = len(data_b)
        
        bootstrap_differences = []
        
        logger.info("Computing bootstrap CI for difference...")
        
        for i in range(self.n_samples):
            # Resample both datasets
            indices_a = np.random.choice(n_a, size=n_a, replace=True)
            indices_b = np.random.choice(n_b, size=n_b, replace=True)
            
            sample_a = data_a[indices_a]
            sample_b = data_b[indices_b]
            
            # Compute estimates and difference
            est_a = estimator(sample_a)
            est_b = estimator(sample_b)
            difference = est_a - est_b
            
            bootstrap_differences.append(difference)
        
        bootstrap_differences = np.array(bootstrap_differences)
        
        # Compute point estimate of difference
        point_diff = estimator(data_a) - estimator(data_b)
        
        # Compute confidence bounds
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        lower_bound = np.percentile(bootstrap_differences, lower_percentile)
        upper_bound = np.percentile(bootstrap_differences, upper_percentile)
        
        logger.info(
            "Bootstrap CI for difference computed",
            difference=point_diff,
            ci=f"[{lower_bound:.4f}, {upper_bound:.4f}]"
        )
        
        return float(point_diff), float(lower_bound), float(upper_bound)
