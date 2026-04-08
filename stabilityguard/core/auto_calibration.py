"""
Auto-Calibration for Spike Detection Threshold

Automatically determines the optimal spike_threshold during warmup by analyzing
the distribution of gradient norms. This eliminates manual tuning and adapts to
different model architectures and datasets.

Key Insight: Gradient norms typically follow a log-normal or Weibull distribution.
By fitting this distribution during warmup, we can set the threshold at the 99th
percentile to minimize false positives while catching true spikes.
"""

import torch
import numpy as np
from typing import List, Optional, Tuple, Dict, Any
import logging
import math

logger = logging.getLogger(__name__)


class AutoCalibrator:
    """
    Automatically calibrates spike detection threshold based on observed gradient norms.
    
    During warmup period:
    1. Collect gradient norm samples from all layers
    2. Fit a statistical distribution (log-normal)
    3. Set threshold at 99th percentile (3-sigma)
    
    This ensures the threshold adapts to:
    - Model architecture (different layers have different norm scales)
    - Dataset characteristics (different data distributions)
    - Training dynamics (norms change over time)
    
    Args:
        warmup_steps: Number of steps to collect samples (default: 100)
        percentile: Percentile for threshold (default: 99.0, i.e., 99th percentile)
        min_samples: Minimum samples before calibration (default: 50)
        distribution: Distribution to fit ("lognormal" or "weibull", default: "lognormal")
        verbose: Print calibration results (default: True)
    
    Example:
        >>> calibrator = AutoCalibrator(warmup_steps=100, percentile=99.0)
        >>> 
        >>> # During warmup
        >>> for step in range(100):
        >>>     norms = compute_gradient_norms(model)
        >>>     calibrator.add_samples(norms)
        >>> 
        >>> # After warmup
        >>> threshold = calibrator.get_threshold()
        >>> print(f"Calibrated threshold: {threshold:.2f}")
    """
    
    def __init__(
        self,
        warmup_steps: int = 100,
        percentile: float = 99.0,
        min_samples: int = 50,
        distribution: str = "lognormal",
        verbose: bool = True
    ):
        self.warmup_steps = warmup_steps
        self.percentile = percentile
        self.min_samples = min_samples
        self.distribution = distribution
        self.verbose = verbose
        
        # Sample collection
        self.samples: List[float] = []
        self.step_count = 0
        self.is_calibrated = False
        
        # Fitted parameters
        self.threshold = None
        self.distribution_params = None
        
    def add_samples(self, norms: Dict[str, float]):
        """
        Add gradient norm samples from current step.
        
        Args:
            norms: Dictionary mapping layer names to gradient norms
                   e.g., {"layer1": 1.5, "layer2": 2.3, ...}
        """
        if self.is_calibrated:
            return  # Already calibrated, no need to collect more samples
        
        # Add all norms to samples
        self.samples.extend(norms.values())
        self.step_count += 1
        
        # Check if we have enough samples to calibrate
        if self.step_count >= self.warmup_steps and len(self.samples) >= self.min_samples:
            self._calibrate()
    
    def _calibrate(self):
        """
        Fit distribution to collected samples and compute threshold.
        """
        if len(self.samples) < self.min_samples:
            logger.warning(
                f"Not enough samples for calibration: {len(self.samples)} < {self.min_samples}"
            )
            # Use default threshold
            self.threshold = 10.0
            self.is_calibrated = True
            return
        
        # Convert to numpy array
        samples = np.array(self.samples)
        
        # Remove zeros and negative values (shouldn't happen, but be safe)
        samples = samples[samples > 0]
        
        if len(samples) == 0:
            logger.warning("All samples are zero or negative")
            self.threshold = 10.0
            self.is_calibrated = True
            return
        
        try:
            if self.distribution == "lognormal":
                self.threshold, self.distribution_params = self._fit_lognormal(samples)
            elif self.distribution == "weibull":
                self.threshold, self.distribution_params = self._fit_weibull(samples)
            else:
                logger.warning(f"Unknown distribution: {self.distribution}, using empirical")
                self.threshold, self.distribution_params = self._empirical_threshold(samples)
            
            self.is_calibrated = True
            
            if self.verbose:
                logger.info(
                    f"✅ AUTO-CALIBRATION COMPLETE\n"
                    f"   Samples collected: {len(self.samples)}\n"
                    f"   Distribution: {self.distribution}\n"
                    f"   Parameters: {self.distribution_params}\n"
                    f"   Threshold (ratio): {self.threshold:.2f}\n"
                    f"   Percentile: {self.percentile}th"
                )
        
        except Exception as e:
            logger.error(f"Calibration failed: {e}, using default threshold")
            self.threshold = 10.0
            self.is_calibrated = True
    
    def _fit_lognormal(self, samples: np.ndarray) -> Tuple[float, Dict[str, float]]:
        """
        Fit log-normal distribution to samples.
        
        Log-normal is appropriate for gradient norms because:
        - Norms are always positive
        - Distribution is right-skewed (most norms are small, few are large)
        - Log of norms is approximately normal
        
        Args:
            samples: Array of gradient norm samples
            
        Returns:
            Tuple of (threshold, parameters)
        """
        # Take log of samples
        log_samples = np.log(samples)
        
        # Fit normal distribution to log samples
        mu = np.mean(log_samples)
        sigma = np.std(log_samples)
        
        # Compute threshold at desired percentile
        # For log-normal: threshold = exp(mu + z * sigma)
        # where z is the z-score for the desired percentile
        
        # For 99th percentile, z ≈ 2.33
        # For 99.9th percentile, z ≈ 3.09
        z_score = self._percentile_to_z_score(self.percentile)
        
        # Threshold in absolute terms
        threshold_abs = np.exp(mu + z_score * sigma)
        
        # Convert to ratio (threshold / median)
        median = np.exp(mu)
        threshold_ratio = threshold_abs / median
        
        params = {
            "mu": float(mu),
            "sigma": float(sigma),
            "median": float(median),
            "threshold_abs": float(threshold_abs)
        }
        
        return float(threshold_ratio), params
    
    def _fit_weibull(self, samples: np.ndarray) -> Tuple[float, Dict[str, float]]:
        """
        Fit Weibull distribution to samples.
        
        Weibull is another option for modeling gradient norms, especially
        when the distribution has a heavier tail than log-normal.
        
        Args:
            samples: Array of gradient norm samples
            
        Returns:
            Tuple of (threshold, parameters)
        """
        # Simple Weibull fitting using method of moments
        # (More sophisticated fitting would use scipy, but we avoid dependencies)
        
        mean = np.mean(samples)
        std = np.std(samples)
        
        # Estimate shape parameter k
        cv = std / mean  # coefficient of variation
        # For Weibull: cv ≈ sqrt(Γ(1+2/k)/Γ(1+1/k)^2 - 1)
        # Approximate: k ≈ 1 / cv for cv < 1
        k = 1.0 / max(cv, 0.1)
        
        # Estimate scale parameter λ
        # For Weibull: mean = λ * Γ(1 + 1/k)
        # Approximate: λ ≈ mean
        lambda_param = mean
        
        # Compute threshold at desired percentile
        # For Weibull: F(x) = 1 - exp(-(x/λ)^k)
        # Solve for x: x = λ * (-ln(1-p))^(1/k)
        p = self.percentile / 100.0
        threshold_abs = lambda_param * ((-np.log(1 - p)) ** (1.0 / k))
        
        # Convert to ratio
        median = lambda_param * (np.log(2) ** (1.0 / k))
        threshold_ratio = threshold_abs / median
        
        params = {
            "k": float(k),
            "lambda": float(lambda_param),
            "median": float(median),
            "threshold_abs": float(threshold_abs)
        }
        
        return float(threshold_ratio), params
    
    def _empirical_threshold(self, samples: np.ndarray) -> Tuple[float, Dict[str, float]]:
        """
        Compute threshold empirically from samples (no distribution fitting).
        
        Simply use the desired percentile of the observed samples.
        
        Args:
            samples: Array of gradient norm samples
            
        Returns:
            Tuple of (threshold, parameters)
        """
        threshold_abs = np.percentile(samples, self.percentile)
        median = np.median(samples)
        threshold_ratio = threshold_abs / median if median > 0 else 10.0
        
        params = {
            "median": float(median),
            "threshold_abs": float(threshold_abs),
            "method": "empirical"
        }
        
        return float(threshold_ratio), params
    
    def _percentile_to_z_score(self, percentile: float) -> float:
        """
        Convert percentile to z-score for normal distribution.
        
        Args:
            percentile: Percentile (0-100)
            
        Returns:
            Corresponding z-score
        """
        # Approximate inverse CDF of standard normal
        # Using Abramowitz and Stegun approximation
        
        p = percentile / 100.0
        
        if p <= 0 or p >= 1:
            return 0.0
        
        # For p > 0.5, use symmetry
        if p > 0.5:
            return -self._percentile_to_z_score(100 - percentile)
        
        # Rational approximation for 0 < p <= 0.5
        t = math.sqrt(-2.0 * math.log(p))
        c0 = 2.515517
        c1 = 0.802853
        c2 = 0.010328
        d1 = 1.432788
        d2 = 0.189269
        d3 = 0.001308
        
        z = t - (c0 + c1*t + c2*t*t) / (1.0 + d1*t + d2*t*t + d3*t*t*t)
        
        return z
    
    def get_threshold(self) -> float:
        """
        Get the calibrated threshold.
        
        Returns:
            Calibrated threshold ratio, or default (10.0) if not yet calibrated
        """
        if not self.is_calibrated:
            logger.warning("Calibration not complete, using default threshold")
            return 10.0
        
        return self.threshold
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get calibration statistics.
        
        Returns:
            Dictionary with statistics:
            - is_calibrated: Whether calibration is complete
            - samples_collected: Number of samples collected
            - threshold: Calibrated threshold
            - distribution: Distribution type
            - parameters: Fitted distribution parameters
        """
        return {
            "is_calibrated": self.is_calibrated,
            "samples_collected": len(self.samples),
            "threshold": self.threshold,
            "distribution": self.distribution,
            "parameters": self.distribution_params
        }
    
    def reset(self):
        """
        Reset calibrator to collect new samples.
        """
        self.samples = []
        self.step_count = 0
        self.is_calibrated = False
        self.threshold = None
        self.distribution_params = None

