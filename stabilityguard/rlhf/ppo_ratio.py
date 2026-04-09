"""
PPO Ratio Monitor for RLHF training.

Monitors the importance sampling ratio in PPO to detect extreme values.
"""

import torch
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class PPORatioMonitor:
    """
    Monitors PPO importance sampling ratio for stability.
    
    In PPO, the ratio r = π_θ(a|s) / π_old(a|s) measures how much the policy has changed.
    The PPO objective clips this ratio to [1-ε, 1+ε] to prevent large policy updates.
    
    Extreme ratios indicate:
    - Policy is changing too rapidly
    - Training may become unstable
    - Clipping is happening too frequently
    
    This monitor tracks:
    1. Ratio statistics (mean, max, min)
    2. Clipping frequency
    3. Ratio variance
    4. Extreme ratio events (>10 or <0.1)
    
    Args:
        clip_range: PPO clip range ε (default: 0.2)
        extreme_ratio_threshold: Threshold for extreme ratios (default: 10.0)
        clipping_frequency_threshold: Max acceptable clipping frequency (default: 0.5)
        window_size: Number of steps to track (default: 100)
        alert_patience: Steps above threshold before alerting (default: 5)
    
    Example:
        >>> monitor = PPORatioMonitor(clip_range=0.2)
        >>> 
        >>> # During PPO update
        >>> ratio = torch.exp(policy_logprobs - old_logprobs)
        >>> 
        >>> stats = monitor.check_ratio(policy_logprobs, old_logprobs)
        >>> if stats['extreme_ratio_detected']:
        >>>     print("WARNING: Extreme PPO ratio detected!")
        >>>     # Consider reducing learning rate
    """
    
    def __init__(
        self,
        clip_range: float = 0.2,
        extreme_ratio_threshold: float = 10.0,
        clipping_frequency_threshold: float = 0.5,
        window_size: int = 100,
        alert_patience: int = 5
    ):
        self.clip_range = clip_range
        self.extreme_ratio_threshold = extreme_ratio_threshold
        self.clipping_frequency_threshold = clipping_frequency_threshold
        self.window_size = window_size
        self.alert_patience = alert_patience
        
        # Statistics tracking
        self.ratio_history: List[float] = []
        self.clipping_freq_history: List[float] = []
        self.extreme_ratio_history: List[bool] = []
        
        self.steps_above_threshold = 0
        self.alert_triggered = False
        
        logger.info(
            f"PPORatioMonitor initialized: clip_range={clip_range}, "
            f"extreme_threshold={extreme_ratio_threshold}, patience={alert_patience}"
        )
    
    def compute_ratio(
        self,
        policy_logprobs: torch.Tensor,
        old_logprobs: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute importance sampling ratio: r = π_θ(a|s) / π_old(a|s)
        
        In log-space: r = exp(log π_θ - log π_old)
        
        Args:
            policy_logprobs: Log probabilities from current policy [batch_size]
            old_logprobs: Log probabilities from old policy [batch_size]
        
        Returns:
            Importance sampling ratios [batch_size]
        """
        ratio = torch.exp(policy_logprobs - old_logprobs)
        return ratio
    
    def compute_clipping_frequency(
        self,
        ratio: torch.Tensor,
        clip_range: Optional[float] = None
    ) -> float:
        """
        Compute fraction of ratios that are clipped.
        
        Args:
            ratio: Importance sampling ratios [batch_size]
            clip_range: Clip range (default: self.clip_range)
        
        Returns:
            Fraction of clipped ratios (0.0 to 1.0)
        """
        if clip_range is None:
            clip_range = self.clip_range
        
        lower_bound = 1.0 - clip_range
        upper_bound = 1.0 + clip_range
        
        clipped = (ratio < lower_bound) | (ratio > upper_bound)
        clipping_freq = clipped.float().mean().item()
        
        return clipping_freq
    
    def is_extreme_ratio(self, ratio: torch.Tensor) -> bool:
        """
        Check if any ratios are extreme (>threshold or <1/threshold).
        
        Args:
            ratio: Importance sampling ratios [batch_size]
        
        Returns:
            True if extreme ratios detected
        """
        extreme_high = (ratio > self.extreme_ratio_threshold).any()
        extreme_low = (ratio < 1.0 / self.extreme_ratio_threshold).any()
        
        return extreme_high or extreme_low
    
    def check_ratio(
        self,
        policy_logprobs: torch.Tensor,
        old_logprobs: torch.Tensor,
        clip_range: Optional[float] = None
    ) -> Dict[str, any]:
        """
        Analyze PPO ratio health.
        
        Args:
            policy_logprobs: Log probabilities from current policy
            old_logprobs: Log probabilities from old policy
            clip_range: Optional clip range (default: self.clip_range)
        
        Returns:
            Dictionary with:
            - ratio_mean: Mean ratio
            - ratio_max: Maximum ratio
            - ratio_min: Minimum ratio
            - ratio_std: Standard deviation
            - clipping_frequency: Fraction of clipped ratios
            - extreme_ratio_detected: Whether extreme ratios detected
            - alert_triggered: Whether alert has been triggered
        """
        # Compute ratio
        ratio = self.compute_ratio(policy_logprobs, old_logprobs)
        
        # Compute statistics
        ratio_mean = ratio.mean().item()
        ratio_max = ratio.max().item()
        ratio_min = ratio.min().item()
        ratio_std = ratio.std().item()
        
        clipping_freq = self.compute_clipping_frequency(ratio, clip_range)
        extreme_detected = self.is_extreme_ratio(ratio)
        
        # Update history
        self.ratio_history.append(ratio_mean)
        self.clipping_freq_history.append(clipping_freq)
        self.extreme_ratio_history.append(extreme_detected)
        
        # Keep only recent history
        if len(self.ratio_history) > self.window_size:
            self.ratio_history.pop(0)
            self.clipping_freq_history.pop(0)
            self.extreme_ratio_history.pop(0)
        
        # Check alert conditions
        alert_signals = []
        
        # 1. High clipping frequency
        if clipping_freq > self.clipping_frequency_threshold:
            alert_signals.append("high_clipping_frequency")
        
        # 2. Extreme ratios
        if extreme_detected:
            alert_signals.append("extreme_ratio")
        
        # 3. High ratio variance
        if ratio_std > 2.0:  # Arbitrary threshold
            alert_signals.append("high_variance")
        
        # Update alert counter
        if len(alert_signals) >= 1:
            self.steps_above_threshold += 1
        else:
            self.steps_above_threshold = 0
        
        # Trigger alert if persistent
        if self.steps_above_threshold >= self.alert_patience:
            if not self.alert_triggered:
                logger.warning(
                    f"PPO RATIO ALERT!\n"
                    f"  Signals: {', '.join(alert_signals)}\n"
                    f"  Ratio: mean={ratio_mean:.4f}, max={ratio_max:.4f}, min={ratio_min:.4f}, std={ratio_std:.4f}\n"
                    f"  Clipping frequency: {clipping_freq:.2%} (threshold: {self.clipping_frequency_threshold:.2%})\n"
                    f"  Extreme ratio: {extreme_detected}\n"
                    f"  Steps above threshold: {self.steps_above_threshold}"
                )
                self.alert_triggered = True
        
        return {
            "ratio_mean": ratio_mean,
            "ratio_max": ratio_max,
            "ratio_min": ratio_min,
            "ratio_std": ratio_std,
            "clipping_frequency": clipping_freq,
            "extreme_ratio_detected": extreme_detected,
            "alert_triggered": self.alert_triggered,
            "alert_signals": alert_signals,
        }
    
    def get_stats(self) -> Dict[str, any]:
        """
        Get current monitor statistics.
        
        Returns:
            Dictionary with:
            - current_ratio: Current mean ratio
            - current_clipping_freq: Current clipping frequency
            - extreme_ratio_rate: Fraction of steps with extreme ratios
            - alert_triggered: Whether alert has been triggered
            - steps_above_threshold: Steps above threshold
        """
        extreme_ratio_rate = (
            sum(self.extreme_ratio_history) / len(self.extreme_ratio_history)
            if self.extreme_ratio_history else 0.0
        )
        
        return {
            "current_ratio": self.ratio_history[-1] if self.ratio_history else None,
            "current_clipping_freq": self.clipping_freq_history[-1] if self.clipping_freq_history else None,
            "extreme_ratio_rate": extreme_ratio_rate,
            "alert_triggered": self.alert_triggered,
            "steps_above_threshold": self.steps_above_threshold,
        }
    
    def reset(self):
        """Reset monitor state."""
        self.ratio_history = []
        self.clipping_freq_history = []
        self.extreme_ratio_history = []
        self.steps_above_threshold = 0
        self.alert_triggered = False
        logger.info("PPORatioMonitor reset")

# Made with Bob
