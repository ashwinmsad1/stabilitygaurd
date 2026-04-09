"""
Reward Model Collapse Detector for RLHF training.

Detects when the reward model becomes degenerate (all outputs converge to same value).
"""

import torch
import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class RewardCollapseDetector:
    """
    Detects reward model collapse during RLHF training.
    
    Reward model collapse occurs when the reward model becomes degenerate:
    - All rewards converge to the same value
    - Reward variance approaches zero
    - Reward distribution becomes bimodal (only high/low)
    - Gradients vanish (model stops learning)
    
    This detector monitors:
    1. Reward variance over time
    2. Reward distribution shape
    3. Gradient norms of reward model
    4. Reward prediction entropy
    
    Args:
        variance_threshold: Minimum acceptable reward variance (default: 0.01)
        gradient_threshold: Minimum acceptable gradient norm (default: 1e-6)
        entropy_threshold: Minimum acceptable entropy (default: 0.1)
        window_size: Number of steps to track for statistics (default: 100)
        collapse_patience: Steps below threshold before declaring collapse (default: 10)
    
    Example:
        >>> detector = RewardCollapseDetector()
        >>> 
        >>> # During training
        >>> rewards = reward_model(responses)
        >>> reward_grads = get_gradients(reward_model)
        >>> 
        >>> is_collapsed = detector.detect_collapse(rewards, reward_grads)
        >>> if is_collapsed:
        >>>     print("WARNING: Reward model collapse detected!")
        >>>     # Stop training, investigate reward model
    """
    
    def __init__(
        self,
        variance_threshold: float = 0.01,
        gradient_threshold: float = 1e-6,
        entropy_threshold: float = 0.1,
        window_size: int = 100,
        collapse_patience: int = 10
    ):
        self.variance_threshold = variance_threshold
        self.gradient_threshold = gradient_threshold
        self.entropy_threshold = entropy_threshold
        self.window_size = window_size
        self.collapse_patience = collapse_patience
        
        # Statistics tracking
        self.reward_history: List[float] = []
        self.variance_history: List[float] = []
        self.gradient_history: List[float] = []
        self.entropy_history: List[float] = []
        
        self.steps_below_threshold = 0
        self.collapse_detected = False
        
        logger.info(
            f"RewardCollapseDetector initialized: var_threshold={variance_threshold}, "
            f"grad_threshold={gradient_threshold}, patience={collapse_patience}"
        )
    
    def compute_reward_variance(self, rewards: torch.Tensor) -> float:
        """
        Compute variance of reward predictions.
        
        Args:
            rewards: Reward predictions [batch_size] or [batch_size, seq_len]
        
        Returns:
            Reward variance
        """
        return rewards.var().item()
    
    def compute_reward_entropy(self, rewards: torch.Tensor) -> float:
        """
        Compute entropy of reward distribution.
        
        Higher entropy = more diverse rewards (good)
        Lower entropy = rewards clustering (bad)
        
        Args:
            rewards: Reward predictions [batch_size]
        
        Returns:
            Entropy of reward distribution
        """
        # Normalize rewards to [0, 1] for probability-like distribution
        rewards_norm = (rewards - rewards.min()) / (rewards.max() - rewards.min() + 1e-8)
        
        # Compute histogram-based entropy
        hist, _ = np.histogram(rewards_norm.cpu().numpy(), bins=20, density=False)
        # Normalize histogram to get probabilities
        hist = hist / (hist.sum() + 1e-8)
        # Remove zero bins
        hist = hist[hist > 0]
        # Compute entropy: H = -sum(p * log(p))
        entropy = -np.sum(hist * np.log(hist + 1e-8))
        
        return float(entropy)
    
    def compute_gradient_norm(self, reward_grads: torch.Tensor) -> float:
        """
        Compute L2 norm of reward model gradients.
        
        Args:
            reward_grads: Gradients of reward model parameters
        
        Returns:
            L2 norm of gradients
        """
        if reward_grads is None:
            return 0.0
        
        return torch.norm(reward_grads).item()
    
    def is_bimodal(self, rewards: torch.Tensor) -> bool:
        """
        Check if reward distribution is bimodal (only high/low values).
        
        Args:
            rewards: Reward predictions
        
        Returns:
            True if distribution appears bimodal
        """
        # Compute histogram
        hist, bin_edges = np.histogram(rewards.cpu().numpy(), bins=10)
        
        # Find peaks
        peaks = []
        for i in range(1, len(hist) - 1):
            if hist[i] > hist[i-1] and hist[i] > hist[i+1]:
                peaks.append(i)
        
        # Bimodal if exactly 2 peaks and middle bins are sparse
        if len(peaks) == 2:
            middle_start = peaks[0] + 1
            middle_end = peaks[1]
            middle_density = sum(hist[middle_start:middle_end]) / len(hist)
            
            return middle_density < 0.1  # Less than 10% in middle
        
        return False
    
    def detect_collapse(
        self,
        rewards: torch.Tensor,
        reward_grads: Optional[torch.Tensor] = None
    ) -> bool:
        """
        Detect if reward model has collapsed.
        
        Collapse is detected if:
        1. Reward variance < threshold for multiple steps
        2. Gradient norm < threshold (vanishing gradients)
        3. Entropy < threshold (low diversity)
        4. Distribution is bimodal
        
        Args:
            rewards: Reward predictions [batch_size]
            reward_grads: Optional gradients of reward model
        
        Returns:
            True if collapse detected
        """
        # Compute statistics
        variance = self.compute_reward_variance(rewards)
        entropy = self.compute_reward_entropy(rewards)
        bimodal = self.is_bimodal(rewards)
        
        # Update history
        self.reward_history.append(rewards.mean().item())
        self.variance_history.append(variance)
        self.entropy_history.append(entropy)
        
        # Compute gradient norm if provided
        grad_norm = 0.0
        if reward_grads is not None:
            grad_norm = self.compute_gradient_norm(reward_grads)
            self.gradient_history.append(grad_norm)
        
        # Keep only recent history
        if len(self.reward_history) > self.window_size:
            self.reward_history.pop(0)
            self.variance_history.pop(0)
            self.entropy_history.pop(0)
            if len(self.gradient_history) > 0:
                self.gradient_history.pop(0)
        
        # Check collapse conditions
        collapse_signals = []
        
        # 1. Low variance
        if variance < self.variance_threshold:
            collapse_signals.append("low_variance")
        
        # 2. Vanishing gradients
        if reward_grads is not None and grad_norm < self.gradient_threshold:
            collapse_signals.append("vanishing_gradients")
        
        # 3. Low entropy
        if entropy < self.entropy_threshold:
            collapse_signals.append("low_entropy")
        
        # 4. Bimodal distribution
        if bimodal:
            collapse_signals.append("bimodal_distribution")
        
        # Update collapse counter
        if len(collapse_signals) >= 2:  # At least 2 signals
            self.steps_below_threshold += 1
        else:
            self.steps_below_threshold = 0
        
        # Declare collapse if persistent
        if self.steps_below_threshold >= self.collapse_patience:
            if not self.collapse_detected:
                logger.warning(
                    f"REWARD MODEL COLLAPSE DETECTED!\n"
                    f"  Signals: {', '.join(collapse_signals)}\n"
                    f"  Variance: {variance:.6f} (threshold: {self.variance_threshold})\n"
                    f"  Entropy: {entropy:.4f} (threshold: {self.entropy_threshold})\n"
                    f"  Grad norm: {grad_norm:.6e} (threshold: {self.gradient_threshold})\n"
                    f"  Steps below threshold: {self.steps_below_threshold}"
                )
                self.collapse_detected = True
            return True
        
        return False
    
    def get_stats(self) -> Dict[str, any]:
        """
        Get current detector statistics.
        
        Returns:
            Dictionary with:
            - current_variance: Current reward variance
            - current_entropy: Current reward entropy
            - current_grad_norm: Current gradient norm
            - mean_reward: Mean reward over window
            - collapse_detected: Whether collapse has been detected
            - steps_below_threshold: Steps below threshold
        """
        return {
            "current_variance": self.variance_history[-1] if self.variance_history else None,
            "current_entropy": self.entropy_history[-1] if self.entropy_history else None,
            "current_grad_norm": self.gradient_history[-1] if self.gradient_history else None,
            "mean_reward": np.mean(self.reward_history) if self.reward_history else None,
            "collapse_detected": self.collapse_detected,
            "steps_below_threshold": self.steps_below_threshold,
        }
    
    def reset(self):
        """Reset detector state."""
        self.reward_history = []
        self.variance_history = []
        self.gradient_history = []
        self.entropy_history = []
        self.steps_below_threshold = 0
        self.collapse_detected = False
        logger.info("RewardCollapseDetector reset")

# Made with Bob
