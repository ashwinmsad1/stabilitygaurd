"""
Value Function Divergence Monitor for RLHF training.

Monitors the critic (value function) for signs of divergence or instability.
"""

import torch
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class ValueDivergenceMonitor:
    """
    Monitors value function (critic) stability during RLHF training.
    
    The value function V(s) estimates expected return from state s. Divergence occurs when:
    - Value predictions become unrealistic (too high/low)
    - TD errors explode
    - Value gradients vanish or explode
    - Advantage estimates become unreliable
    
    This monitor tracks:
    1. Value prediction error (V - actual return)
    2. TD error magnitude and trends
    3. Value gradient norms
    4. Advantage estimation quality
    
    Args:
        td_error_threshold: Maximum acceptable TD error (default: 10.0)
        value_grad_threshold_max: Maximum value gradient norm (default: 100.0)
        value_grad_threshold_min: Minimum value gradient norm (default: 1e-6)
        advantage_std_threshold: Maximum advantage std deviation (default: 50.0)
        window_size: Number of steps to track (default: 100)
        divergence_patience: Steps above threshold before declaring divergence (default: 10)
    
    Example:
        >>> monitor = ValueDivergenceMonitor()
        >>> 
        >>> # During training
        >>> values = critic(states)
        >>> returns = compute_returns(rewards)
        >>> advantages = compute_advantages(values, returns)
        >>> 
        >>> is_diverged = monitor.check_divergence(values, returns, advantages)
        >>> if is_diverged:
        >>>     print("WARNING: Value function divergence detected!")
    """
    
    def __init__(
        self,
        td_error_threshold: float = 10.0,
        value_grad_threshold_max: float = 100.0,
        value_grad_threshold_min: float = 1e-6,
        advantage_std_threshold: float = 50.0,
        window_size: int = 100,
        divergence_patience: int = 10
    ):
        self.td_error_threshold = td_error_threshold
        self.value_grad_threshold_max = value_grad_threshold_max
        self.value_grad_threshold_min = value_grad_threshold_min
        self.advantage_std_threshold = advantage_std_threshold
        self.window_size = window_size
        self.divergence_patience = divergence_patience
        
        # Statistics tracking
        self.td_error_history: List[float] = []
        self.value_grad_history: List[float] = []
        self.advantage_std_history: List[float] = []
        self.value_error_history: List[float] = []
        
        self.steps_above_threshold = 0
        self.divergence_detected = False
        
        logger.info(
            f"ValueDivergenceMonitor initialized: td_threshold={td_error_threshold}, "
            f"grad_max={value_grad_threshold_max}, patience={divergence_patience}"
        )
    
    def compute_td_error(
        self,
        values: torch.Tensor,
        next_values: torch.Tensor,
        rewards: torch.Tensor,
        gamma: float = 0.99
    ) -> torch.Tensor:
        """
        Compute TD error: δ = r + γV(s') - V(s)
        
        Args:
            values: Value predictions for current states [batch_size]
            next_values: Value predictions for next states [batch_size]
            rewards: Rewards [batch_size]
            gamma: Discount factor (default: 0.99)
        
        Returns:
            TD errors [batch_size]
        """
        td_error = rewards + gamma * next_values - values
        return td_error
    
    def compute_value_error(
        self,
        values: torch.Tensor,
        returns: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute value prediction error: V(s) - G (actual return)
        
        Args:
            values: Value predictions [batch_size]
            returns: Actual returns [batch_size]
        
        Returns:
            Value errors [batch_size]
        """
        return values - returns
    
    def check_divergence(
        self,
        values: torch.Tensor,
        returns: torch.Tensor,
        advantages: torch.Tensor,
        value_grads: Optional[torch.Tensor] = None
    ) -> bool:
        """
        Check if value function has diverged.
        
        Divergence is detected if:
        1. TD errors exceed threshold
        2. Value gradients explode or vanish
        3. Advantage std deviation is too high
        4. Value prediction errors are extreme
        
        Args:
            values: Value predictions [batch_size]
            returns: Actual returns [batch_size]
            advantages: Advantage estimates [batch_size]
            value_grads: Optional gradients of value function
        
        Returns:
            True if divergence detected
        """
        # Compute statistics
        value_error = self.compute_value_error(values, returns)
        value_error_mean = value_error.abs().mean().item()
        
        advantage_std = advantages.std().item()
        
        # Update history
        self.value_error_history.append(value_error_mean)
        self.advantage_std_history.append(advantage_std)
        
        # Compute gradient norm if provided
        grad_norm = 0.0
        if value_grads is not None:
            grad_norm = torch.norm(value_grads).item()
            self.value_grad_history.append(grad_norm)
        
        # Keep only recent history
        if len(self.value_error_history) > self.window_size:
            self.value_error_history.pop(0)
            self.advantage_std_history.pop(0)
            if len(self.value_grad_history) > 0:
                self.value_grad_history.pop(0)
        
        # Check divergence conditions
        divergence_signals = []
        
        # 1. High value prediction error
        if value_error_mean > self.td_error_threshold:
            divergence_signals.append("high_value_error")
        
        # 2. Exploding gradients
        if value_grads is not None and grad_norm > self.value_grad_threshold_max:
            divergence_signals.append("exploding_gradients")
        
        # 3. Vanishing gradients
        if value_grads is not None and grad_norm < self.value_grad_threshold_min:
            divergence_signals.append("vanishing_gradients")
        
        # 4. High advantage variance
        if advantage_std > self.advantage_std_threshold:
            divergence_signals.append("high_advantage_variance")
        
        # Update divergence counter
        # Count if we have any divergence signal
        if len(divergence_signals) >= 1:
            self.steps_above_threshold += 1
        else:
            self.steps_above_threshold = 0
        
        # Declare divergence if persistent
        if self.steps_above_threshold >= self.divergence_patience:
            if not self.divergence_detected:
                logger.warning(
                    f"VALUE FUNCTION DIVERGENCE DETECTED!\n"
                    f"  Signals: {', '.join(divergence_signals)}\n"
                    f"  Value error: {value_error_mean:.4f} (threshold: {self.td_error_threshold})\n"
                    f"  Advantage std: {advantage_std:.4f} (threshold: {self.advantage_std_threshold})\n"
                    f"  Grad norm: {grad_norm:.6e} (min: {self.value_grad_threshold_min}, max: {self.value_grad_threshold_max})\n"
                    f"  Steps above threshold: {self.steps_above_threshold}"
                )
                self.divergence_detected = True
            return True
        
        return False
    
    def get_stats(self) -> Dict[str, float]:
        """
        Get current monitor statistics.
        
        Returns:
            Dictionary with:
            - current_value_error: Current value prediction error
            - current_advantage_std: Current advantage std deviation
            - current_grad_norm: Current gradient norm
            - divergence_detected: Whether divergence has been detected
            - steps_above_threshold: Steps above threshold
        """
        return {
            "current_value_error": self.value_error_history[-1] if self.value_error_history else None,
            "current_advantage_std": self.advantage_std_history[-1] if self.advantage_std_history else None,
            "current_grad_norm": self.value_grad_history[-1] if self.value_grad_history else None,
            "divergence_detected": self.divergence_detected,
            "steps_above_threshold": self.steps_above_threshold,
        }
    
    def reset(self):
        """Reset monitor state."""
        self.td_error_history = []
        self.value_grad_history = []
        self.advantage_std_history = []
        self.value_error_history = []
        self.steps_above_threshold = 0
        self.divergence_detected = False
        logger.info("ValueDivergenceMonitor reset")

# Made with Bob
