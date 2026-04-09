"""
Precision Monitor for mixed precision training.

Monitors numerical stability in FP16, BF16, and FP8 training.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Tuple
import logging

logger = logging.getLogger(__name__)


class PrecisionMonitor:
    """
    Monitors numerical stability in mixed precision training.
    
    Mixed precision training (FP16/BF16/FP8) can suffer from:
    - Gradient overflow (values too large for reduced precision)
    - Gradient underflow (values too small, become zero)
    - Loss scaling failures
    - Accumulation errors
    
    This monitor:
    1. Detects overflow/underflow events
    2. Tracks loss scale adjustments
    3. Recommends optimal precision (FP16 vs BF16)
    4. Estimates accumulation errors
    
    Args:
        precision: Precision format ("fp16", "bf16", "fp8", "fp32")
        loss_scale_init: Initial loss scale for FP16 (default: 2^16)
        loss_scale_window: Window for loss scale adjustment (default: 1000)
        overflow_threshold: Consecutive overflows before alert (default: 5)
        underflow_threshold: Fraction of zeros before alert (default: 0.5)
    
    Example:
        >>> monitor = PrecisionMonitor(precision="fp16")
        >>> 
        >>> # After backward pass
        >>> overflow = monitor.check_overflow(model.parameters())
        >>> underflow = monitor.check_underflow(model.parameters())
        >>> 
        >>> if overflow:
        >>>     print("WARNING: Gradient overflow detected!")
        >>>     # Reduce loss scale
        >>> 
        >>> if monitor.should_switch_to_bf16():
        >>>     print("SUGGESTION: Consider switching to BF16")
    """
    
    def __init__(
        self,
        precision: str = "fp16",
        loss_scale_init: float = 2**16,
        loss_scale_window: int = 1000,
        overflow_threshold: int = 5,
        underflow_threshold: float = 0.5
    ):
        if precision not in ["fp16", "bf16", "fp8", "fp32"]:
            raise ValueError(f"Invalid precision: {precision}. Must be one of: fp16, bf16, fp8, fp32")
        
        self.precision = precision
        self.loss_scale = loss_scale_init
        self.loss_scale_window = loss_scale_window
        self.overflow_threshold = overflow_threshold
        self.underflow_threshold = underflow_threshold
        
        # Statistics
        self.overflow_count = 0
        self.underflow_count = 0
        self.consecutive_overflows = 0
        self.loss_scale_history: List[float] = []
        self.overflow_history: List[bool] = []
        self.underflow_history: List[bool] = []
        self.step_count = 0
        
        # Precision characteristics
        self.precision_info = {
            "fp32": {"range": (1.4e-45, 3.4e38), "mantissa_bits": 23},
            "fp16": {"range": (6.1e-5, 6.5e4), "mantissa_bits": 10},
            "bf16": {"range": (1.2e-38, 3.4e38), "mantissa_bits": 7},
            "fp8": {"range": (1e-10, 1e5), "mantissa_bits": 3},  # Approximate
        }
        
        logger.info(
            f"PrecisionMonitor initialized: {precision}\n"
            f"  Loss scale: {loss_scale_init}\n"
            f"  Overflow threshold: {overflow_threshold}\n"
            f"  Underflow threshold: {underflow_threshold}"
        )
    
    def check_overflow(self, gradients: List[torch.Tensor]) -> bool:
        """
        Check if any gradients have overflowed (inf/nan).
        
        Args:
            gradients: List of gradient tensors
        
        Returns:
            True if overflow detected
        """
        overflow_detected = False
        
        for grad in gradients:
            if grad is not None:
                if torch.isnan(grad).any() or torch.isinf(grad).any():
                    overflow_detected = True
                    break
        
        # Update statistics
        self.overflow_history.append(overflow_detected)
        if overflow_detected:
            self.overflow_count += 1
            self.consecutive_overflows += 1
        else:
            self.consecutive_overflows = 0
        
        # Keep history bounded
        if len(self.overflow_history) > self.loss_scale_window:
            self.overflow_history.pop(0)
        
        # Alert on consecutive overflows
        if self.consecutive_overflows >= self.overflow_threshold:
            logger.warning(
                f"OVERFLOW ALERT: {self.consecutive_overflows} consecutive overflows detected!"
            )
        
        return overflow_detected
    
    def check_underflow(self, gradients: List[torch.Tensor]) -> bool:
        """
        Check if gradients have underflowed (too many zeros).
        
        Underflow occurs when values are too small for the precision format
        and get rounded to zero.
        
        Args:
            gradients: List of gradient tensors
        
        Returns:
            True if underflow detected
        """
        total_elements = 0
        zero_elements = 0
        
        for grad in gradients:
            if grad is not None:
                total_elements += grad.numel()
                zero_elements += (grad == 0).sum().item()
        
        if total_elements == 0:
            return False
        
        zero_fraction = zero_elements / total_elements
        underflow_detected = zero_fraction > self.underflow_threshold
        
        # Update statistics
        self.underflow_history.append(underflow_detected)
        if underflow_detected:
            self.underflow_count += 1
            logger.warning(
                f"UNDERFLOW: {zero_fraction:.2%} of gradients are zero "
                f"(threshold: {self.underflow_threshold:.2%})"
            )
        
        # Keep history bounded
        if len(self.underflow_history) > self.loss_scale_window:
            self.underflow_history.pop(0)
        
        return underflow_detected
    
    def get_gradient_range(self, gradients: List[torch.Tensor]) -> Tuple[float, float]:
        """
        Get the range of gradient values (min, max absolute values).
        
        Args:
            gradients: List of gradient tensors
        
        Returns:
            Tuple of (min_abs, max_abs) gradient values
        """
        min_val = float('inf')
        max_val = 0.0
        
        for grad in gradients:
            if grad is not None:
                grad_abs = grad.abs()
                # Filter out zeros and inf/nan
                valid_grads = grad_abs[(grad_abs > 0) & torch.isfinite(grad_abs)]
                
                if valid_grads.numel() > 0:
                    min_val = min(min_val, valid_grads.min().item())
                    max_val = max(max_val, valid_grads.max().item())
        
        if min_val == float('inf'):
            min_val = 0.0
        
        return min_val, max_val
    
    def recommend_precision(self) -> str:
        """
        Recommend optimal precision based on observed statistics.
        
        Recommendations:
        - FP16: Good for most cases, but prone to overflow
        - BF16: Better range than FP16, less prone to overflow
        - FP32: Fallback for numerical stability
        
        Returns:
            Recommended precision ("fp16", "bf16", "fp32")
        """
        if not self.overflow_history:
            return self.precision
        
        # Calculate overflow rate
        recent_overflows = self.overflow_history[-100:] if len(self.overflow_history) > 100 else self.overflow_history
        overflow_rate = sum(recent_overflows) / len(recent_overflows)
        
        # Calculate underflow rate
        recent_underflows = self.underflow_history[-100:] if len(self.underflow_history) > 100 else self.underflow_history
        underflow_rate = sum(recent_underflows) / len(recent_underflows) if recent_underflows else 0.0
        
        # Recommendation logic
        if overflow_rate > 0.1:  # >10% overflow rate
            if self.precision == "fp16":
                return "bf16"  # BF16 has better range
            else:
                return "fp32"  # Fallback to FP32
        
        if underflow_rate > 0.2:  # >20% underflow rate
            if self.precision == "fp16":
                return "fp32"  # Need more precision
            else:
                return "fp32"
        
        return self.precision  # Current precision is fine
    
    def should_switch_to_bf16(self) -> bool:
        """
        Check if switching from FP16 to BF16 is recommended.
        
        Returns:
            True if switch recommended
        """
        if self.precision != "fp16":
            return False
        
        return self.recommend_precision() == "bf16"
    
    def update_loss_scale(self, new_scale: float):
        """
        Update loss scale and track history.
        
        Args:
            new_scale: New loss scale value
        """
        self.loss_scale = new_scale
        self.loss_scale_history.append(new_scale)
        
        # Keep history bounded
        if len(self.loss_scale_history) > self.loss_scale_window:
            self.loss_scale_history.pop(0)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get monitoring statistics.
        
        Returns:
            Dictionary with:
            - precision: Current precision
            - overflow_count: Total overflows
            - underflow_count: Total underflows
            - consecutive_overflows: Current consecutive overflows
            - overflow_rate: Recent overflow rate
            - underflow_rate: Recent underflow rate
            - loss_scale: Current loss scale
            - recommended_precision: Recommended precision
        """
        # Calculate rates
        overflow_rate = (
            sum(self.overflow_history[-100:]) / len(self.overflow_history[-100:])
            if self.overflow_history else 0.0
        )
        underflow_rate = (
            sum(self.underflow_history[-100:]) / len(self.underflow_history[-100:])
            if self.underflow_history else 0.0
        )
        
        return {
            "precision": self.precision,
            "overflow_count": self.overflow_count,
            "underflow_count": self.underflow_count,
            "consecutive_overflows": self.consecutive_overflows,
            "overflow_rate": overflow_rate,
            "underflow_rate": underflow_rate,
            "loss_scale": self.loss_scale,
            "recommended_precision": self.recommend_precision(),
            "step_count": self.step_count,
        }
    
    def reset(self):
        """Reset monitoring statistics."""
        self.overflow_count = 0
        self.underflow_count = 0
        self.consecutive_overflows = 0
        self.loss_scale_history = []
        self.overflow_history = []
        self.underflow_history = []
        self.step_count = 0
        
        logger.info("PrecisionMonitor reset")

