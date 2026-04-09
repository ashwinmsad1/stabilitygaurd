"""
Adaptive Loss Scaler for FP16 training.

Provides stability-aware loss scaling that adapts based on gradient health.
"""

import torch
from typing import Dict, Optional, List
import logging

logger = logging.getLogger(__name__)


class AdaptiveLossScaler:
    """
    Stability-aware adaptive loss scaling for FP16 training.
    
    Loss scaling multiplies the loss by a large factor before backward pass,
    then unscales gradients before optimizer step. This prevents gradient
    underflow in FP16.
    
    This scaler adapts the scale based on:
    - Overflow frequency
    - Gradient spike detection
    - Training stability
    
    Features:
    - Conservative scaling near spikes
    - Faster recovery after stable periods
    - Per-layer scale recommendations (future)
    
    Args:
        init_scale: Initial loss scale (default: 2^16)
        scale_factor: Factor to multiply/divide scale (default: 2.0)
        scale_window: Steps between scale increases (default: 2000)
        min_scale: Minimum loss scale (default: 1.0)
        max_scale: Maximum loss scale (default: 2^24)
        conservative_mode: Use conservative scaling near spikes (default: True)
    
    Example:
        >>> scaler = AdaptiveLossScaler()
        >>> 
        >>> # Training loop
        >>> for batch in dataloader:
        ...     with autocast():
        ...         loss = model(batch)
        ...     
        ...     # Scale loss
        ...     scaled_loss = scaler.scale(loss)
        ...     scaled_loss.backward()
        ...     
        ...     # Check for overflow
        ...     overflow = scaler.check_overflow(model.parameters())
        ...     
        ...     if not overflow:
        ...         scaler.unscale_(optimizer)
        ...         optimizer.step()
        ...     
        ...     scaler.update()
    """
    
    def __init__(
        self,
        init_scale: float = 2**16,
        scale_factor: float = 2.0,
        scale_window: int = 2000,
        min_scale: float = 1.0,
        max_scale: float = 2**24,
        conservative_mode: bool = True
    ):
        self.scale = init_scale
        self.scale_factor = scale_factor
        self.scale_window = scale_window
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.conservative_mode = conservative_mode
        
        # State tracking
        self.growth_tracker = 0
        self.overflow_count = 0
        self.consecutive_no_overflow = 0
        self.scale_history: List[float] = []
        self.overflow_history: List[bool] = []
        self.spike_detected_recently = False
        self.steps_since_spike = 0
        
        logger.info(
            f"AdaptiveLossScaler initialized: scale={init_scale}, "
            f"factor={scale_factor}, window={scale_window}"
        )
    
    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """
        Scale loss before backward pass.
        
        Args:
            loss: Unscaled loss tensor
        
        Returns:
            Scaled loss tensor
        """
        return loss * self.scale
    
    def unscale_gradients(self, optimizer: torch.optim.Optimizer):
        """
        Unscale gradients after backward pass.
        
        Args:
            optimizer: PyTorch optimizer
        """
        inv_scale = 1.0 / self.scale
        
        for param_group in optimizer.param_groups:
            for param in param_group['params']:
                if param.grad is not None:
                    param.grad.data.mul_(inv_scale)
    
    def check_overflow(self, parameters) -> bool:
        """
        Check if gradients have overflowed.
        
        Args:
            parameters: Model parameters or list of tensors
        
        Returns:
            True if overflow detected
        """
        overflow = False
        
        for p in parameters:
            if p.grad is not None:
                if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                    overflow = True
                    break
        
        return overflow
    
    def update(
        self,
        overflow: Optional[bool] = None,
        spike_detected: bool = False
    ):
        """
        Update loss scale based on overflow status.
        
        Args:
            overflow: Whether overflow occurred (if None, no update)
            spike_detected: Whether gradient spike was detected
        """
        # Track spike
        if spike_detected:
            self.spike_detected_recently = True
            self.steps_since_spike = 0
        else:
            self.steps_since_spike += 1
            if self.steps_since_spike > self.scale_window:
                self.spike_detected_recently = False
        
        # Update based on overflow
        if overflow is not None:
            self.overflow_history.append(overflow)
            
            if overflow:
                # Decrease scale on overflow
                self.overflow_count += 1
                self.consecutive_no_overflow = 0
                self.growth_tracker = 0
                
                # Reduce scale
                new_scale = self.scale / self.scale_factor
                new_scale = max(new_scale, self.min_scale)
                
                logger.warning(
                    f"Overflow detected! Reducing loss scale: "
                    f"{self.scale:.1f} -> {new_scale:.1f}"
                )
                
                self.scale = new_scale
            else:
                # Increase scale after stable period
                self.consecutive_no_overflow += 1
                self.growth_tracker += 1
                
                # Determine growth window based on mode
                if self.conservative_mode and self.spike_detected_recently:
                    # Be more conservative near spikes
                    growth_window = self.scale_window * 2
                else:
                    growth_window = self.scale_window
                
                if self.growth_tracker >= growth_window:
                    # Increase scale
                    new_scale = self.scale * self.scale_factor
                    new_scale = min(new_scale, self.max_scale)
                    
                    if new_scale != self.scale:
                        logger.info(
                            f"Increasing loss scale: "
                            f"{self.scale:.1f} -> {new_scale:.1f}"
                        )
                    
                    self.scale = new_scale
                    self.growth_tracker = 0
            
            # Track history
            self.scale_history.append(self.scale)
            
            # Keep history bounded
            max_history = 10000
            if len(self.scale_history) > max_history:
                self.scale_history = self.scale_history[-max_history:]
            if len(self.overflow_history) > max_history:
                self.overflow_history = self.overflow_history[-max_history:]
    
    def get_scale(self) -> float:
        """Get current loss scale."""
        return self.scale
    
    def set_scale(self, scale: float):
        """
        Manually set loss scale.
        
        Args:
            scale: New loss scale value
        """
        scale = max(self.min_scale, min(scale, self.max_scale))
        self.scale = scale
        logger.info(f"Loss scale manually set to {scale:.1f}")
    
    def get_stats(self) -> Dict[str, any]:
        """
        Get scaler statistics.
        
        Returns:
            Dictionary with:
            - current_scale: Current loss scale
            - overflow_count: Total overflows
            - consecutive_no_overflow: Consecutive steps without overflow
            - overflow_rate: Recent overflow rate
            - spike_detected_recently: Whether spike detected recently
            - steps_since_spike: Steps since last spike
        """
        # Calculate overflow rate
        overflow_rate = 0.0
        if self.overflow_history:
            recent = self.overflow_history[-1000:]
            overflow_rate = sum(recent) / len(recent)
        
        return {
            "current_scale": self.scale,
            "overflow_count": self.overflow_count,
            "consecutive_no_overflow": self.consecutive_no_overflow,
            "overflow_rate": overflow_rate,
            "spike_detected_recently": self.spike_detected_recently,
            "steps_since_spike": self.steps_since_spike,
            "growth_tracker": self.growth_tracker,
        }
    
    def state_dict(self) -> Dict:
        """
        Get state dict for checkpointing.
        
        Returns:
            State dictionary
        """
        return {
            "scale": self.scale,
            "growth_tracker": self.growth_tracker,
            "overflow_count": self.overflow_count,
            "consecutive_no_overflow": self.consecutive_no_overflow,
            "spike_detected_recently": self.spike_detected_recently,
            "steps_since_spike": self.steps_since_spike,
        }
    
    def load_state_dict(self, state_dict: Dict):
        """
        Load state dict from checkpoint.
        
        Args:
            state_dict: State dictionary
        """
        self.scale = state_dict.get("scale", self.scale)
        self.growth_tracker = state_dict.get("growth_tracker", 0)
        self.overflow_count = state_dict.get("overflow_count", 0)
        self.consecutive_no_overflow = state_dict.get("consecutive_no_overflow", 0)
        self.spike_detected_recently = state_dict.get("spike_detected_recently", False)
        self.steps_since_spike = state_dict.get("steps_since_spike", 0)
        
        logger.info(f"AdaptiveLossScaler state loaded: scale={self.scale:.1f}")
    
    def reset(self):
        """Reset scaler state."""
        self.scale = 2**16
        self.growth_tracker = 0
        self.overflow_count = 0
        self.consecutive_no_overflow = 0
        self.scale_history = []
        self.overflow_history = []
        self.spike_detected_recently = False
        self.steps_since_spike = 0
        
        logger.info("AdaptiveLossScaler reset")
