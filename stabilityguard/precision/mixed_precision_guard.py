"""
Mixed Precision Guard - Unified interface for mixed precision stability monitoring.

Combines PrecisionMonitor and AdaptiveLossScaler for complete mixed precision support.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Any, List
import logging

from .precision_monitor import PrecisionMonitor
from .loss_scaler import AdaptiveLossScaler

logger = logging.getLogger(__name__)


class MixedPrecisionGuard:
    """
    Unified guard for mixed precision training stability.
    
    Combines precision monitoring and adaptive loss scaling to provide
    comprehensive stability for FP16/BF16/FP8 training.
    
    Features:
    - Automatic overflow/underflow detection
    - Adaptive loss scaling
    - Precision recommendations
    - Integration with PyTorch AMP
    
    Args:
        model: PyTorch model
        precision: Precision format ("fp16", "bf16", "fp8", "fp32")
        enable_adaptive_scaling: Enable adaptive loss scaling (default: True)
        enable_overflow_detection: Enable overflow detection (default: True)
        enable_underflow_detection: Enable underflow detection (default: True)
        init_scale: Initial loss scale (default: 2^16)
        conservative_mode: Use conservative scaling (default: True)
    
    Example:
        >>> from torch.cuda.amp import autocast
        >>> 
        >>> guard = MixedPrecisionGuard(model, precision="fp16")
        >>> 
        >>> for batch in dataloader:
        ...     with autocast():
        ...         loss = model(batch)
        ...     
        ...     # Check stability and scale loss
        ...     stability_report = guard.check_stability(model)
        ...     
        ...     if stability_report['recommend_bf16']:
        ...         print("SUGGESTION: Consider switching to BF16")
        ...
        ...     # Backward with scaled loss
        ...     scaled_loss = guard.scale_loss(loss)
        ...     scaled_loss.backward()
        ...     
        ...     # Unscale and step
        ...     if not stability_report['overflow_detected']:
        ...         guard.unscale_gradients(optimizer)
        ...         optimizer.step()
        ...     
        ...     guard.update()
    """
    
    def __init__(
        self,
        model: nn.Module,
        precision: str = "fp16",
        enable_adaptive_scaling: bool = True,
        enable_overflow_detection: bool = True,
        enable_underflow_detection: bool = True,
        init_scale: float = 2**16,
        conservative_mode: bool = True
    ):
        self.model = model
        self.precision = precision
        self.enable_adaptive_scaling = enable_adaptive_scaling
        self.enable_overflow_detection = enable_overflow_detection
        self.enable_underflow_detection = enable_underflow_detection
        
        # Initialize components
        self.precision_monitor = PrecisionMonitor(
            precision=precision,
            loss_scale_init=init_scale
        )
        
        if enable_adaptive_scaling:
            self.loss_scaler = AdaptiveLossScaler(
                init_scale=init_scale,
                conservative_mode=conservative_mode
            )
        else:
            self.loss_scaler = None
        
        logger.info(
            f"MixedPrecisionGuard initialized: {precision}\n"
            f"  Adaptive scaling: {enable_adaptive_scaling}\n"
            f"  Overflow detection: {enable_overflow_detection}\n"
            f"  Underflow detection: {enable_underflow_detection}"
        )
    
    def get_gradients(self) -> List[torch.Tensor]:
        """Get all gradients from model."""
        gradients = []
        for p in self.model.parameters():
            if p.grad is not None:
                gradients.append(p.grad)
        return gradients
    
    def check_stability(
        self,
        model: Optional[nn.Module] = None
    ) -> Dict[str, Any]:
        """
        Check mixed precision training stability.
        
        Args:
            model: Model to check (default: self.model)
        
        Returns:
            Stability report with:
            - overflow_detected: Whether overflow detected
            - underflow_detected: Whether underflow detected
            - recommend_bf16: Whether to switch to BF16
            - gradient_range: (min, max) gradient values
            - precision_stats: Precision monitor statistics
            - scaler_stats: Loss scaler statistics (if enabled)
        """
        if model is None:
            model = self.model
        
        gradients = self.get_gradients()
        
        report = {
            "overflow_detected": False,
            "underflow_detected": False,
            "recommend_bf16": False,
            "gradient_range": (0.0, 0.0),
            "precision_stats": {},
            "scaler_stats": {},
        }
        
        if not gradients:
            return report
        
        # Check overflow
        if self.enable_overflow_detection:
            overflow = self.precision_monitor.check_overflow(gradients)
            report["overflow_detected"] = overflow
        
        # Check underflow
        if self.enable_underflow_detection:
            underflow = self.precision_monitor.check_underflow(gradients)
            report["underflow_detected"] = underflow
        
        # Get gradient range
        grad_range = self.precision_monitor.get_gradient_range(gradients)
        report["gradient_range"] = grad_range
        
        # Check if should switch to BF16
        report["recommend_bf16"] = self.precision_monitor.should_switch_to_bf16()
        
        # Get statistics
        report["precision_stats"] = self.precision_monitor.get_stats()
        
        if self.loss_scaler is not None:
            report["scaler_stats"] = self.loss_scaler.get_stats()
        
        return report
    
    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """
        Scale loss before backward pass.
        
        Args:
            loss: Unscaled loss
        
        Returns:
            Scaled loss
        """
        if self.loss_scaler is not None:
            return self.loss_scaler.scale_loss(loss)
        return loss
    
    def unscale_gradients(self, optimizer: torch.optim.Optimizer):
        """
        Unscale gradients after backward pass.
        
        Args:
            optimizer: PyTorch optimizer
        """
        if self.loss_scaler is not None:
            self.loss_scaler.unscale_gradients(optimizer)
    
    def update(self, spike_detected: bool = False):
        """
        Update guard state after optimizer step.
        
        Args:
            spike_detected: Whether gradient spike was detected
        """
        # Check for overflow
        gradients = self.get_gradients()
        overflow = False
        if gradients and self.loss_scaler is not None:
            overflow = self.loss_scaler.check_overflow(gradients)
        
        # Update loss scaler
        if self.loss_scaler is not None:
            self.loss_scaler.update(
                overflow=overflow,
                spike_detected=spike_detected
            )
        
        # Update precision monitor
        if overflow:
            new_scale = self.loss_scaler.get_scale() if self.loss_scaler else None
            if new_scale is not None:
                self.precision_monitor.update_loss_scale(new_scale)
        
        # Increment step count
        self.precision_monitor.step_count += 1
    
    def get_loss_scale(self) -> float:
        """Get current loss scale."""
        if self.loss_scaler is not None:
            return self.loss_scaler.get_scale()
        return 1.0
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics from all components.
        
        Returns:
            Dictionary with all statistics
        """
        stats = {
            "precision": self.precision,
            "precision_monitor": self.precision_monitor.get_stats(),
        }
        
        if self.loss_scaler is not None:
            stats["loss_scaler"] = self.loss_scaler.get_stats()
        
        return stats
    
    def state_dict(self) -> Dict:
        """Get state dict for checkpointing."""
        state = {
            "precision": self.precision,
            "precision_monitor": self.precision_monitor.get_stats(),
        }
        
        if self.loss_scaler is not None:
            state["loss_scaler"] = self.loss_scaler.state_dict()
        
        return state
    
    def load_state_dict(self, state_dict: Dict):
        """Load state dict from checkpoint."""
        if "loss_scaler" in state_dict and self.loss_scaler is not None:
            self.loss_scaler.load_state_dict(state_dict["loss_scaler"])
        
        logger.info("MixedPrecisionGuard state loaded")
    
    def reset(self):
        """Reset all components."""
        self.precision_monitor.reset()
        
        if self.loss_scaler is not None:
            self.loss_scaler.reset()
        
        logger.info("MixedPrecisionGuard reset")

