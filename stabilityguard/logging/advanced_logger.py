"""
Advanced Logger - Unified interface for all logging components.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional
import logging

from .gradient_flow import GradientFlowTracker
from .activation_stats import ActivationStatsLogger
from .weight_updates import WeightUpdateTracker
from .checkpoint_scorer import CheckpointHealthScorer

logger = logging.getLogger(__name__)


class AdvancedLogger:
    """
    Unified interface for comprehensive training logging.
    
    Combines:
    - Gradient flow tracking
    - Activation statistics
    - Weight update monitoring
    - Checkpoint health scoring
    
    Args:
        log_dir: Directory to save logs
        enable_gradient_flow: Enable gradient flow tracking
        enable_activation_stats: Enable activation statistics
        enable_weight_updates: Enable weight update tracking
        log_frequency: How often to log
    
    Example:
        >>> logger = AdvancedLogger(log_dir="./logs")
        >>> 
        >>> # Training loop
        >>> loss.backward()
        >>> logger.log_step(step, loss.item(), model, optimizer)
        >>> optimizer.step()
    """
    
    def __init__(
        self,
        log_dir: str = "./sg_logs",
        enable_gradient_flow: bool = True,
        enable_activation_stats: bool = False,
        enable_weight_updates: bool = True,
        log_frequency: int = 100
    ):
        self.log_dir = log_dir
        self.log_frequency = log_frequency
        self.step_count = 0
        
        # Initialize components
        self.gradient_flow = GradientFlowTracker() if enable_gradient_flow else None
        self.activation_stats = ActivationStatsLogger() if enable_activation_stats else None
        self.weight_updates = WeightUpdateTracker() if enable_weight_updates else None
        self.checkpoint_scorer = CheckpointHealthScorer()
        
        logger.info(f"AdvancedLogger initialized: {log_dir}")
    
    def log_step(
        self,
        step: int,
        loss: float,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None
    ) -> Dict:
        """
        Log comprehensive statistics for a training step.
        
        Args:
            step: Current training step
            loss: Current loss value
            model: PyTorch model
            optimizer: PyTorch optimizer
        
        Returns:
            Dictionary with all logged statistics
        """
        self.step_count += 1
        
        log_data = {
            "step": step,
            "loss": loss,
        }
        
        # Gradient flow
        if self.gradient_flow is not None:
            flow_data = self.gradient_flow.track_flow(model)
            if flow_data:
                log_data["gradient_flow"] = flow_data
        
        # Activation stats
        if self.activation_stats is not None:
            act_data = self.activation_stats.log_activations(model)
            if act_data:
                log_data["activation_stats"] = act_data
        
        # Weight updates
        if self.weight_updates is not None:
            update_data = self.weight_updates.track_updates(model)
            if update_data:
                log_data["weight_updates"] = update_data
        
        return log_data
    
    def get_comprehensive_stats(self) -> Dict:
        """Get statistics from all components."""
        stats = {}
        
        if self.gradient_flow is not None:
            stats["gradient_flow"] = self.gradient_flow.get_stats()
        
        if self.activation_stats is not None:
            stats["activation_stats"] = self.activation_stats.get_stats()
        
        if self.weight_updates is not None:
            stats["weight_updates"] = self.weight_updates.get_stats()
        
        stats["checkpoint_scorer"] = self.checkpoint_scorer.get_stats()
        
        return stats

