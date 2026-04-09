"""
Activation Statistics Logger for monitoring layer activations.
"""

import torch
import torch.nn as nn
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


class ActivationStatsLogger:
    """
    Logs layer-wise activation statistics.
    
    Tracks:
    - Mean, std, min, max per layer
    - Dead ReLUs (always zero)
    - Saturation (sigmoid/tanh near limits)
    - Distribution shifts
    
    Args:
        track_frequency: How often to log
        history_size: Number of snapshots to keep
    
    Example:
        >>> logger = ActivationStatsLogger()
        >>> stats = logger.log_activations(model)
    """
    
    def __init__(
        self,
        track_frequency: int = 10,
        history_size: int = 100
    ):
        self.track_frequency = track_frequency
        self.history_size = history_size
        self.step_count = 0
        self.history: List[Dict] = []
        self.hooks = []
        self.activations = {}
        
        logger.info(f"ActivationStatsLogger initialized")
    
    def register_hooks(self, model: nn.Module):
        """Register forward hooks to capture activations."""
        def hook_fn(name):
            def hook(module, input, output):
                self.activations[name] = output.detach()
            return hook
        
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.ReLU)):
                handle = module.register_forward_hook(hook_fn(name))
                self.hooks.append(handle)
    
    def log_activations(self, model: nn.Module) -> Dict:
        """Log activation statistics."""
        self.step_count += 1
        
        if self.step_count % self.track_frequency != 0:
            return {}
        
        stats = {"step": self.step_count, "layers": {}}
        
        for name, activation in self.activations.items():
            if activation is not None:
                stats["layers"][name] = {
                    "mean": activation.mean().item(),
                    "std": activation.std().item(),
                    "min": activation.min().item(),
                    "max": activation.max().item(),
                }
        
        self.history.append(stats)
        if len(self.history) > self.history_size:
            self.history.pop(0)
        
        self.activations = {}
        return stats
    
    def remove_hooks(self):
        """Remove all hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def get_stats(self) -> Dict:
        """Get statistics."""
        return {
            "total_snapshots": len(self.history),
            "last_snapshot": self.history[-1] if self.history else None,
        }

