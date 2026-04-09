"""
Gradient Flow Tracker for analyzing gradient propagation through the network.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class GradientFlowTracker:
    """
    Tracks gradient flow through neural network layers.
    
    Monitors:
    - Per-layer gradient norms
    - Gradient flow bottlenecks
    - Dead neurons (zero gradients)
    - Exploding gradients
    
    Args:
        track_frequency: How often to track (every N steps)
        history_size: Number of snapshots to keep
    
    Example:
        >>> tracker = GradientFlowTracker()
        >>> 
        >>> loss.backward()
        >>> flow_data = tracker.track_flow(model)
        >>> 
        >>> if flow_data['bottlenecks']:
        >>>     print(f"Bottlenecks: {flow_data['bottlenecks']}")
    """
    
    def __init__(
        self,
        track_frequency: int = 1,
        history_size: int = 100
    ):
        self.track_frequency = track_frequency
        self.history_size = history_size
        self.step_count = 0
        self.history: List[Dict] = []
        
        logger.info(f"GradientFlowTracker initialized (freq={track_frequency})")
    
    def track_flow(self, model: nn.Module) -> Dict:
        """Track gradient flow through model."""
        self.step_count += 1
        
        if self.step_count % self.track_frequency != 0:
            return {}
        
        flow_data = {
            "step": self.step_count,
            "layer_norms": {},
            "bottlenecks": [],
            "dead_layers": [],
        }
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                flow_data["layer_norms"][name] = grad_norm
                
                if grad_norm < 1e-7:
                    flow_data["dead_layers"].append(name)
                elif grad_norm > 100:
                    flow_data["bottlenecks"].append(name)
        
        self.history.append(flow_data)
        if len(self.history) > self.history_size:
            self.history.pop(0)
        
        return flow_data
    
    def get_stats(self) -> Dict:
        """Get gradient flow statistics."""
        if not self.history:
            return {}
        
        return {
            "total_snapshots": len(self.history),
            "last_snapshot": self.history[-1] if self.history else None,
        }

# Made with Bob
