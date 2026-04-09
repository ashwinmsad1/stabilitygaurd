"""
Weight Update Tracker for monitoring parameter changes.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class WeightUpdateTracker:
    """
    Tracks magnitude of weight updates.
    
    Useful for:
    - Detecting learning rate issues
    - Finding frozen layers
    - Monitoring convergence
    
    Args:
        track_frequency: How often to track
    
    Example:
        >>> tracker = WeightUpdateTracker()
        >>> prev_state = {n: p.clone() for n, p in model.named_parameters()}
        >>> optimizer.step()
        >>> updates = tracker.track_updates(model, prev_state)
    """
    
    def __init__(self, track_frequency: int = 1):
        self.track_frequency = track_frequency
        self.step_count = 0
        self.prev_state: Optional[Dict[str, torch.Tensor]] = None
        
        logger.info("WeightUpdateTracker initialized")
    
    def track_updates(
        self,
        model: nn.Module,
        prev_state: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, float]:
        """Compute weight update magnitudes."""
        self.step_count += 1
        
        if self.step_count % self.track_frequency != 0:
            return {}
        
        if prev_state is None:
            prev_state = self.prev_state
        
        if prev_state is None:
            self.prev_state = {n: p.data.clone() for n, p in model.named_parameters()}
            return {}
        
        updates = {}
        for name, param in model.named_parameters():
            if name in prev_state:
                update_norm = (param.data - prev_state[name]).norm().item()
                updates[name] = update_norm
        
        self.prev_state = {n: p.data.clone() for n, p in model.named_parameters()}
        return updates
    
    def get_stats(self) -> Dict:
        """Get statistics."""
        return {"step_count": self.step_count}
