"""
Checkpoint Health Scorer for evaluating checkpoint quality.
"""

import torch
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


class CheckpointHealthScorer:
    """
    Scores checkpoint quality for recovery decisions.
    
    Metrics:
    - Gradient norm stability
    - Loss trajectory
    - Training time since last spike
    
    Args:
        history_size: Number of checkpoints to track
    
    Example:
        >>> scorer = CheckpointHealthScorer()
        >>> score = scorer.score_checkpoint("checkpoint.pt", history)
    """
    
    def __init__(self, history_size: int = 10):
        self.history_size = history_size
        self.checkpoint_history: List[Dict] = []
        
        logger.info("CheckpointHealthScorer initialized")
    
    def score_checkpoint(
        self,
        checkpoint_path: str,
        history: List[Dict]
    ) -> float:
        """
        Score checkpoint health (0-100).
        
        Args:
            checkpoint_path: Path to checkpoint
            history: Training history
        
        Returns:
            Health score (0-100)
        """
        if not history:
            return 50.0
        
        score = 100.0
        
        # Check gradient stability
        recent_grads = [h.get("grad_norm", 0) for h in history[-10:]]
        if recent_grads:
            grad_std = torch.tensor(recent_grads).std().item()
            if grad_std > 10:
                score -= 20
        
        # Check loss trajectory
        recent_losses = [h.get("loss", 0) for h in history[-10:]]
        if len(recent_losses) > 1:
            if recent_losses[-1] > recent_losses[0]:
                score -= 15
        
        # Check spike history
        recent_spikes = sum(1 for h in history[-10:] if h.get("spike_detected", False))
        score -= recent_spikes * 10
        
        return max(0.0, min(100.0, score))
    
    def get_stats(self) -> Dict:
        """Get statistics."""
        return {
            "total_checkpoints": len(self.checkpoint_history),
        }
