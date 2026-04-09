"""
FSDP (Fully Sharded Data Parallel) Stability Guard.

Specialized monitoring for PyTorch FSDP training with parameter sharding.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any
import logging

try:
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import StateDictType
    FSDP_AVAILABLE = True
except ImportError:
    FSDP_AVAILABLE = False
    FSDP = None

from .spike_detector import DistributedSpikeDetector

logger = logging.getLogger(__name__)


class FSDPStabilityGuard:
    """
    Stability monitoring for FSDP (Fully Sharded Data Parallel) training.
    
    FSDP shards model parameters across ranks, which introduces unique challenges:
    - Parameters are sharded, so local gradient norms are incomplete
    - Need to gather gradients across ranks for accurate monitoring
    - All-reduce happens automatically during backward pass
    - Must monitor both local and global gradient statistics
    
    This guard:
    1. Monitors local gradient norms on each shard
    2. Computes global gradient norms via all-reduce
    3. Detects spikes in both local and global gradients
    4. Provides FSDP-aware checkpoint management
    
    Args:
        model: FSDP-wrapped model
        spike_threshold: Gradient norm threshold (default: 10.0)
        monitor_local_grads: Monitor local (sharded) gradients (default: True)
        monitor_global_grads: Monitor global (all-reduced) gradients (default: True)
        enable_spike_detection: Enable spike detection (default: True)
    
    Example:
        >>> from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        >>> 
        >>> # Wrap model with FSDP
        >>> model = FSDP(model)
        >>> 
        >>> # Initialize guard
        >>> guard = FSDPStabilityGuard(model)
        >>> 
        >>> # During training
        >>> loss.backward()
        >>> spike_info = guard.check_gradients()
        >>> if not spike_info['spike_detected']:
        >>>     optimizer.step()
    """
    
    def __init__(
        self,
        model: nn.Module,
        spike_threshold: float = 10.0,
        monitor_local_grads: bool = True,
        monitor_global_grads: bool = True,
        enable_spike_detection: bool = True
    ):
        if not FSDP_AVAILABLE:
            raise ImportError(
                "FSDP is not available. Please upgrade to PyTorch >= 1.12.0"
            )
        
        # Verify model is FSDP-wrapped
        if not isinstance(model, FSDP):
            raise TypeError(
                f"Model must be FSDP-wrapped. Got {type(model)}. "
                f"Wrap your model with: model = FSDP(model)"
            )
        
        self.model = model
        self.spike_threshold = spike_threshold
        self.monitor_local_grads = monitor_local_grads
        self.monitor_global_grads = monitor_global_grads
        self.enable_spike_detection = enable_spike_detection
        
        # Get distributed info from FSDP
        self.rank = self.model.rank
        self.world_size = self.model.world_size
        
        # Initialize distributed spike detector
        if enable_spike_detection:
            self.spike_detector = DistributedSpikeDetector(
                world_size=self.world_size,
                rank=self.rank,
                spike_threshold=spike_threshold
            )
        else:
            self.spike_detector = None
        
        # Statistics
        self.local_grad_history: List[float] = []
        self.global_grad_history: List[float] = []
        self.step_count = 0
        
        logger.info(
            f"FSDPStabilityGuard initialized on rank {self.rank}/{self.world_size}\n"
            f"  Monitor local grads: {monitor_local_grads}\n"
            f"  Monitor global grads: {monitor_global_grads}\n"
            f"  Spike detection: {enable_spike_detection}"
        )
    
    def compute_local_grad_norm(self) -> float:
        """
        Compute gradient norm for local (sharded) parameters.
        
        This is the norm of gradients on THIS rank only, before all-reduce.
        
        Returns:
            Local gradient norm
        """
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        return total_norm
    
    def compute_global_grad_norm(self) -> float:
        """
        Compute global gradient norm across all ranks.
        
        This requires gathering gradient norms from all ranks.
        
        Returns:
            Global gradient norm
        """
        # In FSDP, gradients are already all-reduced during backward()
        # So we can compute the norm directly
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        return total_norm
    
    def check_gradients(self) -> Dict[str, Any]:
        """
        Check gradient health in FSDP training.
        
        Returns:
            Dictionary with:
            - local_grad_norm: Gradient norm on this rank
            - global_grad_norm: Global gradient norm (if monitored)
            - spike_detected: Whether spike detected
            - spike_info: Detailed spike information (if detected)
        """
        result = {
            "local_grad_norm": None,
            "global_grad_norm": None,
            "spike_detected": False,
            "spike_info": None,
        }
        
        # Compute local gradient norm
        if self.monitor_local_grads:
            local_norm = self.compute_local_grad_norm()
            result["local_grad_norm"] = local_norm
            self.local_grad_history.append(local_norm)
        
        # Compute global gradient norm
        if self.monitor_global_grads:
            global_norm = self.compute_global_grad_norm()
            result["global_grad_norm"] = global_norm
            self.global_grad_history.append(global_norm)
        
        # Detect spikes
        if self.enable_spike_detection and self.spike_detector is not None:
            # Use global norm for spike detection
            norm_to_check = result["global_grad_norm"] or result["local_grad_norm"]
            
            if norm_to_check is not None:
                spike_info = self.spike_detector.detect_distributed_spike(
                    local_grad_norm=norm_to_check,
                    threshold=self.spike_threshold
                )
                result["spike_detected"] = spike_info["spike_detected"]
                result["spike_info"] = spike_info
        
        self.step_count += 1
        
        # Keep history bounded
        max_history = 1000
        if len(self.local_grad_history) > max_history:
            self.local_grad_history = self.local_grad_history[-max_history:]
        if len(self.global_grad_history) > max_history:
            self.global_grad_history = self.global_grad_history[-max_history:]
        
        return result
    
    def save_checkpoint(
        self,
        checkpoint_path: str,
        optimizer: Optional[torch.optim.Optimizer] = None,
        **kwargs
    ):
        """
        Save FSDP checkpoint with full state dict.
        
        Args:
            checkpoint_path: Path to save checkpoint
            optimizer: Optional optimizer to save
            **kwargs: Additional items to save in checkpoint
        """
        # Use FSDP's state dict API
        with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT):
            state_dict = self.model.state_dict()
        
        checkpoint = {
            "model": state_dict,
            "step": self.step_count,
        }
        
        if optimizer is not None:
            checkpoint["optimizer"] = optimizer.state_dict()
        
        checkpoint.update(kwargs)
        
        # Only rank 0 saves
        if self.rank == 0:
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"FSDP checkpoint saved to {checkpoint_path}")
    
    def load_checkpoint(
        self,
        checkpoint_path: str,
        optimizer: Optional[torch.optim.Optimizer] = None
    ) -> Dict[str, Any]:
        """
        Load FSDP checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint
            optimizer: Optional optimizer to load
        
        Returns:
            Checkpoint dictionary
        """
        # Load checkpoint on all ranks
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        
        # Load model state
        with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT):
            self.model.load_state_dict(checkpoint["model"])
        
        # Load optimizer state
        if optimizer is not None and "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        
        # Restore step count
        if "step" in checkpoint:
            self.step_count = checkpoint["step"]
        
        logger.info(f"Rank {self.rank}: FSDP checkpoint loaded from {checkpoint_path}")
        
        return checkpoint
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get monitoring statistics.
        
        Returns:
            Dictionary with:
            - step_count: Number of steps monitored
            - local_grad_stats: Statistics for local gradients
            - global_grad_stats: Statistics for global gradients
            - spike_stats: Spike detection statistics
        """
        stats = {
            "step_count": self.step_count,
            "rank": self.rank,
            "world_size": self.world_size,
        }
        
        if self.local_grad_history:
            stats["local_grad_stats"] = {
                "mean": sum(self.local_grad_history) / len(self.local_grad_history),
                "max": max(self.local_grad_history),
                "min": min(self.local_grad_history),
            }
        
        if self.global_grad_history:
            stats["global_grad_stats"] = {
                "mean": sum(self.global_grad_history) / len(self.global_grad_history),
                "max": max(self.global_grad_history),
                "min": min(self.global_grad_history),
            }
        
        if self.spike_detector is not None:
            stats["spike_stats"] = self.spike_detector.get_stats()
        
        return stats
    
    def reset(self):
        """Reset monitoring statistics."""
        self.local_grad_history = []
        self.global_grad_history = []
        self.step_count = 0
        
        if self.spike_detector is not None:
            self.spike_detector.reset()
        
        if self.rank == 0:
            logger.info("FSDPStabilityGuard reset")
