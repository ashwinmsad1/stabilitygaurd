"""
Distributed Guarded Optimizer.

Wraps any optimizer with distributed spike detection and coordinated recovery.
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from typing import Dict, Optional, Any
import logging

from ..core.guarded_optimizer import GuardedOptimizer
from .spike_detector import DistributedSpikeDetector

logger = logging.getLogger(__name__)


class DistributedGuardedOptimizer(GuardedOptimizer):
    """
    Distributed version of GuardedOptimizer with rank-aware spike detection.
    
    Extends GuardedOptimizer to handle distributed training:
    - Detects spikes across all ranks using all-reduce
    - Coordinates recovery actions across all ranks
    - Attributes spikes to specific ranks
    - Supports DDP, FSDP, and DeepSpeed
    
    All ranks must call step() together, even if a spike is detected.
    The optimizer will skip the update on all ranks if any rank detects a spike.
    
    Args:
        base_optimizer: Base PyTorch optimizer
        model: PyTorch model
        world_size: Total number of processes
        rank: Current process rank
        spike_threshold: Gradient norm threshold (default: 10.0)
        nan_action: Action on NaN detection ("skip", "rollback", "raise")
        enable_coordinated_rollback: Enable coordinated checkpoint rollback (default: True)
        **kwargs: Additional arguments passed to GuardedOptimizer
    
    Example:
        >>> import torch.distributed as dist
        >>> from torch.nn.parallel import DistributedDataParallel as DDP
        >>> 
        >>> # Initialize distributed training
        >>> dist.init_process_group(backend="nccl")
        >>> rank = dist.get_rank()
        >>> world_size = dist.get_world_size()
        >>> 
        >>> # Wrap model with DDP
        >>> model = DDP(model, device_ids=[rank])
        >>> 
        >>> # Create distributed guarded optimizer
        >>> optimizer = DistributedGuardedOptimizer(
        ...     base_optimizer=torch.optim.AdamW(model.parameters(), lr=1e-4),
        ...     model=model,
        ...     world_size=world_size,
        ...     rank=rank,
        ...     spike_threshold=10.0
        ... )
        >>> 
        >>> # Training loop
        >>> for batch in dataloader:
        ...     loss = model(batch)
        ...     loss.backward()
        ...     
        ...     # All ranks call step together
        ...     optimizer.step()
        ...     optimizer.zero_grad()
    """
    
    def __init__(
        self,
        base_optimizer: torch.optim.Optimizer,
        model: nn.Module,
        world_size: int,
        rank: int,
        spike_threshold: float = 10.0,
        nan_action: str = "skip",
        enable_coordinated_rollback: bool = True,
        **kwargs
    ):
        # Initialize base GuardedOptimizer
        super().__init__(
            base_optimizer=base_optimizer,
            model=model,
            spike_threshold=spike_threshold,
            nan_action=nan_action,
            **kwargs
        )
        
        self.world_size = world_size
        self.rank = rank
        self.enable_coordinated_rollback = enable_coordinated_rollback
        
        # Check if distributed is initialized
        if not dist.is_initialized():
            raise RuntimeError(
                "torch.distributed is not initialized. "
                "Call dist.init_process_group() before creating DistributedGuardedOptimizer."
            )
        
        # Replace spike detector with distributed version
        self._distributed_spike_detector = DistributedSpikeDetector(
            world_size=world_size,
            rank=rank,
            spike_threshold=spike_threshold,
            enable_coordinated_rollback=enable_coordinated_rollback
        )
        
        logger.info(
            f"DistributedGuardedOptimizer initialized on rank {rank}/{world_size}"
        )
    
    def _compute_grad_norm(self) -> float:
        """
        Compute gradient norm with distributed awareness.
        
        For DDP, gradients are already all-reduced, so we can compute directly.
        For FSDP/DeepSpeed, use their specific methods.
        
        Returns:
            Global gradient norm
        """
        # Use the distributed spike detector's method
        return DistributedSpikeDetector.compute_global_grad_norm(self.model)
    
    def _detect_spike(self, grad_norm: float) -> bool:
        """
        Detect spike across all ranks using distributed spike detector.
        
        Args:
            grad_norm: Local gradient norm
        
        Returns:
            True if spike detected on any rank
        """
        spike_info = self._distributed_spike_detector.detect_distributed_spike(
            local_grad_norm=grad_norm,
            threshold=self.spike_threshold
        )
        
        # Log spike information on rank 0
        if spike_info["spike_detected"] and self.rank == 0:
            spike_rank = spike_info["spike_rank"]
            if spike_rank >= 0:
                logger.warning(
                    f"DISTRIBUTED SPIKE: Rank {spike_rank} spiked "
                    f"(norm: {spike_info['all_grad_norms'][spike_rank]:.4f})"
                )
            elif spike_rank == -2:
                logger.warning(
                    f"MULTIPLE RANKS SPIKED: "
                    f"{[i for i, n in enumerate(spike_info['all_grad_norms']) if n > self.spike_threshold]}"
                )
        
        return spike_info["spike_detected"]
    
    def step(self, closure=None, **kwargs):
        """
        Perform optimizer step with distributed spike detection.
        
        All ranks must call this method together. If any rank detects a spike,
        all ranks will skip the update.
        
        Args:
            closure: Optional closure to re-evaluate the model
            **kwargs: Additional arguments
        
        Returns:
            Dictionary with step information
        """
        # Compute gradient norm
        grad_norm = self._compute_grad_norm()
        
        # Detect spike across all ranks
        spike_detected = self._detect_spike(grad_norm)
        
        # Detect NaN/Inf
        nan_detected = self._detect_nan()
        
        # Coordinate action across all ranks
        should_skip = spike_detected or nan_detected
        
        # Broadcast decision from rank 0 to ensure consistency
        should_skip_tensor = torch.tensor([1 if should_skip else 0], device=f"cuda:{self.rank}")
        dist.broadcast(should_skip_tensor, src=0)
        should_skip = bool(should_skip_tensor.item())
        
        # Prepare result
        result = {
            "step_taken": False,
            "grad_norm": grad_norm,
            "spike_detected": spike_detected,
            "nan_detected": nan_detected,
            "rank": self.rank,
        }
        
        if should_skip:
            # Skip update on all ranks
            if self.rank == 0:
                logger.warning(
                    f"Skipping optimizer step on all ranks "
                    f"(spike: {spike_detected}, nan: {nan_detected})"
                )
            
            # Handle action
            if spike_detected:
                self._handle_spike_action()
            if nan_detected:
                self._handle_nan_action()
            
            result["action_taken"] = "skip"
        else:
            # Perform update on all ranks
            if closure is not None:
                loss = self.base_optimizer.step(closure)
            else:
                self.base_optimizer.step()
            
            result["step_taken"] = True
            self.steps_taken += 1
        
        self.total_steps += 1
        
        return result
    
    def coordinate_rollback(self, checkpoint_path: Optional[str] = None) -> bool:
        """
        Coordinate checkpoint rollback across all ranks.
        
        Args:
            checkpoint_path: Path to checkpoint (optional)
        
        Returns:
            True if rollback successful
        """
        return self._distributed_spike_detector.coordinate_rollback(checkpoint_path)
    
    def get_distributed_stats(self) -> Dict[str, Any]:
        """
        Get distributed training statistics.
        
        Returns:
            Dictionary with:
            - rank: Current rank
            - world_size: Total number of ranks
            - spike_stats: Spike statistics from distributed detector
            - optimizer_stats: Standard optimizer statistics
        """
        stats = {
            "rank": self.rank,
            "world_size": self.world_size,
            "spike_stats": self._distributed_spike_detector.get_stats(),
            "optimizer_stats": self.get_stats(),
        }
        
        return stats
    
    def reset_distributed_stats(self):
        """Reset distributed statistics."""
        self._distributed_spike_detector.reset()
        
        if self.rank == 0:
            logger.info("Distributed statistics reset")
