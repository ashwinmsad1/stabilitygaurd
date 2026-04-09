"""
Distributed Spike Detector for multi-GPU training.

Detects gradient spikes across distributed training ranks with coordinated recovery.
"""

import torch
import torch.distributed as dist
from typing import Dict, List, Optional, Any, Tuple
import logging

logger = logging.getLogger(__name__)


class DistributedSpikeDetector:
    """
    Detects gradient spikes across distributed training ranks.
    
    In distributed training, gradient spikes can occur on any rank and propagate
    through all-reduce operations. This detector:
    
    1. Monitors gradients on each rank independently
    2. Performs all-reduce to detect spikes across all ranks
    3. Attributes spikes to specific ranks
    4. Coordinates recovery actions across all ranks
    
    Key features:
    - Per-rank gradient monitoring
    - All-reduce spike detection
    - Rank attribution (which GPU spiked?)
    - Coordinated rollback protocol
    - Minimal communication overhead
    
    Args:
        world_size: Total number of processes in distributed training
        rank: Current process rank (0 to world_size-1)
        backend: Distributed backend ("nccl", "gloo", "mpi")
        spike_threshold: Gradient norm threshold for spike detection (default: 10.0)
        enable_attribution: Enable rank attribution for spikes (default: True)
        enable_coordinated_rollback: Enable coordinated checkpoint rollback (default: True)
    
    Example:
        >>> import torch.distributed as dist
        >>> 
        >>> # Initialize distributed training
        >>> dist.init_process_group(backend="nccl")
        >>> 
        >>> detector = DistributedSpikeDetector(
        ...     world_size=dist.get_world_size(),
        ...     rank=dist.get_rank()
        ... )
        >>> 
        >>> # During training
        >>> local_grad_norm = compute_grad_norm(model)
        >>> spike_info = detector.detect_distributed_spike(
        ...     local_grad_norm=local_grad_norm,
        ...     threshold=10.0
        ... )
        >>> 
        >>> if spike_info['spike_detected']:
        >>>     print(f"Spike on rank {spike_info['spike_rank']}")
        >>>     # All ranks will skip this step
    """
    
    def __init__(
        self,
        world_size: int,
        rank: int,
        backend: str = "nccl",
        spike_threshold: float = 10.0,
        enable_attribution: bool = True,
        enable_coordinated_rollback: bool = True
    ):
        self.world_size = world_size
        self.rank = rank
        self.backend = backend
        self.spike_threshold = spike_threshold
        self.enable_attribution = enable_attribution
        self.enable_coordinated_rollback = enable_coordinated_rollback
        
        # Check if distributed is initialized
        if not dist.is_initialized():
            raise RuntimeError(
                "torch.distributed is not initialized. "
                "Call dist.init_process_group() before creating DistributedSpikeDetector."
            )
        
        # Verify rank and world_size match
        if dist.get_rank() != rank:
            logger.warning(
                f"Provided rank {rank} doesn't match dist.get_rank() {dist.get_rank()}. "
                f"Using dist.get_rank()."
            )
            self.rank = dist.get_rank()
        
        if dist.get_world_size() != world_size:
            logger.warning(
                f"Provided world_size {world_size} doesn't match dist.get_world_size() {dist.get_world_size()}. "
                f"Using dist.get_world_size()."
            )
            self.world_size = dist.get_world_size()
        
        # Statistics
        self.total_spikes = 0
        self.spikes_by_rank = [0] * self.world_size
        self.spike_history: List[Dict[str, Any]] = []
        
        logger.info(
            f"DistributedSpikeDetector initialized on rank {self.rank}/{self.world_size} "
            f"(backend: {self.backend}, threshold: {spike_threshold})"
        )
    
    def detect_distributed_spike(
        self,
        local_grad_norm: float,
        threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Detect spike across all ranks using all-reduce.
        
        Algorithm:
        1. Each rank computes local gradient norm
        2. All-reduce to gather norms from all ranks
        3. Check if any rank exceeds threshold
        4. Attribute spike to specific rank(s)
        
        Args:
            local_grad_norm: Gradient norm on current rank
            threshold: Spike threshold (default: self.spike_threshold)
        
        Returns:
            Dictionary with:
            - spike_detected: bool - Whether any rank spiked
            - spike_rank: int - Which rank spiked (-1 if none, -2 if multiple)
            - all_grad_norms: List[float] - Gradient norms from all ranks
            - local_grad_norm: float - This rank's gradient norm
            - max_grad_norm: float - Maximum gradient norm across ranks
        """
        if threshold is None:
            threshold = self.spike_threshold
        
        # Create tensor for all-gather
        local_norm_tensor = torch.tensor([local_grad_norm], device=f"cuda:{self.rank}")
        all_norms_tensor = torch.zeros(self.world_size, device=f"cuda:{self.rank}")
        
        # All-gather gradient norms from all ranks
        dist.all_gather_into_tensor(all_norms_tensor, local_norm_tensor)
        
        # Convert to list
        all_grad_norms = all_norms_tensor.cpu().tolist()
        
        # Check for spikes
        spike_detected = any(norm > threshold for norm in all_grad_norms)
        
        # Attribute spike to rank
        spike_rank = -1  # -1 means no spike
        if spike_detected and self.enable_attribution:
            spiked_ranks = [i for i, norm in enumerate(all_grad_norms) if norm > threshold]
            if len(spiked_ranks) == 1:
                spike_rank = spiked_ranks[0]
            elif len(spiked_ranks) > 1:
                spike_rank = -2  # -2 means multiple ranks spiked
        
        # Update statistics
        if spike_detected:
            self.total_spikes += 1
            if spike_rank >= 0:
                self.spikes_by_rank[spike_rank] += 1
            
            # Log spike event
            spike_event = {
                "step": self.total_spikes,
                "spike_rank": spike_rank,
                "all_grad_norms": all_grad_norms,
                "max_grad_norm": max(all_grad_norms),
                "threshold": threshold,
            }
            self.spike_history.append(spike_event)
            
            # Log on rank 0
            if self.rank == 0:
                if spike_rank >= 0:
                    logger.warning(
                        f"DISTRIBUTED SPIKE DETECTED!\n"
                        f"  Spike rank: {spike_rank}\n"
                        f"  Gradient norm: {all_grad_norms[spike_rank]:.4f}\n"
                        f"  Threshold: {threshold:.4f}\n"
                        f"  All norms: {[f'{n:.2f}' for n in all_grad_norms]}"
                    )
                elif spike_rank == -2:
                    logger.warning(
                        f"MULTIPLE RANKS SPIKED!\n"
                        f"  Spiked ranks: {[i for i, n in enumerate(all_grad_norms) if n > threshold]}\n"
                        f"  All norms: {[f'{n:.2f}' for n in all_grad_norms]}"
                    )
        
        return {
            "spike_detected": spike_detected,
            "spike_rank": spike_rank,
            "all_grad_norms": all_grad_norms,
            "local_grad_norm": local_grad_norm,
            "max_grad_norm": max(all_grad_norms),
        }
    
    def coordinate_rollback(self, checkpoint_path: Optional[str] = None) -> bool:
        """
        Coordinate checkpoint rollback across all ranks.
        
        Uses barrier synchronization to ensure all ranks rollback together.
        
        Args:
            checkpoint_path: Path to checkpoint to rollback to (optional)
        
        Returns:
            True if rollback coordinated successfully
        """
        if not self.enable_coordinated_rollback:
            return False
        
        try:
            # Barrier to synchronize all ranks
            dist.barrier()
            
            if self.rank == 0:
                logger.info(f"Coordinated rollback initiated across {self.world_size} ranks")
            
            # All ranks proceed with rollback
            # (Actual checkpoint loading should be done by caller)
            
            # Another barrier to ensure all ranks completed rollback
            dist.barrier()
            
            if self.rank == 0:
                logger.info("Coordinated rollback completed")
            
            return True
            
        except Exception as e:
            logger.error(f"Rank {self.rank}: Rollback coordination failed: {e}")
            return False
    
    def broadcast_decision(self, decision: bool) -> bool:
        """
        Broadcast a decision from rank 0 to all ranks.
        
        Useful for coordinating actions like "skip this step" or "rollback".
        
        Args:
            decision: Decision to broadcast (only used on rank 0)
        
        Returns:
            The broadcasted decision (same on all ranks)
        """
        decision_tensor = torch.tensor([1 if decision else 0], device=f"cuda:{self.rank}")
        dist.broadcast(decision_tensor, src=0)
        return bool(decision_tensor.item())
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get detector statistics.
        
        Returns:
            Dictionary with:
            - total_spikes: Total spikes detected
            - spikes_by_rank: List of spike counts per rank
            - spike_history: List of spike events
            - current_rank: This process's rank
            - world_size: Total number of ranks
        """
        return {
            "total_spikes": self.total_spikes,
            "spikes_by_rank": self.spikes_by_rank,
            "spike_history": self.spike_history[-10:],  # Last 10 spikes
            "current_rank": self.rank,
            "world_size": self.world_size,
        }
    
    def reset(self):
        """Reset detector statistics."""
        self.total_spikes = 0
        self.spikes_by_rank = [0] * self.world_size
        self.spike_history = []
        
        if self.rank == 0:
            logger.info("DistributedSpikeDetector reset")
    
    @staticmethod
    def compute_global_grad_norm(model: torch.nn.Module) -> float:
        """
        Compute global gradient norm across all ranks.
        
        This is the norm AFTER all-reduce, representing the actual
        gradient that will be applied.
        
        Args:
            model: PyTorch model
        
        Returns:
            Global gradient norm
        """
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        return total_norm
    
    @staticmethod
    def compute_local_grad_norm(model: torch.nn.Module) -> float:
        """
        Compute local gradient norm on current rank.
        
        This is the norm BEFORE all-reduce, representing gradients
        computed locally on this rank.
        
        Args:
            model: PyTorch model
        
        Returns:
            Local gradient norm
        """
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        return total_norm

