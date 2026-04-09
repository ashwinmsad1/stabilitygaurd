"""
DeepSpeed Stability Guard.

Specialized monitoring for DeepSpeed ZeRO training with optimizer state partitioning.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any
import logging

try:
    import deepspeed
    from deepspeed import DeepSpeedEngine
    DEEPSPEED_AVAILABLE = True
except ImportError:
    DEEPSPEED_AVAILABLE = False
    DeepSpeedEngine = None

from .spike_detector import DistributedSpikeDetector

logger = logging.getLogger(__name__)


class DeepSpeedStabilityGuard:
    """
    Stability monitoring for DeepSpeed ZeRO training.
    
    DeepSpeed ZeRO partitions optimizer states and optionally parameters/gradients:
    - ZeRO-1: Optimizer state partitioning
    - ZeRO-2: + Gradient partitioning
    - ZeRO-3: + Parameter partitioning
    
    This guard handles:
    1. Gradient monitoring across ZeRO stages
    2. Spike detection with ZeRO-aware norm computation
    3. Pipeline parallelism support
    4. Gradient accumulation awareness
    5. DeepSpeed checkpoint management
    
    Args:
        model_engine: DeepSpeed model engine
        spike_threshold: Gradient norm threshold (default: 10.0)
        monitor_gradients: Monitor gradient norms (default: True)
        enable_spike_detection: Enable spike detection (default: True)
        zero_stage: ZeRO optimization stage (1, 2, or 3)
    
    Example:
        >>> import deepspeed
        >>> 
        >>> # Initialize DeepSpeed
        >>> model_engine, optimizer, _, _ = deepspeed.initialize(
        ...     model=model,
        ...     config=ds_config
        ... )
        >>> 
        >>> # Initialize guard
        >>> guard = DeepSpeedStabilityGuard(model_engine)
        >>> 
        >>> # Training loop
        >>> for batch in dataloader:
        ...     loss = model_engine(batch)
        ...     model_engine.backward(loss)
        ...     
        ...     spike_info = guard.check_gradients()
        ...     if not spike_info['spike_detected']:
        ...         model_engine.step()
    """
    
    def __init__(
        self,
        model_engine: Any,  # DeepSpeedEngine
        spike_threshold: float = 10.0,
        monitor_gradients: bool = True,
        enable_spike_detection: bool = True,
        zero_stage: Optional[int] = None
    ):
        if not DEEPSPEED_AVAILABLE:
            raise ImportError(
                "DeepSpeed is not available. Install with: pip install deepspeed"
            )
        
        # Verify model is DeepSpeed engine
        if DeepSpeedEngine is not None and not isinstance(model_engine, DeepSpeedEngine):
            raise TypeError(
                f"Model must be DeepSpeedEngine. Got {type(model_engine)}. "
                f"Initialize with: model_engine, _, _, _ = deepspeed.initialize(...)"
            )
        
        self.model_engine = model_engine
        self.spike_threshold = spike_threshold
        self.monitor_gradients = monitor_gradients
        self.enable_spike_detection = enable_spike_detection
        
        # Get distributed info
        self.rank = self.model_engine.local_rank
        self.world_size = self.model_engine.world_size
        
        # Detect ZeRO stage from config
        if zero_stage is None:
            try:
                self.zero_stage = self.model_engine.zero_optimization_stage()
            except:
                self.zero_stage = 0
                logger.warning("Could not detect ZeRO stage, assuming 0 (disabled)")
        else:
            self.zero_stage = zero_stage
        
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
        self.grad_norm_history: List[float] = []
        self.step_count = 0
        
        logger.info(
            f"DeepSpeedStabilityGuard initialized on rank {self.rank}/{self.world_size}\n"
            f"  ZeRO stage: {self.zero_stage}\n"
            f"  Monitor gradients: {monitor_gradients}\n"
            f"  Spike detection: {enable_spike_detection}"
        )
    
    def compute_grad_norm(self) -> float:
        """
        Compute gradient norm with ZeRO-aware handling.
        
        For ZeRO-2/3, gradients are partitioned across ranks.
        DeepSpeed provides get_global_grad_norm() for this.
        
        Returns:
            Global gradient norm
        """
        # DeepSpeed provides a built-in method for this
        if hasattr(self.model_engine, 'get_global_grad_norm'):
            return self.model_engine.get_global_grad_norm()
        
        # Fallback: compute manually
        total_norm = 0.0
        for p in self.model_engine.module.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        return total_norm
    
    def check_gradients(self) -> Dict[str, Any]:
        """
        Check gradient health in DeepSpeed training.
        
        Returns:
            Dictionary with:
            - grad_norm: Global gradient norm
            - spike_detected: Whether spike detected
            - spike_info: Detailed spike information (if detected)
            - zero_stage: ZeRO optimization stage
        """
        result = {
            "grad_norm": None,
            "spike_detected": False,
            "spike_info": None,
            "zero_stage": self.zero_stage,
        }
        
        # Compute gradient norm
        if self.monitor_gradients:
            grad_norm = self.compute_grad_norm()
            result["grad_norm"] = grad_norm
            self.grad_norm_history.append(grad_norm)
        
        # Detect spikes
        if self.enable_spike_detection and self.spike_detector is not None:
            if result["grad_norm"] is not None:
                spike_info = self.spike_detector.detect_distributed_spike(
                    local_grad_norm=result["grad_norm"],
                    threshold=self.spike_threshold
                )
                result["spike_detected"] = spike_info["spike_detected"]
                result["spike_info"] = spike_info
        
        self.step_count += 1
        
        # Keep history bounded
        max_history = 1000
        if len(self.grad_norm_history) > max_history:
            self.grad_norm_history = self.grad_norm_history[-max_history:]
        
        return result
    
    def save_checkpoint(
        self,
        checkpoint_dir: str,
        tag: Optional[str] = None,
        client_state: Optional[Dict] = None
    ):
        """
        Save DeepSpeed checkpoint.
        
        DeepSpeed handles checkpoint saving differently than standard PyTorch.
        It saves model, optimizer, and scheduler states separately.
        
        Args:
            checkpoint_dir: Directory to save checkpoint
            tag: Optional tag for checkpoint (default: step number)
            client_state: Optional additional state to save
        """
        if tag is None:
            tag = f"step_{self.step_count}"
        
        # Prepare client state
        if client_state is None:
            client_state = {}
        
        client_state["stability_guard_step"] = self.step_count
        
        # Save using DeepSpeed's API
        self.model_engine.save_checkpoint(
            checkpoint_dir,
            tag=tag,
            client_state=client_state
        )
        
        if self.rank == 0:
            logger.info(f"DeepSpeed checkpoint saved to {checkpoint_dir}/{tag}")
    
    def load_checkpoint(
        self,
        checkpoint_dir: str,
        tag: Optional[str] = None,
        load_optimizer_states: bool = True,
        load_lr_scheduler_states: bool = True
    ) -> Dict[str, Any]:
        """
        Load DeepSpeed checkpoint.
        
        Args:
            checkpoint_dir: Directory containing checkpoint
            tag: Optional tag for checkpoint
            load_optimizer_states: Load optimizer states (default: True)
            load_lr_scheduler_states: Load LR scheduler states (default: True)
        
        Returns:
            Client state dictionary
        """
        # Load using DeepSpeed's API
        _, client_state = self.model_engine.load_checkpoint(
            checkpoint_dir,
            tag=tag,
            load_optimizer_states=load_optimizer_states,
            load_lr_scheduler_states=load_lr_scheduler_states
        )
        
        # Restore step count
        if client_state and "stability_guard_step" in client_state:
            self.step_count = client_state["stability_guard_step"]
        
        logger.info(f"Rank {self.rank}: DeepSpeed checkpoint loaded from {checkpoint_dir}/{tag}")
        
        return client_state or {}
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get monitoring statistics.
        
        Returns:
            Dictionary with:
            - step_count: Number of steps monitored
            - grad_norm_stats: Gradient norm statistics
            - spike_stats: Spike detection statistics
            - zero_stage: ZeRO optimization stage
        """
        stats = {
            "step_count": self.step_count,
            "rank": self.rank,
            "world_size": self.world_size,
            "zero_stage": self.zero_stage,
        }
        
        if self.grad_norm_history:
            stats["grad_norm_stats"] = {
                "mean": sum(self.grad_norm_history) / len(self.grad_norm_history),
                "max": max(self.grad_norm_history),
                "min": min(self.grad_norm_history),
            }
        
        if self.spike_detector is not None:
            stats["spike_stats"] = self.spike_detector.get_stats()
        
        return stats
    
    def reset(self):
        """Reset monitoring statistics."""
        self.grad_norm_history = []
        self.step_count = 0
        
        if self.spike_detector is not None:
            self.spike_detector.reset()
        
        if self.rank == 0:
            logger.info("DeepSpeedStabilityGuard reset")
    
    def is_gradient_accumulation_boundary(self) -> bool:
        """
        Check if current step is a gradient accumulation boundary.
        
        DeepSpeed handles gradient accumulation internally.
        
        Returns:
            True if this is an accumulation boundary (optimizer will step)
        """
        if hasattr(self.model_engine, 'is_gradient_accumulation_boundary'):
            return self.model_engine.is_gradient_accumulation_boundary()
        return True
    
    def get_pipeline_parallel_rank(self) -> int:
        """
        Get pipeline parallel rank (for pipeline parallelism).
        
        Returns:
            Pipeline parallel rank, or -1 if not using pipeline parallelism
        """
        if hasattr(self.model_engine, 'grid') and self.model_engine.grid is not None:
            return self.model_engine.grid.get_pipe_parallel_rank()
        return -1

