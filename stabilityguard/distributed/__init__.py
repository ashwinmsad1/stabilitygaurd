"""
Distributed training stability monitoring.

This module provides specialized monitoring for distributed training across multiple GPUs:
- Rank-aware spike detection
- Coordinated rollback across all ranks
- FSDP (Fully Sharded Data Parallel) support
- DeepSpeed ZeRO support
- All-reduce spike detection
"""

from .spike_detector import DistributedSpikeDetector
from .fsdp_guard import FSDPStabilityGuard
from .deepspeed_guard import DeepSpeedStabilityGuard
from .distributed_optimizer import DistributedGuardedOptimizer

__all__ = [
    "DistributedSpikeDetector",
    "FSDPStabilityGuard",
    "DeepSpeedStabilityGuard",
    "DistributedGuardedOptimizer",
]


