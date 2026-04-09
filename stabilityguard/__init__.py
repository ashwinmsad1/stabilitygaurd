"""
StabilityGuard — Comprehensive PyTorch training stability platform.

v0.3.0 Features:
- RLHF/PPO stability monitoring
- Distributed training support (DDP, FSDP, DeepSpeed)
- Mixed precision stability (FP16/BF16/FP8)
- Advanced logging and diagnostics

Basic Usage:
    from stabilityguard import GuardedOptimizer

    base_opt = AdamW(model.parameters(), lr=2e-4)
    optimizer = GuardedOptimizer(base_opt, model,
        spike_threshold=10.0,
        nan_action="skip",
        log_every=50,
    )

Advanced Usage:
    from stabilityguard.rlhf import RLHFGuard
    from stabilityguard.distributed import DistributedGuardedOptimizer
    from stabilityguard.precision import MixedPrecisionGuard
    from stabilityguard.logging import AdvancedLogger
"""

__version__ = "0.3.0"

from stabilityguard.core.guarded_optimizer import GuardedOptimizer
from stabilityguard.core.snapshot import GradientSnapshot, SpikeReport

__all__ = [
    "GuardedOptimizer",
    "GradientSnapshot",
    "SpikeReport",
    "__version__",
]
