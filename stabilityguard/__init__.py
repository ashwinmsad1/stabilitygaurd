"""
StabilityGuard — One-line circuit breaker for PyTorch training.

Add one line. See exactly which layer is about to explode.

Usage:
    from stabilityguard import GuardedOptimizer

    base_opt = AdamW(model.parameters(), lr=2e-4)
    optimizer = GuardedOptimizer(base_opt, model,
        spike_threshold=10.0,
        nan_action="skip",
        log_every=50,
    )
"""

__version__ = "0.1.0"

from stabilityguard.core.guarded_optimizer import GuardedOptimizer
from stabilityguard.core.snapshot import GradientSnapshot, SpikeReport

__all__ = [
    "GuardedOptimizer",
    "GradientSnapshot",
    "SpikeReport",
    "__version__",
]
