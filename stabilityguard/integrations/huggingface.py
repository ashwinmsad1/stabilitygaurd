"""
HuggingFace Transformers integration for StabilityGuard.

Provides a TrainerCallback that automatically wraps the Trainer's
optimizer with GuardedOptimizer. Zero-change usage for HF Trainer users.
"""

import logging
from typing import Optional

logger = logging.getLogger("stabilityguard.integrations.huggingface")


class StabilityGuardCallback:
    """HuggingFace TrainerCallback that wraps the optimizer with GuardedOptimizer.

    Usage:
        from stabilityguard.integrations.huggingface import StabilityGuardCallback

        trainer = Trainer(
            model=model,
            args=training_args,
            callbacks=[StabilityGuardCallback(spike_threshold=10.0)],
        )

    The callback intercepts optimizer creation and wraps it transparently.
    All existing Trainer functionality (gradient accumulation, mixed precision,
    DeepSpeed) continues to work as expected.

    Requires: pip install stabilityguard[huggingface]
    """

    def __init__(
        self,
        spike_threshold: float = 10.0,
        nan_action: str = "skip",
        log_every: int = 50,
        log_dir: str = "./sg_logs",
        verbose: bool = True,
    ):
        self.spike_threshold = spike_threshold
        self.nan_action = nan_action
        self.log_every = log_every
        self.log_dir = log_dir
        self.verbose = verbose
        self._guarded_optimizer = None

        # Check if transformers is available
        try:
            from transformers import TrainerCallback

            self._base_class = TrainerCallback
            self._available = True
        except ImportError:
            self._base_class = object
            self._available = False
            logger.warning(
                "transformers not installed. "
                "Install with: pip install stabilityguard[huggingface]"
            )

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        """Wrap the optimizer when training begins."""
        if not self._available:
            return

        optimizer = kwargs.get("optimizer") or getattr(state, "optimizer", None)
        if optimizer is None:
            logger.warning(
                "Could not find optimizer in Trainer state — "
                "StabilityGuard not activated"
            )
            return

        from stabilityguard.core.guarded_optimizer import GuardedOptimizer

        if isinstance(optimizer, GuardedOptimizer):
            logger.info("Optimizer already guarded — skipping")
            return

        self._guarded_optimizer = GuardedOptimizer(
            base_optimizer=optimizer,
            model=model,
            spike_threshold=self.spike_threshold,
            nan_action=self.nan_action,
            log_every=self.log_every,
            log_dir=self.log_dir,
            verbose=self.verbose,
        )

        logger.info(
            f"StabilityGuard activated via HuggingFace callback "
            f"(threshold={self.spike_threshold}, action={self.nan_action})"
        )

    def on_train_end(self, args, state, control, **kwargs):
        """Clean up hooks when training ends."""
        if self._guarded_optimizer is not None:
            self._guarded_optimizer.close()
            self._guarded_optimizer = None
            logger.info("StabilityGuard hooks detached")
