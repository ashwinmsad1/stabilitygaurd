"""
Actions executed when a gradient spike is detected.

Three actions are supported:
  - skip:     Skip optimizer.step() entirely — weights unchanged.
  - rollback: Restore model to previous checkpoint state.
  - raise:    Raise GradientSpikeError for interactive debugging.
"""

import copy
import logging
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger("stabilityguard.actions")


class GradientSpikeError(Exception):
    """Raised when nan_action='raise' and a spike is detected."""

    def __init__(self, layer: str, ratio: float, step: int):
        self.layer = layer
        self.ratio = ratio
        self.step = step
        super().__init__(
            f"Gradient spike detected at step {step}: "
            f"layer='{layer}', ratio={ratio:.1f}x"
        )


class SkipAction:
    """Skip optimizer.step() — corrupted gradients are not applied."""

    name = "skip"

    def execute(self, optimizer, model, spike_info, step: int):
        """Do nothing — step() is simply not called by GuardedOptimizer."""
        logger.info(
            f"[StabilityGuard] SKIPPED step {step} — "
            f"spike in '{spike_info.layer}' ({spike_info.ratio:.1f}x baseline)"
        )


class RollbackAction:
    """Restore model weights to the last saved checkpoint."""

    name = "rollback"

    def __init__(self):
        self._checkpoint: Optional[dict] = None
        self._optimizer_checkpoint: Optional[dict] = None

    def save_checkpoint(self, model: nn.Module, optimizer):
        """Save current model and optimizer state as checkpoint.

        Called after every successful (non-spike) step.
        Uses deepcopy for safety — overhead is acceptable since
        it only runs on successful steps.
        """
        self._checkpoint = copy.deepcopy(model.state_dict())
        self._optimizer_checkpoint = copy.deepcopy(optimizer.state_dict())

    def execute(self, optimizer, model, spike_info, step: int):
        """Restore model and optimizer to last checkpoint."""
        if self._checkpoint is None:
            logger.warning(
                f"[StabilityGuard] ROLLBACK requested at step {step} "
                f"but no checkpoint available — falling back to SKIP"
            )
            return

        model.load_state_dict(self._checkpoint)
        optimizer.load_state_dict(self._optimizer_checkpoint)
        logger.info(
            f"[StabilityGuard] ROLLED BACK to previous checkpoint at step {step} — "
            f"spike in '{spike_info.layer}' ({spike_info.ratio:.1f}x baseline)"
        )


class RaiseAction:
    """Raise a GradientSpikeError for interactive debugging."""

    name = "raise"

    def execute(self, optimizer, model, spike_info, step: int):
        raise GradientSpikeError(
            layer=spike_info.layer,
            ratio=spike_info.ratio,
            step=step,
        )


def get_action(action_name: str):
    """Factory: create an action handler from a string name.

    Args:
        action_name: One of 'skip', 'rollback', 'raise'.

    Returns:
        An action instance.

    Raises:
        ValueError: If action_name is not recognized.
    """
    actions = {
        "skip": SkipAction,
        "rollback": RollbackAction,
        "raise": RaiseAction,
    }
    if action_name not in actions:
        raise ValueError(
            f"Unknown action '{action_name}'. "
            f"Must be one of: {list(actions.keys())}"
        )
    return actions[action_name]()
