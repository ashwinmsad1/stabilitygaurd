"""
Gradient hook manager for PyTorch models.

Registers backward hooks on every nn.Module to capture per-layer
gradient norms during backpropagation. Hooks are non-blocking —
if a hook callback fails, training continues uninterrupted.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Set
import logging

logger = logging.getLogger("stabilityguard.hooks")


class GradientHookManager:
    """Registers and manages PyTorch backward hooks for gradient monitoring.

    Attaches `register_full_backward_hook` on every nn.Module in the model.
    Hooks capture grad_output L2 norms per layer, deposited into a shared
    dictionary that is consumed by the SpikeDetector after each backward pass.

    Hooks are non-blocking: execution continues even if a hook callback
    raises, preventing training halt on monitor failure.
    """

    def __init__(self):
        self._hooks = []
        self._current_norms: Dict[str, float] = {}
        self._nan_layers: Set[str] = set()
        self._attached = False

    def attach(self, model: nn.Module, skip_containers: bool = True):
        """Register backward hooks on all modules in the model.

        Args:
            model: The PyTorch model to monitor.
            skip_containers: If True, skip pure container modules (Sequential,
                ModuleList, ModuleDict) that don't have their own parameters,
                to reduce hook overhead.
        """
        if self._attached:
            self.detach()

        container_types = (nn.Sequential, nn.ModuleList, nn.ModuleDict)

        for name, module in model.named_modules():
            if skip_containers and isinstance(module, container_types):
                continue
            # Skip the top-level model itself if it's just a container
            if name == "" and not list(module.parameters(recurse=False)):
                continue

            hook = module.register_full_backward_hook(
                self._make_hook(name if name else "root")
            )
            self._hooks.append(hook)

        self._attached = True
        logger.debug(
            f"Attached {len(self._hooks)} gradient hooks to model"
        )

    def detach(self):
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        self._attached = False
        logger.debug("Detached all gradient hooks")

    def collect(self) -> tuple:
        """Collect gradient norms from the last backward pass.

        Returns:
            Tuple of (layer_norms dict, nan_layers set). Clears internal
            buffers after collection.
        """
        norms = dict(self._current_norms)
        nan_layers = set(self._nan_layers)
        self._current_norms.clear()
        self._nan_layers.clear()
        return norms, nan_layers

    def _make_hook(self, name: str):
        """Create a backward hook closure for a named module.

        The hook computes the L2 norm of grad_output tensors.
        All errors are caught and logged to prevent training interruption.
        """
        manager = self

        def hook(module: nn.Module, grad_input, grad_output):
            try:
                for i, grad in enumerate(grad_output):
                    if grad is None:
                        continue

                    # Check for NaN/Inf first (cheap short-circuit)
                    if not torch.isfinite(grad).all():
                        manager._nan_layers.add(name)
                        has_nan = torch.isnan(grad).any().item()
                        has_inf = torch.isinf(grad).any().item()
                        # Still compute norm for diagnostics (using finite values)
                        finite_grad = grad[torch.isfinite(grad)]
                        if finite_grad.numel() > 0:
                            norm = finite_grad.float().norm().item()
                        else:
                            norm = float("inf")
                        manager._current_norms[name] = norm
                        return

                    # Compute L2 norm
                    norm = grad.float().norm().item()
                    # If multiple grad_outputs, keep the max norm
                    if name in manager._current_norms:
                        manager._current_norms[name] = max(
                            manager._current_norms[name], norm
                        )
                    else:
                        manager._current_norms[name] = norm

            except Exception as e:
                # Non-blocking: log and continue, never halt training
                logger.warning(
                    f"StabilityGuard hook error on '{name}': {e}"
                )

        return hook

    @property
    def is_attached(self) -> bool:
        return self._attached
