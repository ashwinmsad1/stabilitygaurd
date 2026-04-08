"""
GuardedOptimizer — The one-line circuit breaker for PyTorch training.

Wraps any torch.optim.Optimizer. Intercepts step() to run spike
detection before applying gradients. If a spike or NaN is detected,
the configured action is executed (skip, rollback, or raise).

Usage:
    base_opt = AdamW(model.parameters(), lr=2e-4)
    optimizer = GuardedOptimizer(base_opt, model,
        spike_threshold=10.0,
        nan_action="skip",
        log_every=50,
    )

    for batch in dataloader:
        loss = model(batch)
        loss.backward()
        optimizer.step()       # GuardedOptimizer intercepts here
        optimizer.zero_grad()
"""

import logging
import math
from typing import Optional

import torch
import torch.nn as nn

from stabilityguard.core.hooks import GradientHookManager
from stabilityguard.core.spike_detector import SpikeDetector
from stabilityguard.core.actions import get_action, RollbackAction
from stabilityguard.core.snapshot import GradientSnapshot, SpikeReport
from stabilityguard.utils.logging import SpikeLogger

logger = logging.getLogger("stabilityguard")


class GuardedOptimizer:
    """Wraps a PyTorch optimizer with gradient spike detection and remediation.

    On every call to step(), GuardedOptimizer:
      1. Collects per-layer gradient norms from backward hooks.
      2. Runs spike detection (EMA baseline comparison + NaN/Inf check).
      3. If no spike: delegates to the base optimizer's step().
      4. If spike: executes the configured action (skip/rollback/raise).

    The wrapper is transparent — it proxies param_groups, state_dict(),
    load_state_dict(), and zero_grad() to the base optimizer.

    Args:
        base_optimizer: Any torch.optim.Optimizer instance.
        model: The nn.Module being trained (hooks are attached to this).
        spike_threshold: Gradient norm ratio to trigger spike alert.
            A value of 10.0 means 10x deviation from EMA baseline.
        nan_action: Action on spike detection. One of 'skip', 'rollback', 'raise'.
        log_every: Steps between periodic diagnostic summaries.
        log_dir: Directory for JSON spike reports. Default './sg_logs'.
        ema_alpha: EMA smoothing factor for baseline tracking.
        warmup_steps: Steps before spike detection activates.
        verbose: If True, print diagnostic summaries at log_every intervals.
    """

    def __init__(
        self,
        base_optimizer: torch.optim.Optimizer,
        model: nn.Module,
        spike_threshold: float = 10.0,
        nan_action: str = "skip",
        log_every: int = 50,
        log_dir: str = "./sg_logs",
        ema_alpha: float = 0.01,
        warmup_steps: int = 10,
        verbose: bool = True,
    ):
        self._base_optimizer = base_optimizer
        self._model = model
        self._log_every = log_every
        self._verbose = verbose

        # Hook manager — attaches backward hooks to capture gradient norms
        self._hook_manager = GradientHookManager()
        self._hook_manager.attach(model)

        # Spike detector — EMA baselines + threshold comparison
        self._spike_detector = SpikeDetector(
            threshold=spike_threshold,
            ema_alpha=ema_alpha,
            warmup_steps=warmup_steps,
        )

        # Action handler
        self._action = get_action(nan_action)

        # Spike logger — coloured stdout + JSON file output
        self._spike_logger = SpikeLogger(log_dir=log_dir)

        # Step counter
        self._step: int = 0

        # Stats tracking
        self._total_spikes: int = 0
        self._total_skips: int = 0

        # Model hash for spike reports
        self._model_hash = SpikeReport.compute_model_hash(model)

    def step(self, closure=None, loss: Optional[float] = None):
        """Guarded optimizer step.

        Intercepts the standard optimizer.step() call:
          1. Collects gradient norms from hooks.
          2. Checks for spikes.
          3. If spike → executes action (skip/rollback/raise).
          4. If clean → delegates to base optimizer step().
          5. If rollback action → saves checkpoint after clean steps.

        Args:
            closure: Optional closure for optimizers that require it.
            loss: Optional loss value for diagnostic logging.
        """
        self._step += 1

        # 1. Collect gradient norms from hooks
        layer_norms, nan_layers = self._hook_manager.collect()

        # 1b. Direct param.grad scan — catches NaN/Inf injected after backward()
        #     This complements hook-based detection (hooks fire during backward,
        #     but gradients can be modified afterwards)
        for name, param in self._model.named_parameters():
            if param.grad is not None:
                grad = param.grad
                if not torch.isfinite(grad).all():
                    nan_layers.add(name)
                    finite_mask = torch.isfinite(grad)
                    if finite_mask.any():
                        norm = grad[finite_mask].float().norm().item()
                    else:
                        norm = float("inf")
                    layer_norms[name] = norm
                elif name not in layer_norms:
                    # Add param-level norm if not already captured by hooks
                    layer_norms[name] = grad.float().norm().item()

        # 2. Run spike detection
        spike_detected, spike_info, ema_baselines = self._spike_detector.check(
            layer_norms, nan_layers
        )

        # Compute global gradient norm
        global_norm = math.sqrt(sum(n ** 2 for n in layer_norms.values())) if layer_norms else 0.0

        # Build snapshot
        snapshot = GradientSnapshot(
            step=self._step,
            layer_norms=layer_norms,
            ema_baselines=ema_baselines,
            global_norm=global_norm,
            spike_detected=spike_detected,
            spike_layer=spike_info.layer if spike_info else None,
            spike_ratio=spike_info.ratio if spike_info else None,
            action=self._action.name if spike_detected else None,
            loss=loss,
        )

        if spike_detected:
            self._total_spikes += 1
            self._total_skips += 1

            # Build spike report
            sorted_norms = dict(
                sorted(layer_norms.items(), key=lambda x: x[1], reverse=True)[:10]
            )
            report = SpikeReport(
                snapshot=snapshot,
                top_10_norms=sorted_norms,
                nan_layers=list(nan_layers),
                model_arch_hash=self._model_hash,
                cuda_device_id=torch.cuda.current_device() if torch.cuda.is_available() else -1,
                cuda_mem_allocated_gb=(
                    torch.cuda.memory_allocated() / (1024 ** 3)
                    if torch.cuda.is_available()
                    else 0.0
                ),
            )

            # Log spike to stdout and disk
            self._spike_logger.log_spike(report)

            # Execute action (skip/rollback/raise)
            self._action.execute(
                self._base_optimizer, self._model, spike_info, self._step
            )

            # Zero grads to clear corrupted gradients
            self._base_optimizer.zero_grad()

        else:
            # No spike — proceed with normal optimizer step
            if closure is not None:
                self._base_optimizer.step(closure)
            else:
                self._base_optimizer.step()

            # Save checkpoint for rollback (if rollback action is configured)
            if isinstance(self._action, RollbackAction):
                self._action.save_checkpoint(self._model, self._base_optimizer)

        # Periodic diagnostic summary
        if self._verbose and self._step % self._log_every == 0:
            self._spike_logger.log_summary(
                step=self._step,
                global_norm=global_norm,
                total_spikes=self._total_spikes,
                total_skips=self._total_skips,
                num_layers=len(layer_norms),
            )

    def zero_grad(self, set_to_none: bool = True):
        """Proxy to base optimizer zero_grad."""
        self._base_optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        """Proxy to base optimizer state_dict."""
        return self._base_optimizer.state_dict()

    def load_state_dict(self, state_dict):
        """Proxy to base optimizer load_state_dict."""
        self._base_optimizer.load_state_dict(state_dict)

    @property
    def param_groups(self):
        """Proxy to base optimizer param_groups."""
        return self._base_optimizer.param_groups

    @param_groups.setter
    def param_groups(self, value):
        self._base_optimizer.param_groups = value

    @property
    def defaults(self):
        """Proxy to base optimizer defaults."""
        return self._base_optimizer.defaults

    @property
    def state(self):
        """Proxy to base optimizer state."""
        return self._base_optimizer.state

    @property
    def step_count(self) -> int:
        """Number of step() calls made."""
        return self._step

    @property
    def total_spikes(self) -> int:
        """Total number of spikes detected."""
        return self._total_spikes

    @property
    def total_skips(self) -> int:
        """Total number of steps skipped."""
        return self._total_skips

    def close(self):
        """Detach hooks and clean up resources."""
        self._hook_manager.detach()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self):
        return (
            f"GuardedOptimizer(\n"
            f"  base={self._base_optimizer.__class__.__name__},\n"
            f"  threshold={self._spike_detector.threshold},\n"
            f"  action={self._action.name},\n"
            f"  step={self._step}, spikes={self._total_spikes}\n"
            f")"
        )
