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
from typing import Optional, Dict, Any

import torch
import torch.nn as nn

from stabilityguard.core.hooks import GradientHookManager
from stabilityguard.core.spike_detector import SpikeDetector
from stabilityguard.core.actions import get_action, RollbackAction
from stabilityguard.core.snapshot import GradientSnapshot, SpikeReport
from stabilityguard.utils.logging import SpikeLogger

# v0.2.0 imports
from stabilityguard.core.edge_of_stability import EdgeOfStabilityDetector
from stabilityguard.core.spam_optimizer import SPAMOptimizer
from stabilityguard.core.auto_calibration import AutoCalibrator
from stabilityguard.core.helene_clipper import HELENEClipper

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
        
        # v0.1.x parameters
        spike_threshold: Gradient norm ratio to trigger spike alert (default: 10.0)
        nan_action: Action on spike detection ('skip', 'rollback', 'raise')
        log_every: Steps between periodic diagnostic summaries
        log_dir: Directory for JSON spike reports
        ema_alpha: EMA smoothing factor for baseline tracking
        warmup_steps: Steps before spike detection activates
        verbose: Print diagnostic summaries
        
        # v0.2.0 parameters (opt-in features)
        enable_edge_detection: Enable Edge of Stability detection (default: False)
        edge_power_iterations: Power iterations for λ_max estimation (default: 20)
        edge_estimation_frequency: Estimate λ_max every N steps (default: 10)
        
        enable_spam: Enable SPAM momentum reset (default: False)
        spam_reset_strategy: How to reset momentum ('zero', 'ema', 'partial')
        spam_lr_reduction: LR reduction factor after spike (default: 1.0)
        
        enable_helene: Enable HELENE adaptive clipping (default: False)
        helene_base_clip: Base clipping value (default: 1.0)
        helene_estimation_method: Conditioning estimation method (default: 'gradient_variance')
        
        auto_calibrate: Enable auto-calibration (default: False)
        calibration_warmup: Steps for calibration (default: 100)
        calibration_percentile: Threshold percentile (default: 99.0)
    """

    def __init__(
        self,
        base_optimizer: torch.optim.Optimizer,
        model: nn.Module,
        # v0.1.x parameters
        spike_threshold: float = 10.0,
        nan_action: str = "skip",
        log_every: int = 50,
        log_dir: str = "./sg_logs",
        ema_alpha: float = 0.01,
        warmup_steps: int = 10,
        verbose: bool = True,
        # v0.2.0 parameters (opt-in)
        enable_edge_detection: bool = False,
        edge_power_iterations: int = 20,
        edge_estimation_frequency: int = 10,
        enable_spam: bool = False,
        spam_reset_strategy: str = "zero",
        spam_lr_reduction: float = 1.0,
        enable_helene: bool = False,
        helene_base_clip: float = 1.0,
        helene_estimation_method: str = "gradient_variance",
        auto_calibrate: bool = False,
        calibration_warmup: int = 100,
        calibration_percentile: float = 99.0,
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
        
        # v0.2.0 features (opt-in via parameters)
        self._edge_detector: Optional[EdgeOfStabilityDetector] = None
        self._spam_optimizer: Optional[SPAMOptimizer] = None
        self._helene_clipper: Optional[HELENEClipper] = None
        self._auto_calibrator: Optional[AutoCalibrator] = None
        
        # Initialize v0.2.0 features if enabled
        if enable_edge_detection:
            self._edge_detector = EdgeOfStabilityDetector(
                model=model,
                power_iterations=edge_power_iterations,
                estimation_frequency=edge_estimation_frequency,
                warmup_steps=warmup_steps,
                verbose=verbose,
            )
            if verbose:
                logger.info("Edge of Stability detection enabled")
        
        if enable_spam:
            self._spam_optimizer = SPAMOptimizer(
                optimizer=base_optimizer,
                reset_strategy=spam_reset_strategy,
                lr_reduction_factor=spam_lr_reduction,
                verbose=verbose,
            )
            if verbose:
                logger.info("SPAM momentum reset enabled")
        
        if enable_helene:
            self._helene_clipper = HELENEClipper(
                model=model,
                base_clip=helene_base_clip,
                estimation_method=helene_estimation_method,
                verbose=verbose,
            )
            if verbose:
                logger.info("HELENE adaptive clipping enabled")
        
        if auto_calibrate:
            self._auto_calibrator = AutoCalibrator(
                warmup_steps=calibration_warmup,
                percentile=calibration_percentile,
                verbose=verbose,
            )
            if verbose:
                logger.info("Auto-calibration enabled")

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
        
        # Get current learning rate
        current_lr = self._base_optimizer.param_groups[0]['lr']
        
        # v0.2.0: Edge of Stability check (predictive, before spike detection)
        edge_warning = False
        if self._edge_detector and loss is not None:
            lambda_max, sharpness, is_unstable = self._edge_detector.check_stability(
                loss, current_lr, self._step
            )
            if is_unstable:
                edge_warning = True
        
        # v0.2.0: HELENE adaptive clipping (before collecting norms)
        if self._helene_clipper:
            self._helene_clipper.clip_gradients(self._step)

        # 1. Collect gradient norms from hooks
        layer_norms, nan_layers = self._hook_manager.collect()

        # 1b. Direct param.grad scan — catches NaN/Inf injected after backward()
        #     This complements hook-based detection for two reasons:
        #     a) Hooks fire during backward, but gradients can be modified afterwards
        #     b) Some parameters might not have hooks (e.g., if added dynamically)
        #
        #     Note: This adds overhead but is necessary for complete coverage.
        #     For performance-critical applications, consider disabling if you're
        #     certain no post-backward gradient modifications occur.
        for name, param in self._model.named_parameters():
            if param.grad is not None:
                grad = param.grad
                # Check for NaN/Inf that might have been introduced after hooks
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
                    # This handles parameters that might not have hooks attached
                    layer_norms[name] = grad.float().norm().item()
        
        # v0.2.0: Auto-calibration sample collection (during warmup)
        if self._auto_calibrator and not self._auto_calibrator.is_calibrated:
            self._auto_calibrator.add_samples(layer_norms)
            # Update threshold if calibration just completed
            if self._auto_calibrator.is_calibrated:
                new_threshold = self._auto_calibrator.get_threshold()
                self._spike_detector.threshold = new_threshold
                if self._verbose:
                    logger.info(f"Threshold auto-calibrated to {new_threshold:.2f}")

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
            
            # v0.2.0: SPAM momentum reset
            if self._spam_optimizer:
                self._spam_optimizer.handle_spike(
                    step=self._step,
                    spike_layer_name=spike_info.layer if spike_info else None
                )

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
            
            # v0.2.0: SPAM LR recovery (gradually restore LR after spike)
            if self._spam_optimizer:
                self._spam_optimizer.step_recovery()

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
