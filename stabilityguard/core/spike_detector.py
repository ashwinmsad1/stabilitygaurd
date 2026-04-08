"""
Spike detection engine using Exponential Moving Average (EMA) baselines.

Maintains an EMA (α=0.01) of per-layer gradient norms. A spike is
detected when current_norm / ema_baseline > threshold or when NaN/Inf
gradients are found.
"""

import math
from typing import Dict, Optional, Set, Tuple
from dataclasses import dataclass


@dataclass
class SpikeInfo:
    """Information about a detected gradient spike."""

    layer: str
    current_norm: float
    baseline: float
    ratio: float
    is_nan: bool = False
    is_inf: bool = False


class SpikeDetector:
    """Detects gradient spikes using EMA baseline comparison.

    For each layer, maintains an exponential moving average of gradient
    norms. When the current norm exceeds the baseline by more than
    `threshold`, a spike is flagged.

    The EMA uses α=0.01 by default, meaning the baseline adapts slowly
    and represents the stable gradient norm over the last ~100 steps.

    NaN/Inf gradients are always flagged as spikes regardless of threshold.

    Args:
        threshold: Gradient norm ratio (current/baseline) to trigger spike.
            Default 10.0 means a 10x deviation from baseline triggers.
        ema_alpha: EMA smoothing factor. Smaller = slower adaptation.
            Default 0.01 provides a ~100-step lookback window.
        warmup_steps: Steps before spike detection activates. During warmup,
            only EMA baselines are updated (no spike alerts). Default 10.
    """

    def __init__(
        self,
        threshold: float = 10.0,
        ema_alpha: float = 0.01,
        warmup_steps: int = 10,
    ):
        self.threshold = threshold
        self.ema_alpha = ema_alpha
        self.warmup_steps = warmup_steps

        self._ema_baselines: Dict[str, float] = {}
        self._step_count: int = 0

    def check(
        self,
        layer_norms: Dict[str, float],
        nan_layers: Set[str],
    ) -> Tuple[bool, Optional[SpikeInfo], Dict[str, float]]:
        """Check gradient norms for spikes and update EMA baselines.

        Args:
            layer_norms: Dict mapping layer name → current gradient L2 norm.
            nan_layers: Set of layer names where NaN/Inf was detected.

        Returns:
            Tuple of:
                - spike_detected (bool): True if any spike was found.
                - spike_info (SpikeInfo or None): Info about the worst spike.
                - ema_baselines (dict): Current EMA baselines (copy).
        """
        self._step_count += 1
        worst_spike: Optional[SpikeInfo] = None
        worst_ratio: float = 0.0

        # NaN/Inf layers are always spikes
        for layer_name in nan_layers:
            norm = layer_norms.get(layer_name, float("inf"))
            baseline = self._ema_baselines.get(layer_name, 1.0)
            spike = SpikeInfo(
                layer=layer_name,
                current_norm=norm,
                baseline=baseline,
                ratio=float("inf"),
                is_nan=True,
            )
            # NaN is always the worst spike
            worst_spike = spike
            worst_ratio = float("inf")

        # Check each layer for threshold violations
        for layer_name, norm in layer_norms.items():
            if layer_name in nan_layers:
                # Already handled above; still update EMA with a clamped value
                continue

            if not math.isfinite(norm):
                continue

            # Update EMA baseline
            if layer_name not in self._ema_baselines:
                self._ema_baselines[layer_name] = norm
            else:
                self._ema_baselines[layer_name] = (
                    (1 - self.ema_alpha) * self._ema_baselines[layer_name]
                    + self.ema_alpha * norm
                )

            # Skip spike detection during warmup
            if self._step_count <= self.warmup_steps:
                continue

            baseline = self._ema_baselines[layer_name]
            if baseline < 1e-10:
                # Baseline near zero — can't compute meaningful ratio
                continue

            ratio = norm / baseline
            if ratio > self.threshold and ratio > worst_ratio:
                worst_spike = SpikeInfo(
                    layer=layer_name,
                    current_norm=norm,
                    baseline=baseline,
                    ratio=ratio,
                )
                worst_ratio = ratio

        spike_detected = worst_spike is not None
        return spike_detected, worst_spike, dict(self._ema_baselines)

    @property
    def baselines(self) -> Dict[str, float]:
        """Current EMA baselines (read-only copy)."""
        return dict(self._ema_baselines)

    @property
    def step_count(self) -> int:
        return self._step_count

    def reset(self):
        """Reset all baselines and step counter."""
        self._ema_baselines.clear()
        self._step_count = 0
