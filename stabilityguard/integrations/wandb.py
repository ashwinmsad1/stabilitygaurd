"""
Weights & Biases integration for StabilityGuard.

Streams gradient telemetry and spike events to W&B runs with
the 'sg/' namespace prefix. W&B is an optional dependency —
this module gracefully handles its absence.
"""

import logging
from typing import Optional

logger = logging.getLogger("stabilityguard.integrations.wandb")


class WandBBridge:
    """Streams gradient telemetry directly into W&B runs.

    All StabilityGuard metrics are namespaced under 'sg/' to avoid
    collisions with user metrics.

    Usage:
        from stabilityguard.integrations.wandb import WandBBridge

        bridge = WandBBridge()
        # After each step:
        bridge.log_snapshot(snapshot)

    Requires: pip install stabilityguard[wandb]
    """

    METRICS_PREFIX = "sg/"

    def __init__(self):
        try:
            import wandb

            self._wandb = wandb
            self._available = True
        except ImportError:
            self._wandb = None
            self._available = False
            logger.warning(
                "wandb not installed. Install with: pip install stabilityguard[wandb]"
            )

    @property
    def available(self) -> bool:
        return self._available

    def log_snapshot(self, snapshot) -> None:
        """Log a GradientSnapshot to the active W&B run.

        Args:
            snapshot: GradientSnapshot instance.
        """
        if not self._available:
            return

        wandb = self._wandb
        if wandb.run is None:
            logger.debug("No active W&B run — skipping log")
            return

        payload = {
            f"{self.METRICS_PREFIX}global_grad_norm": snapshot.global_norm,
            f"{self.METRICS_PREFIX}spike_detected": int(snapshot.spike_detected),
        }

        if snapshot.spike_detected and snapshot.spike_layer:
            payload[f"{self.METRICS_PREFIX}spike_ratio"] = snapshot.spike_ratio
            payload[f"{self.METRICS_PREFIX}spike_layer"] = snapshot.spike_layer

        # Log per-layer norms as a summary table (not every step — too expensive)
        if snapshot.step % 50 == 0 and snapshot.layer_norms:
            try:
                layer_table = wandb.Table(
                    columns=["layer", "norm", "ema_baseline", "ratio"],
                    data=[
                        (
                            k,
                            v,
                            snapshot.ema_baselines.get(k, 0.0),
                            v / max(snapshot.ema_baselines.get(k, 1.0), 1e-10),
                        )
                        for k, v in snapshot.layer_norms.items()
                    ],
                )
                payload[f"{self.METRICS_PREFIX}layer_norms"] = layer_table
            except Exception as e:
                logger.debug(f"Failed to create W&B table: {e}")

        try:
            wandb.log(payload, step=snapshot.step)
        except Exception as e:
            logger.warning(f"Failed to log to W&B: {e}")

    def log_spike_report(self, report) -> None:
        """Log a SpikeReport as a W&B alert.

        Args:
            report: SpikeReport instance.
        """
        if not self._available or self._wandb.run is None:
            return

        try:
            self._wandb.alert(
                title=f"Gradient Spike @ step {report.snapshot.step}",
                text=(
                    f"Layer: {report.snapshot.spike_layer}\n"
                    f"Ratio: {report.snapshot.spike_ratio:.1f}x\n"
                    f"Action: {report.snapshot.action}\n"
                    f"NaN layers: {report.nan_layers}"
                ),
                level=self._wandb.AlertLevel.WARN,
            )
        except Exception as e:
            logger.warning(f"Failed to send W&B alert: {e}")
