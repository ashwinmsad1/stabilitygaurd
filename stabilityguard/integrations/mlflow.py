"""
MLflow integration for StabilityGuard.

Logs gradient metrics and spike report artifacts to MLflow.
MLflow is an optional dependency.
"""

import json
import logging
import tempfile
from pathlib import Path
from typing import Optional

logger = logging.getLogger("stabilityguard.integrations.mlflow")


class MLflowBridge:
    """Logs StabilityGuard metrics and artifacts to MLflow.

    Usage:
        from stabilityguard.integrations.mlflow import MLflowBridge

        bridge = MLflowBridge()
        bridge.log_snapshot(snapshot)

    Requires: pip install stabilityguard[mlflow]
    """

    METRICS_PREFIX = "sg."

    def __init__(self):
        try:
            import mlflow

            self._mlflow = mlflow
            self._available = True
        except ImportError:
            self._mlflow = None
            self._available = False
            logger.warning(
                "mlflow not installed. Install with: pip install stabilityguard[mlflow]"
            )

    @property
    def available(self) -> bool:
        return self._available

    def log_snapshot(self, snapshot) -> None:
        """Log gradient metrics from a GradientSnapshot.

        Args:
            snapshot: GradientSnapshot instance.
        """
        if not self._available:
            return

        mlflow = self._mlflow

        try:
            metrics = {
                f"{self.METRICS_PREFIX}global_grad_norm": snapshot.global_norm,
                f"{self.METRICS_PREFIX}spike_detected": int(snapshot.spike_detected),
            }

            if snapshot.spike_detected and snapshot.spike_ratio is not None:
                metrics[f"{self.METRICS_PREFIX}spike_ratio"] = snapshot.spike_ratio

            mlflow.log_metrics(metrics, step=snapshot.step)
        except Exception as e:
            logger.warning(f"Failed to log to MLflow: {e}")

    def log_spike_report(self, report) -> None:
        """Log a SpikeReport as an MLflow artifact.

        Args:
            report: SpikeReport instance.
        """
        if not self._available:
            return

        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".json",
                prefix=f"spike_step{report.snapshot.step}_",
                delete=False,
            ) as f:
                json.dump(report.to_dict(), f, indent=2, default=str)
                temp_path = f.name

            self._mlflow.log_artifact(temp_path, "spike_reports")

            # Clean up temp file
            Path(temp_path).unlink(missing_ok=True)
        except Exception as e:
            logger.warning(f"Failed to log spike artifact to MLflow: {e}")
