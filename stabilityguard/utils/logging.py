"""
Coloured stdout formatter and JSON spike report writer.

Produces the distinctive StabilityGuard terminal output with
box-drawn spike alerts and periodic diagnostic summaries.
"""

import json
import os
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger("stabilityguard.logging")


# ANSI colour codes
class _C:
    RED = "\033[91m"
    YELLOW = "\033[93m"
    GREEN = "\033[92m"
    CYAN = "\033[96m"
    DIM = "\033[2m"
    BOLD = "\033[1m"
    RESET = "\033[0m"


class SpikeLogger:
    """Logs spike events to coloured stdout and JSON files.

    Args:
        log_dir: Directory for JSON spike reports. Created automatically.
        enable_stdout: If True, print coloured spike alerts to stdout.
        enable_file: If True, write JSON spike reports to log_dir.
    """

    def __init__(
        self,
        log_dir: str = "./sg_logs",
        enable_stdout: bool = True,
        enable_file: bool = True,
    ):
        self.log_dir = Path(log_dir)
        self.enable_stdout = enable_stdout
        self.enable_file = enable_file

        if self.enable_file:
            self.log_dir.mkdir(parents=True, exist_ok=True)

    def log_spike(self, report) -> None:
        """Log a spike event with full forensic report.

        Args:
            report: SpikeReport instance.
        """
        snapshot = report.snapshot

        if self.enable_stdout:
            self._print_spike_alert(snapshot, report)

        if self.enable_file:
            self._write_json_report(report)

    def log_summary(
        self,
        step: int,
        global_norm: float,
        total_spikes: int,
        total_skips: int,
        num_layers: int,
    ) -> None:
        """Print periodic diagnostic summary."""
        if not self.enable_stdout:
            return

        print(
            f"{_C.DIM}[StabilityGuard] "
            f"step={step} | "
            f"grad_norm={global_norm:.4f} | "
            f"layers={num_layers} | "
            f"spikes={total_spikes} | "
            f"skipped={total_skips}"
            f"{_C.RESET}"
        )

    def _print_spike_alert(self, snapshot, report) -> None:
        """Print the distinctive box-drawn spike alert to stdout."""
        r = _C.RED
        y = _C.YELLOW
        g = _C.GREEN
        c = _C.CYAN
        d = _C.DIM
        b = _C.BOLD
        x = _C.RESET

        layer = snapshot.spike_layer or "unknown"
        ratio = snapshot.spike_ratio
        ratio_str = f"{ratio:.1f}x" if ratio and ratio != float("inf") else "NaN/Inf"
        action = snapshot.action or "none"
        step = snapshot.step
        norm = snapshot.layer_norms.get(layer, 0.0)
        baseline = snapshot.ema_baselines.get(layer, 0.0)
        loss_str = f"{snapshot.loss:.4f}" if snapshot.loss is not None else "N/A"

        nan_info = ""
        if report.nan_layers:
            nan_info = f"  {c}NaN layers   :{x} {', '.join(report.nan_layers)}"

        print(f"""
{d}╔══════════════════════════════════════════════════════════════╗{x}
{d}║{x}  {r}{b}WARNING: STABILITYGUARD - SPIKE DETECTED @ step {step}{x}                {d}║{x}
{d}╠══════════════════════════════════════════════════════════════╣{x}
{d}║{x}  {g}Trigger layer  :{x} {layer}
{d}║{x}  {g}Grad norm      :{x} {norm:.1f}  (baseline: {baseline:.1f}, ratio: {ratio_str})
{d}║{x}  {g}Action taken   :{x} optimizer.step() {action.upper()}
{d}║{x}  {g}Loss (pre-skip):{x} {loss_str}
{nan_info}{d}╚══════════════════════════════════════════════════════════════╝{x}
{y}stabilityguard.log{x} written → {c}{self.log_dir}/spike_step{step}.json{x}
""")

    def _write_json_report(self, report) -> None:
        """Write spike report to JSON file."""
        filepath = self.log_dir / f"spike_step{report.snapshot.step}.json"
        try:
            with open(filepath, "w") as f:
                json.dump(report.to_dict(), f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"Failed to write spike report: {e}")
