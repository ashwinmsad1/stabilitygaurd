"""
GradientSnapshot and SpikeReport dataclasses.

These are the core data models for StabilityGuard — every gradient
observation and spike event is captured in these structures.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import time
import hashlib


@dataclass
class GradientSnapshot:
    """Captures gradient statistics for a single training step.

    Produced by the hook manager after every backward pass. Contains
    per-layer gradient norms, rolling EMA baselines, and spike metadata.
    """

    step: int
    timestamp: float = field(default_factory=time.time)

    # Per-layer gradient norms — key = module qualified name
    layer_norms: Dict[str, float] = field(default_factory=dict)

    # Rolling EMA baselines (updated in-place each step)
    ema_baselines: Dict[str, float] = field(default_factory=dict)

    # Global gradient norm across all parameters
    global_norm: float = 0.0

    # Whether a spike was detected this step
    spike_detected: bool = False
    spike_layer: Optional[str] = None
    spike_ratio: Optional[float] = None

    # Action taken if spike detected
    action: Optional[str] = None  # "skip" | "rollback" | "raise" | None

    # Global loss value at this step (for spike correlation)
    loss: Optional[float] = None

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "step": self.step,
            "timestamp": self.timestamp,
            "layer_norms": self.layer_norms,
            "ema_baselines": self.ema_baselines,
            "global_norm": self.global_norm,
            "spike_detected": self.spike_detected,
            "spike_layer": self.spike_layer,
            "spike_ratio": self.spike_ratio,
            "action": self.action,
            "loss": self.loss,
        }


@dataclass
class SpikeReport:
    """Detailed report written to disk as JSON when a spike is detected.

    Extends GradientSnapshot with forensic data for post-mortem analysis.
    """

    snapshot: GradientSnapshot
    top_10_norms: Dict[str, float]  # sorted by norm descending
    nan_layers: List[str]
    model_arch_hash: str  # sha256 of model.__repr__()
    cuda_device_id: int = -1
    cuda_mem_allocated_gb: float = 0.0

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "snapshot": self.snapshot.to_dict(),
            "top_10_norms": self.top_10_norms,
            "nan_layers": self.nan_layers,
            "model_arch_hash": self.model_arch_hash,
            "cuda_device_id": self.cuda_device_id,
            "cuda_mem_allocated_gb": self.cuda_mem_allocated_gb,
        }

    @staticmethod
    def compute_model_hash(model) -> str:
        """Compute SHA-256 hash of model architecture representation."""
        return hashlib.sha256(repr(model).encode()).hexdigest()[:16]
