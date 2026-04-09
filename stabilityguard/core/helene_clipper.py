"""
HELENE - Hessian-Estimated Layer-wise Normalization

Adaptive per-layer gradient clipping based on local Hessian conditioning.
More precise than global gradient clipping because it adapts to each layer's
curvature characteristics.

Key Insight: Layers with high curvature (ill-conditioned Hessian) need more
aggressive clipping than well-conditioned layers. HELENE estimates local
conditioning and sets per-layer clip values accordingly.

Based on research showing that adaptive clipping preserves training signal
better than uniform clipping.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, Any
import logging
import math

logger = logging.getLogger(__name__)


class HELENEClipper:
    """
    Adaptive per-layer gradient clipping based on Hessian conditioning.
    
    For each layer:
    1. Estimate condition number κ = λ_max / λ_min
    2. Set clip value: clip = base_clip / sqrt(κ)
    3. Clip gradients: g = g / max(1, ||g|| / clip)
    
    Layers with high κ (ill-conditioned) get more aggressive clipping.
    Layers with low κ (well-conditioned) get less clipping.
    
    Args:
        model: PyTorch model
        base_clip: Base clipping value (default: 1.0)
        estimation_method: How to estimate conditioning
            - "power_iteration": Full power iteration (accurate but slow)
            - "gradient_variance": Use gradient variance as proxy (fast)
            - "fixed": Use fixed clip values per layer type (fastest)
        estimation_frequency: Estimate conditioning every N steps (default: 50)
        min_clip: Minimum clip value (default: 0.1)
        max_clip: Maximum clip value (default: 10.0)
        verbose: Print clipping statistics (default: False)
    
    Example:
        >>> clipper = HELENEClipper(model, base_clip=1.0)
        >>> 
        >>> # After backward pass
        >>> loss.backward()
        >>> 
        >>> # Apply adaptive clipping
        >>> clip_stats = clipper.clip_gradients(step=100)
        >>> print(f"Clipped {clip_stats['layers_clipped']} layers")
        >>> 
        >>> optimizer.step()
    """
    
    def __init__(
        self,
        model: nn.Module,
        base_clip: float = 1.0,
        estimation_method: str = "gradient_variance",
        estimation_frequency: int = 50,
        min_clip: float = 0.1,
        max_clip: float = 10.0,
        verbose: bool = False
    ):
        # Input validation
        if base_clip <= 0:
            raise ValueError(f"base_clip must be positive, got {base_clip}")
        
        valid_methods = ["power_iteration", "gradient_variance", "fixed"]
        if estimation_method not in valid_methods:
            raise ValueError(
                f"Invalid estimation_method: {estimation_method}. "
                f"Must be one of {valid_methods}"
            )
        
        if min_clip >= max_clip:
            raise ValueError(
                f"min_clip ({min_clip}) must be < max_clip ({max_clip})"
            )
        
        if estimation_frequency <= 0:
            raise ValueError(
                f"estimation_frequency must be positive, got {estimation_frequency}"
            )
        
        self.model = model
        self.base_clip = base_clip
        self.estimation_method = estimation_method
        self.estimation_frequency = estimation_frequency
        self.min_clip = min_clip
        self.max_clip = max_clip
        self.verbose = verbose
        
        # Per-layer clip values
        self.layer_clips: Dict[str, float] = {}
        
        # Per-layer conditioning estimates
        self.layer_conditioning: Dict[str, float] = {}
        
        # Gradient history for variance estimation
        self.gradient_history: Dict[str, list] = {}
        self.history_length = 20
        
        # Statistics
        self.step_count = 0
        self.total_clips = 0
        
        # Initialize with default clip values
        self._initialize_clip_values()
    
    def _initialize_clip_values(self):
        """
        Initialize clip values based on layer types.
        
        Different layer types have different typical conditioning:
        - Attention layers: Often ill-conditioned (κ ~ 100-1000)
        - MLP layers: Moderately conditioned (κ ~ 10-100)
        - Normalization layers: Well-conditioned (κ ~ 1-10)
        """
        for name, module in self.model.named_modules():
            if not self._has_parameters(module):
                continue
            
            # Optimize: compute lower() once
            name_lower = name.lower()
            
            # Heuristic based on layer type
            if 'attention' in name_lower or 'attn' in name_lower:
                # Attention layers: more aggressive clipping
                self.layer_clips[name] = self.base_clip * 0.5
            elif 'mlp' in name_lower or 'fc' in name_lower or 'linear' in name_lower:
                # MLP layers: moderate clipping
                self.layer_clips[name] = self.base_clip
            elif 'norm' in name_lower or 'bn' in name_lower or 'ln' in name_lower:
                # Normalization layers: less aggressive clipping
                self.layer_clips[name] = self.base_clip * 2.0
            else:
                # Default
                self.layer_clips[name] = self.base_clip
            
            # Initialize gradient history
            self.gradient_history[name] = []
    
    def _has_parameters(self, module: nn.Module) -> bool:
        """Check if module has trainable parameters."""
        return any(p.requires_grad for p in module.parameters())
    
    def clip_gradients(self, step: int) -> Dict[str, Any]:
        """
        Apply adaptive per-layer gradient clipping.
        
        Args:
            step: Current training step
            
        Returns:
            Dictionary with clipping statistics:
            - layers_clipped: Number of layers where clipping was applied
            - total_norm_before: Total gradient norm before clipping
            - total_norm_after: Total gradient norm after clipping
            - max_clip_ratio: Maximum ratio of norm to clip value
        """
        self.step_count += 1
        
        # Update conditioning estimates periodically
        if step % self.estimation_frequency == 0:
            self._update_conditioning_estimates()
        
        stats = {
            "layers_clipped": 0,
            "total_norm_before": 0.0,
            "total_norm_after": 0.0,
            "max_clip_ratio": 0.0
        }
        
        # Clip each layer's gradients
        for name, module in self.model.named_modules():
            if name not in self.layer_clips:
                continue
            
            # Get parameters with gradients
            params_with_grad = [
                p for p in module.parameters()
                if p.grad is not None and p.requires_grad
            ]
            
            if len(params_with_grad) == 0:
                continue
            
            # Compute layer gradient norm
            layer_norm = torch.sqrt(
                sum(torch.sum(p.grad ** 2) for p in params_with_grad)
            ).item()
            
            stats["total_norm_before"] += layer_norm ** 2
            
            # Get clip value for this layer
            clip_value = self.layer_clips[name]
            
            # Clip if necessary
            if layer_norm > clip_value:
                clip_ratio = clip_value / layer_norm
                for p in params_with_grad:
                    p.grad.mul_(clip_ratio)
                
                stats["layers_clipped"] += 1
                stats["max_clip_ratio"] = max(stats["max_clip_ratio"], layer_norm / clip_value)
                self.total_clips += 1
                
                # Update norm after clipping
                stats["total_norm_after"] += clip_value ** 2
            else:
                stats["total_norm_after"] += layer_norm ** 2
            
            # Update gradient history for variance estimation
            if len(self.gradient_history[name]) >= self.history_length:
                self.gradient_history[name].pop(0)
            self.gradient_history[name].append(layer_norm)
        
        # Convert to actual norms (not squared)
        stats["total_norm_before"] = math.sqrt(stats["total_norm_before"])
        stats["total_norm_after"] = math.sqrt(stats["total_norm_after"])
        
        if self.verbose and stats["layers_clipped"] > 0:
            logger.info(
                f"HELENE: Clipped {stats['layers_clipped']} layers @ step {step}\n"
                f"   Total norm: {stats['total_norm_before']:.4f} -> {stats['total_norm_after']:.4f}\n"
                f"   Max clip ratio: {stats['max_clip_ratio']:.2f}x"
            )
        
        return stats
    
    def _update_conditioning_estimates(self):
        """
        Update per-layer conditioning estimates and adjust clip values.
        """
        if self.estimation_method == "gradient_variance":
            self._estimate_conditioning_from_variance()
        elif self.estimation_method == "power_iteration":
            self._estimate_conditioning_from_hessian()
        elif self.estimation_method == "fixed":
            pass  # Use fixed values from initialization
        else:
            logger.warning(f"Unknown estimation method: {self.estimation_method}")
    
    def _estimate_conditioning_from_variance(self):
        """
        Estimate conditioning using gradient variance as a proxy.
        
        High variance in gradient norms suggests ill-conditioning.
        Low variance suggests well-conditioning.
        
        This is much faster than computing actual Hessian eigenvalues.
        """
        for name in self.gradient_history:
            history = self.gradient_history[name]
            
            if len(history) < 5:
                continue  # Not enough samples
            
            # Compute coefficient of variation: std / mean
            mean_norm = sum(history) / len(history)
            if mean_norm < 1e-10:
                continue
            
            variance = sum((x - mean_norm) ** 2 for x in history) / len(history)
            std_norm = math.sqrt(variance)
            cv = std_norm / mean_norm
            
            # Estimate conditioning: κ ≈ 1 + cv^2
            # (Heuristic: higher variance → higher conditioning)
            kappa = 1.0 + cv ** 2
            
            # Update clip value: clip = base_clip / sqrt(κ)
            new_clip = self.base_clip / math.sqrt(kappa)
            new_clip = max(self.min_clip, min(self.max_clip, new_clip))
            
            self.layer_clips[name] = new_clip
            self.layer_conditioning[name] = kappa
    
    def _estimate_conditioning_from_hessian(self):
        """
        Estimate conditioning using power iteration on Hessian.
        
        This is more accurate but much slower than variance-based estimation.
        Only use if you need precise conditioning estimates.
        
        Note: This feature is not yet implemented. Using variance-based
        estimation as fallback.
        """
        # TODO: Implement full Hessian-based conditioning estimation
        # This requires computing λ_max and λ_min for each layer
        # Implementation plan:
        # 1. For each layer, extract Hessian submatrix
        # 2. Use power iteration to find λ_max
        # 3. Use inverse power iteration to find λ_min
        # 4. Compute κ = λ_max / λ_min
        # 5. Set clip = base_clip / sqrt(κ)
        
        logger.warning(
            "Hessian-based conditioning estimation not yet implemented. "
            "Falling back to gradient variance method. "
            "To use this feature, please contribute an implementation or "
            "use estimation_method='gradient_variance' instead."
        )
        self._estimate_conditioning_from_variance()
    
    def get_clip_values(self) -> Dict[str, float]:
        """
        Get current clip values for all layers.
        
        Returns:
            Dictionary mapping layer names to clip values
        """
        return self.layer_clips.copy()
    
    def get_conditioning_estimates(self) -> Dict[str, float]:
        """
        Get conditioning estimates for all layers.
        
        Returns:
            Dictionary mapping layer names to condition numbers
        """
        return self.layer_conditioning.copy()
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get clipping statistics.
        
        Returns:
            Dictionary with statistics:
            - total_clips: Total number of clipping operations
            - steps: Total steps processed
            - clip_rate: Average clips per step
            - layer_clips: Current clip values per layer
            - layer_conditioning: Current conditioning estimates per layer
        """
        return {
            "total_clips": self.total_clips,
            "steps": self.step_count,
            "clip_rate": self.total_clips / max(1, self.step_count),
            "layer_clips": self.get_clip_values(),
            "layer_conditioning": self.get_conditioning_estimates()
        }
    
    def set_base_clip(self, new_base_clip: float):
        """
        Update base clip value and recompute all layer clips.
        
        Args:
            new_base_clip: New base clipping value
        """
        ratio = new_base_clip / self.base_clip
        self.base_clip = new_base_clip
        
        # Scale all existing clip values
        for name in self.layer_clips:
            self.layer_clips[name] *= ratio
            self.layer_clips[name] = max(
                self.min_clip,
                min(self.max_clip, self.layer_clips[name])
            )
