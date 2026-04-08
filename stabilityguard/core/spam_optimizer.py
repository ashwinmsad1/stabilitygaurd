"""
SPAM Optimizer - Spike-Aware Momentum

Extends GuardedOptimizer to reset momentum buffers when spikes are detected.
This prevents corrupted gradients from propagating through momentum terms.

Key Insight: When a gradient spike occurs, simply skipping the step isn't enough.
The optimizer's momentum buffers (Adam's m and v) are already corrupted and will
affect future steps. SPAM resets these buffers to prevent contamination.

Based on research showing that momentum reset accelerates recovery from spikes.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List
import logging

logger = logging.getLogger(__name__)


class SPAMOptimizer:
    """
    Spike-Aware Momentum optimizer wrapper.
    
    On spike detection:
    1. Skip the corrupted gradient update (like GuardedOptimizer)
    2. Reset momentum buffers to zero
    3. Optionally reduce learning rate temporarily
    
    Supports:
    - Adam/AdamW (resets exp_avg and exp_avg_sq)
    - SGD with momentum (resets momentum_buffer)
    - RMSprop (resets square_avg)
    - Adagrad (resets sum)
    
    Args:
        optimizer: Base PyTorch optimizer to wrap
        reset_strategy: How to reset momentum
            - "zero": Set all momentum buffers to zero (default)
            - "ema": Exponentially decay momentum buffers
            - "partial": Reset only the spiking layer's momentum
        lr_reduction_factor: Temporarily reduce LR by this factor after spike (default: 1.0, no reduction)
        lr_recovery_steps: Steps to recover original LR (default: 10)
        verbose: Print momentum reset notifications (default: True)
    
    Example:
        >>> base_opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        >>> spam_opt = SPAMOptimizer(
        >>>     base_opt,
        >>>     reset_strategy="zero",
        >>>     lr_reduction_factor=0.5,  # Halve LR after spike
        >>>     lr_recovery_steps=10
        >>> )
        >>> 
        >>> # In training loop, when spike detected:
        >>> if spike_detected:
        >>>     spam_opt.handle_spike(spike_layer_name="transformer.h.11.mlp")
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        reset_strategy: str = "zero",
        lr_reduction_factor: float = 1.0,
        lr_recovery_steps: int = 10,
        verbose: bool = True
    ):
        self.optimizer = optimizer
        self.reset_strategy = reset_strategy
        self.lr_reduction_factor = lr_reduction_factor
        self.lr_recovery_steps = lr_recovery_steps
        self.verbose = verbose
        
        # State tracking
        self.spike_count = 0
        self.last_spike_step = -1
        self.original_lrs = [group['lr'] for group in optimizer.param_groups]
        self.lr_recovery_counter = 0
        
        # Detect optimizer type
        self.optimizer_type = type(optimizer).__name__
        
    def handle_spike(
        self,
        step: int,
        spike_layer_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Handle a detected spike by resetting momentum buffers.
        
        Args:
            step: Current training step
            spike_layer_name: Name of layer that spiked (for partial reset)
            
        Returns:
            Dictionary with reset statistics:
            - buffers_reset: Number of momentum buffers reset
            - lr_reduced: Whether learning rate was reduced
            - new_lr: New learning rate (if reduced)
        """
        self.spike_count += 1
        self.last_spike_step = step
        
        stats = {
            "buffers_reset": 0,
            "lr_reduced": False,
            "new_lr": None
        }
        
        # Reset momentum buffers
        if self.reset_strategy == "zero":
            stats["buffers_reset"] = self._reset_all_momentum()
        elif self.reset_strategy == "ema":
            stats["buffers_reset"] = self._decay_momentum(decay_factor=0.1)
        elif self.reset_strategy == "partial" and spike_layer_name:
            stats["buffers_reset"] = self._reset_layer_momentum(spike_layer_name)
        else:
            logger.warning(f"Unknown reset strategy: {self.reset_strategy}")
        
        # Reduce learning rate if configured
        if self.lr_reduction_factor < 1.0:
            self._reduce_learning_rate()
            stats["lr_reduced"] = True
            stats["new_lr"] = self.optimizer.param_groups[0]['lr']
            self.lr_recovery_counter = 0
        
        if self.verbose:
            logger.info(
                f"🔄 SPAM: Momentum reset @ step {step}\n"
                f"   Buffers reset: {stats['buffers_reset']}\n"
                f"   Strategy: {self.reset_strategy}\n"
                f"   LR reduced: {stats['lr_reduced']}"
            )
        
        return stats
    
    def _reset_all_momentum(self) -> int:
        """
        Reset all momentum buffers to zero.
        
        Returns:
            Number of buffers reset
        """
        count = 0
        
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p not in self.optimizer.state:
                    continue
                
                state = self.optimizer.state[p]
                
                # Adam/AdamW
                if 'exp_avg' in state:
                    state['exp_avg'].zero_()
                    count += 1
                if 'exp_avg_sq' in state:
                    state['exp_avg_sq'].zero_()
                    count += 1
                
                # SGD with momentum
                if 'momentum_buffer' in state:
                    state['momentum_buffer'].zero_()
                    count += 1
                
                # RMSprop
                if 'square_avg' in state:
                    state['square_avg'].zero_()
                    count += 1
                
                # Adagrad
                if 'sum' in state:
                    state['sum'].zero_()
                    count += 1
        
        return count
    
    def _decay_momentum(self, decay_factor: float = 0.1) -> int:
        """
        Exponentially decay momentum buffers instead of zeroing.
        
        This is gentler than full reset and may preserve some useful information.
        
        Args:
            decay_factor: Multiply momentum by this factor (default: 0.1)
            
        Returns:
            Number of buffers decayed
        """
        count = 0
        
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p not in self.optimizer.state:
                    continue
                
                state = self.optimizer.state[p]
                
                # Adam/AdamW
                if 'exp_avg' in state:
                    state['exp_avg'].mul_(decay_factor)
                    count += 1
                if 'exp_avg_sq' in state:
                    state['exp_avg_sq'].mul_(decay_factor)
                    count += 1
                
                # SGD with momentum
                if 'momentum_buffer' in state:
                    state['momentum_buffer'].mul_(decay_factor)
                    count += 1
                
                # RMSprop
                if 'square_avg' in state:
                    state['square_avg'].mul_(decay_factor)
                    count += 1
        
        return count
    
    def _reset_layer_momentum(self, layer_name: str) -> int:
        """
        Reset momentum only for parameters in the specified layer.
        
        This is more surgical than resetting all momentum, but requires
        knowing which layer caused the spike.
        
        Args:
            layer_name: Name of the layer to reset (e.g., "transformer.h.11.mlp")
            
        Returns:
            Number of buffers reset
        """
        # This requires access to the model to map layer names to parameters
        # For now, we'll just reset all momentum (same as "zero" strategy)
        # TODO: Implement layer-specific reset when model is available
        logger.warning("Partial reset not yet implemented, falling back to full reset")
        return self._reset_all_momentum()
    
    def _reduce_learning_rate(self):
        """
        Temporarily reduce learning rate after a spike.
        """
        for i, group in enumerate(self.optimizer.param_groups):
            group['lr'] = self.original_lrs[i] * self.lr_reduction_factor
    
    def step_recovery(self):
        """
        Gradually recover learning rate after a spike.
        
        Call this every step to linearly interpolate LR back to original value.
        """
        if self.lr_recovery_counter >= self.lr_recovery_steps:
            return
        
        self.lr_recovery_counter += 1
        progress = self.lr_recovery_counter / self.lr_recovery_steps
        
        for i, group in enumerate(self.optimizer.param_groups):
            # Linear interpolation from reduced LR to original LR
            reduced_lr = self.original_lrs[i] * self.lr_reduction_factor
            target_lr = self.original_lrs[i]
            group['lr'] = reduced_lr + progress * (target_lr - reduced_lr)
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about SPAM interventions.
        
        Returns:
            Dictionary with statistics:
            - total_spikes: Total number of spikes handled
            - last_spike_step: Step number of last spike
            - current_lr: Current learning rate
            - lr_recovery_progress: Progress toward original LR (0.0-1.0)
        """
        return {
            "total_spikes": self.spike_count,
            "last_spike_step": self.last_spike_step,
            "current_lr": self.optimizer.param_groups[0]['lr'],
            "lr_recovery_progress": min(1.0, self.lr_recovery_counter / self.lr_recovery_steps)
        }
    
    def __getattr__(self, name):
        """
        Delegate all other methods to the wrapped optimizer.
        """
        return getattr(self.optimizer, name)
