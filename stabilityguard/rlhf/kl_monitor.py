"""
KL Divergence Monitor for RLHF Training

Monitors KL divergence between policy and reference model to prevent
policy from deviating too far from the reference during RLHF training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)


class KLDivergenceMonitor:
    """
    Monitor KL divergence between policy and reference model.
    
    In RLHF training, we want to maximize reward while keeping the policy
    close to the reference model. This is typically done by adding a KL
    penalty term to the objective:
    
        objective = reward - β * KL(π_θ || π_ref)
    
    This monitor tracks KL divergence and can adaptively adjust β.
    
    Args:
        target_kl: Target KL divergence (default: 0.1)
        kl_penalty: Initial KL penalty coefficient β (default: 0.2)
        adaptive: Enable adaptive KL penalty adjustment (default: True)
        adaptation_rate: Rate of β adjustment (default: 1.5)
        min_kl_penalty: Minimum β value (default: 0.01)
        max_kl_penalty: Maximum β value (default: 10.0)
        verbose: Print KL statistics (default: True)
    
    Example:
        >>> monitor = KLDivergenceMonitor(target_kl=0.1)
        >>> 
        >>> # During training
        >>> policy_logits = policy_model(input_ids)
        >>> ref_logits = ref_model(input_ids)
        >>> 
        >>> kl_div, stats = monitor.compute_kl(policy_logits, ref_logits)
        >>> 
        >>> # Add KL penalty to loss
        >>> loss = reward_loss + monitor.kl_penalty * kl_div
        >>> 
        >>> # Update β based on observed KL
        >>> monitor.update_penalty(kl_div)
    """
    
    def __init__(
        self,
        target_kl: float = 0.1,
        kl_penalty: float = 0.2,
        adaptive: bool = True,
        adaptation_rate: float = 1.5,
        min_kl_penalty: float = 0.01,
        max_kl_penalty: float = 10.0,
        verbose: bool = True
    ):
        # Input validation
        if target_kl <= 0:
            raise ValueError(f"target_kl must be positive, got {target_kl}")
        if kl_penalty <= 0:
            raise ValueError(f"kl_penalty must be positive, got {kl_penalty}")
        if adaptation_rate <= 1.0:
            raise ValueError(f"adaptation_rate must be > 1.0, got {adaptation_rate}")
        if min_kl_penalty >= max_kl_penalty:
            raise ValueError(
                f"min_kl_penalty ({min_kl_penalty}) must be < max_kl_penalty ({max_kl_penalty})"
            )
        
        self.target_kl = target_kl
        self.kl_penalty = kl_penalty
        self.adaptive = adaptive
        self.adaptation_rate = adaptation_rate
        self.min_kl_penalty = min_kl_penalty
        self.max_kl_penalty = max_kl_penalty
        self.verbose = verbose
        
        # Statistics tracking
        self.step_count = 0
        self.kl_history = []
        self.penalty_history = []
        self.total_updates = 0
        
    def compute_kl(
        self,
        policy_logits: torch.Tensor,
        ref_logits: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Compute KL divergence between policy and reference distributions.
        
        KL(π_θ || π_ref) = Σ π_θ(a|s) * log(π_θ(a|s) / π_ref(a|s))
        
        Args:
            policy_logits: Logits from policy model [batch, seq_len, vocab]
            ref_logits: Logits from reference model [batch, seq_len, vocab]
            mask: Optional attention mask [batch, seq_len]
            
        Returns:
            Tuple of (kl_divergence, statistics_dict)
            - kl_divergence: Scalar tensor with mean KL divergence
            - statistics_dict: Dict with detailed statistics
        """
        # Convert logits to log probabilities
        policy_log_probs = F.log_softmax(policy_logits, dim=-1)
        ref_log_probs = F.log_softmax(ref_logits, dim=-1)
        
        # Compute KL divergence: KL(P||Q) = Σ P * log(P/Q) = Σ P * (log P - log Q)
        # Using policy as P and reference as Q
        policy_probs = torch.exp(policy_log_probs)
        kl_div = policy_probs * (policy_log_probs - ref_log_probs)
        kl_div = kl_div.sum(dim=-1)  # Sum over vocabulary
        
        # Apply mask if provided
        if mask is not None:
            kl_div = kl_div * mask
            num_tokens = mask.sum()
        else:
            num_tokens = kl_div.numel()
        
        # Compute mean KL divergence
        mean_kl = kl_div.sum() / num_tokens
        
        # Compute statistics
        stats = {
            "mean_kl": mean_kl.item(),
            "max_kl": kl_div.max().item(),
            "min_kl": kl_div.min().item(),
            "kl_penalty": self.kl_penalty,
            "target_kl": self.target_kl,
            "num_tokens": num_tokens.item() if isinstance(num_tokens, torch.Tensor) else num_tokens
        }
        
        # Update history
        self.kl_history.append(mean_kl.item())
        self.penalty_history.append(self.kl_penalty)
        self.step_count += 1
        
        return mean_kl, stats
    
    def update_penalty(self, observed_kl: float) -> float:
        """
        Adaptively update KL penalty coefficient β based on observed KL.
        
        If KL > target: increase β (penalize more)
        If KL < target: decrease β (penalize less)
        
        Args:
            observed_kl: Observed KL divergence value
            
        Returns:
            New KL penalty coefficient
        """
        if not self.adaptive:
            return self.kl_penalty
        
        old_penalty = self.kl_penalty
        
        # Adaptive adjustment
        if observed_kl > self.target_kl * 1.5:
            # KL too high: increase penalty aggressively
            self.kl_penalty *= self.adaptation_rate
        elif observed_kl > self.target_kl:
            # KL slightly high: increase penalty moderately
            self.kl_penalty *= (1.0 + (self.adaptation_rate - 1.0) * 0.5)
        elif observed_kl < self.target_kl * 0.5:
            # KL too low: decrease penalty aggressively
            self.kl_penalty /= self.adaptation_rate
        elif observed_kl < self.target_kl:
            # KL slightly low: decrease penalty moderately
            self.kl_penalty /= (1.0 + (self.adaptation_rate - 1.0) * 0.5)
        
        # Clamp to valid range
        self.kl_penalty = max(self.min_kl_penalty, min(self.max_kl_penalty, self.kl_penalty))
        
        if self.verbose and abs(self.kl_penalty - old_penalty) > 1e-6:
            logger.info(
                f"KL penalty adjusted: {old_penalty:.4f} -> {self.kl_penalty:.4f} "
                f"(observed KL: {observed_kl:.4f}, target: {self.target_kl:.4f})"
            )
        
        self.total_updates += 1
        return self.kl_penalty
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get monitoring statistics.
        
        Returns:
            Dictionary with statistics:
            - current_kl: Most recent KL divergence
            - mean_kl: Average KL over all steps
            - current_penalty: Current β value
            - mean_penalty: Average β over all steps
            - total_updates: Number of penalty updates
            - steps: Total monitoring steps
        """
        if len(self.kl_history) == 0:
            return {
                "current_kl": 0.0,
                "mean_kl": 0.0,
                "current_penalty": self.kl_penalty,
                "mean_penalty": self.kl_penalty,
                "total_updates": self.total_updates,
                "steps": self.step_count
            }
        
        return {
            "current_kl": self.kl_history[-1],
            "mean_kl": sum(self.kl_history) / len(self.kl_history),
            "current_penalty": self.kl_penalty,
            "mean_penalty": sum(self.penalty_history) / len(self.penalty_history),
            "total_updates": self.total_updates,
            "steps": self.step_count
        }
    
    def reset(self):
        """Reset monitoring statistics."""
        self.step_count = 0
        self.kl_history = []
        self.penalty_history = []
        self.total_updates = 0


