"""
RLHF Stability Guard - Main interface for RLHF training monitoring.

Combines all RLHF-specific monitors into a unified interface.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Any
import logging

from .kl_monitor import KLDivergenceMonitor
from .reward_collapse import RewardCollapseDetector
from .value_divergence import ValueDivergenceMonitor
from .ppo_ratio import PPORatioMonitor

logger = logging.getLogger(__name__)


class RLHFStabilityGuard:
    """
    Unified stability monitoring for RLHF training.
    
    Combines multiple monitors:
    - KL divergence monitoring (policy vs reference)
    - Reward model collapse detection
    - Value function divergence monitoring
    - PPO ratio monitoring
    
    Args:
        policy_model: Current policy model (π_θ)
        ref_model: Reference policy model (π_ref)
        value_model: Value function / critic (V)
        reward_model: Reward model (R)
        target_kl: Target KL divergence (default: 0.1)
        enable_kl_adaptation: Enable automatic KL penalty adjustment (default: True)
        enable_reward_collapse_detection: Enable reward collapse detection (default: True)
        enable_value_divergence_monitoring: Enable value divergence monitoring (default: True)
        enable_ppo_ratio_monitoring: Enable PPO ratio monitoring (default: True)
        clip_range: PPO clip range (default: 0.2)
    
    Example:
        >>> guard = RLHFStabilityGuard(
        ...     policy_model=policy,
        ...     ref_model=reference,
        ...     value_model=critic,
        ...     reward_model=reward_model
        ... )
        >>> 
        >>> # During training
        >>> stability_report = guard.check_stability(
        ...     policy_logprobs=policy_logprobs,
        ...     ref_logprobs=ref_logprobs,
        ...     rewards=rewards,
        ...     values=values,
        ...     returns=returns,
        ...     advantages=advantages,
        ...     old_logprobs=old_logprobs
        ... )
        >>> 
        >>> if stability_report['critical_issues']:
        >>>     print("WARNING: Critical stability issues detected!")
        >>>     # Take action (reduce LR, stop training, etc.)
    """
    
    def __init__(
        self,
        policy_model: Optional[nn.Module] = None,
        ref_model: Optional[nn.Module] = None,
        value_model: Optional[nn.Module] = None,
        reward_model: Optional[nn.Module] = None,
        target_kl: float = 0.1,
        enable_kl_adaptation: bool = True,
        enable_reward_collapse_detection: bool = True,
        enable_value_divergence_monitoring: bool = True,
        enable_ppo_ratio_monitoring: bool = True,
        clip_range: float = 0.2,
        # Backward compatibility aliases
        model: Optional[nn.Module] = None,
        kl_threshold: Optional[float] = None,
    ):
        # Handle backward compatibility
        if model is not None and policy_model is None:
            policy_model = model
        if kl_threshold is not None and target_kl == 0.1:  # Only override if default
            target_kl = kl_threshold
        
        # Validate required parameters
        if policy_model is None:
            raise ValueError("policy_model (or model) is required")
        if ref_model is None:
            raise ValueError("ref_model is required")
        
        self.policy_model = policy_model
        self.ref_model = ref_model
        self.value_model = value_model
        self.reward_model = reward_model
        
        # Initialize monitors
        self.kl_monitor = KLDivergenceMonitor(target_kl=target_kl)
        self.enable_kl_adaptation = enable_kl_adaptation
        
        self.reward_detector = None
        if enable_reward_collapse_detection and reward_model is not None:
            self.reward_detector = RewardCollapseDetector()
        
        self.value_monitor = None
        if enable_value_divergence_monitoring and value_model is not None:
            self.value_monitor = ValueDivergenceMonitor()
        
        self.ppo_monitor = None
        if enable_ppo_ratio_monitoring:
            self.ppo_monitor = PPORatioMonitor(clip_range=clip_range)
        
        logger.info(
            f"RLHFStabilityGuard initialized:\n"
            f"  KL monitoring: enabled\n"
            f"  KL adaptation: {'enabled' if enable_kl_adaptation else 'disabled'}\n"
            f"  Reward collapse detection: {'enabled' if self.reward_detector else 'disabled'}\n"
            f"  Value divergence monitoring: {'enabled' if self.value_monitor else 'disabled'}\n"
            f"  PPO ratio monitoring: {'enabled' if self.ppo_monitor else 'disabled'}"
        )
    def check_step(
        self,
        logits: torch.Tensor,
        ref_logits: torch.Tensor,
        rewards: Optional[torch.Tensor] = None,
        step: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Simplified stability check for backward compatibility.
        
        This is an alias for check_stability() with simplified parameters.
        
        Args:
            logits: Policy model logits [batch, seq_len, vocab]
            ref_logits: Reference model logits [batch, seq_len, vocab]
            rewards: Rewards [batch]
            step: Current training step
            **kwargs: Additional arguments passed to check_stability()
        
        Returns:
            Stability report dictionary
        """
        # Convert logits to log probabilities
        import torch.nn.functional as F
        policy_logprobs = F.log_softmax(logits, dim=-1)
        ref_logprobs = F.log_softmax(ref_logits, dim=-1)
        
        return self.check_stability(
            policy_logprobs=policy_logprobs,
            ref_logprobs=ref_logprobs,
            rewards=rewards,
            **kwargs
        )
    
    
    def check_stability(
        self,
        policy_logprobs: torch.Tensor,
        ref_logprobs: torch.Tensor,
        rewards: Optional[torch.Tensor] = None,
        values: Optional[torch.Tensor] = None,
        returns: Optional[torch.Tensor] = None,
        advantages: Optional[torch.Tensor] = None,
        old_logprobs: Optional[torch.Tensor] = None,
        reward_grads: Optional[torch.Tensor] = None,
        value_grads: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive stability check for RLHF training.
        
        Args:
            policy_logprobs: Log probs from current policy [batch, seq_len]
            ref_logprobs: Log probs from reference model [batch, seq_len]
            rewards: Rewards from reward model [batch] (optional)
            values: Value predictions [batch] (optional)
            returns: Actual returns [batch] (optional)
            advantages: Advantage estimates [batch] (optional)
            old_logprobs: Log probs from old policy [batch, seq_len] (optional)
            reward_grads: Gradients of reward model (optional)
            value_grads: Gradients of value model (optional)
            mask: Mask for valid tokens [batch, seq_len] (optional)
        
        Returns:
            Comprehensive stability report with:
            - kl_stats: KL divergence statistics
            - kl_explosion: Whether KL explosion detected
            - reward_collapse: Whether reward collapse detected
            - value_divergence: Whether value divergence detected
            - ppo_ratio_alert: Whether PPO ratio alert triggered
            - critical_issues: List of critical issues
            - warnings: List of warnings
            - recommended_actions: List of recommended actions
        """
        report = {
            "kl_stats": {},
            "kl_explosion": False,
            "reward_collapse": False,
            "value_divergence": False,
            "ppo_ratio_alert": False,
            "critical_issues": [],
            "warnings": [],
            "recommended_actions": [],
        }
        
        # 1. Check KL divergence
        kl_div, kl_stats = self.kl_monitor.compute_kl(
            policy_logprobs, ref_logprobs, mask
        )
        report["kl_divergence"] = kl_div.item() if isinstance(kl_div, torch.Tensor) else kl_div
        report["kl_stats"] = kl_stats
        
        # Check for KL explosion (> 10x target)
        if kl_stats["mean_kl"] > self.kl_monitor.target_kl * 10:
            report["kl_explosion"] = True
            report["critical_issues"].append("KL divergence explosion")
            report["recommended_actions"].append("Increase KL penalty (β)")
            
            logger.warning(
                f"KL EXPLOSION: {kl_stats['mean_kl']:.4f} "
                f"(target: {self.kl_monitor.target_kl:.4f})"
            )
        
        # Adjust KL penalty if enabled
        if self.enable_kl_adaptation:
            new_beta = self.kl_monitor.update_penalty(kl_stats["mean_kl"])
            report["kl_penalty"] = new_beta
        
        # 2. Check reward model collapse
        if self.reward_detector is not None and rewards is not None:
            reward_collapsed = self.reward_detector.detect_collapse(
                rewards, reward_grads
            )
            report["reward_collapse"] = reward_collapsed
            
            if reward_collapsed:
                report["critical_issues"].append("Reward model collapse")
                report["recommended_actions"].append("Stop training and investigate reward model")
        
        # 3. Check value function divergence
        if self.value_monitor is not None and values is not None and returns is not None:
            value_diverged = self.value_monitor.check_divergence(
                values, returns, advantages, value_grads
            )
            report["value_divergence"] = value_diverged
            
            if value_diverged:
                report["critical_issues"].append("Value function divergence")
                report["recommended_actions"].append("Reduce critic learning rate")
        
        # 4. Check PPO ratio
        if self.ppo_monitor is not None and old_logprobs is not None:
            ppo_stats = self.ppo_monitor.check_ratio(
                policy_logprobs, old_logprobs
            )
            report["ppo_stats"] = ppo_stats
            report["ppo_ratio_alert"] = ppo_stats["alert_triggered"]
            
            if ppo_stats["alert_triggered"]:
                report["warnings"].append("PPO ratio instability")
                report["recommended_actions"].append("Reduce policy learning rate")
        
        # Add general warnings
        if kl_stats["mean_kl"] > self.kl_monitor.target_kl * 2:
            report["warnings"].append(f"KL divergence high: {kl_stats['mean_kl']:.4f}")
        
        return report
    
    def adjust_kl_penalty(self) -> float:
        """
        Manually adjust KL penalty based on current KL.
        
        Returns:
            New KL penalty coefficient β
        """
        return self.kl_monitor.kl_penalty
    
    def get_kl_penalty(self) -> float:
        """Get current KL penalty coefficient β."""
        return self.kl_monitor.kl_penalty
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics from all monitors.
        
        Returns:
            Dictionary with stats from all enabled monitors
        """
        stats = {
            "kl_monitor": self.kl_monitor.get_statistics(),
        }
        
        if self.reward_detector is not None:
            stats["reward_detector"] = self.reward_detector.get_stats()
        
        if self.value_monitor is not None:
            stats["value_monitor"] = self.value_monitor.get_stats()
        
        if self.ppo_monitor is not None:
            stats["ppo_monitor"] = self.ppo_monitor.get_stats()
        
        return stats
    
    def reset_all(self):
        """Reset all monitors."""
        self.kl_monitor.reset()
        
        if self.reward_detector is not None:
            self.reward_detector.reset()
        
        if self.value_monitor is not None:
            self.value_monitor.reset()
        
        if self.ppo_monitor is not None:
            self.ppo_monitor.reset()
        
        logger.info("All RLHF monitors reset")


