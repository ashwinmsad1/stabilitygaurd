"""
RLHF (Reinforcement Learning from Human Feedback) stability monitoring.

This module provides specialized monitoring for RLHF training, including:
- Reward model collapse detection
- Value function divergence monitoring
- PPO ratio monitoring
- Unified RLHF guard
"""

from .kl_monitor import KLDivergenceMonitor
from .reward_collapse import RewardCollapseDetector
from .value_divergence import ValueDivergenceMonitor
from .ppo_ratio import PPORatioMonitor
from .rlhf_guard import RLHFStabilityGuard

# Alias for backward compatibility
RLHFGuard = RLHFStabilityGuard

__all__ = [
    "KLDivergenceMonitor",
    "RewardCollapseDetector",
    "ValueDivergenceMonitor",
    "PPORatioMonitor",
    "RLHFStabilityGuard",
    "RLHFGuard",  # Alias
]


