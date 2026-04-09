"""
Unit tests for RLHF monitoring components.
"""

import torch
import pytest
from stabilityguard.rlhf import (
    KLDivergenceMonitor,
    RewardCollapseDetector,
    ValueDivergenceMonitor,
    PPORatioMonitor,
    RLHFStabilityGuard
)


class TestKLDivergenceMonitor:
    """Tests for KL Divergence Monitor."""
    
    def test_initialization(self):
        """Test monitor initialization."""
        monitor = KLDivergenceMonitor(target_kl=0.1)
        assert monitor.target_kl == 0.1
        assert monitor.kl_penalty > 0
    
    def test_compute_kl(self):
        """Test KL computation."""
        monitor = KLDivergenceMonitor()
        
        # Create dummy log probs
        policy_logprobs = torch.randn(4, 10, 100)  # [batch, seq, vocab]
        ref_logprobs = torch.randn(4, 10, 100)
        
        kl, stats = monitor.compute_kl(policy_logprobs, ref_logprobs)
        
        assert isinstance(kl, torch.Tensor)
        assert kl.dim() == 0  # Scalar
        assert not torch.isnan(kl).any()
        assert "mean_kl" in stats
    
    def test_kl_stats(self):
        """Test KL statistics computation."""
        monitor = KLDivergenceMonitor()
        
        policy_logprobs = torch.randn(4, 10, 100)
        ref_logprobs = torch.randn(4, 10, 100)
        
        kl, stats = monitor.compute_kl(policy_logprobs, ref_logprobs)
        
        assert "mean_kl" in stats
        assert "max_kl" in stats
        assert "min_kl" in stats
        assert stats["mean_kl"] >= 0
    
    def test_explosion_detection(self):
        """Test KL explosion detection."""
        monitor = KLDivergenceMonitor(target_kl=0.1)
        
        # Build baseline with normal KL
        for _ in range(10):
            policy_logprobs = torch.randn(4, 10, 100) * 0.1
            ref_logprobs = torch.randn(4, 10, 100) * 0.1
            kl, stats = monitor.compute_kl(policy_logprobs, ref_logprobs)
        
        # Check that KL is being tracked
        assert len(monitor.kl_history) == 10
        
        # Simulate explosion with very different distributions
        policy_logprobs = torch.randn(4, 10, 100) * 10.0
        ref_logprobs = torch.randn(4, 10, 100) * 0.1
        kl, stats = monitor.compute_kl(policy_logprobs, ref_logprobs)
        
        # High KL should be detected
        assert stats["mean_kl"] > monitor.target_kl
    
    def test_penalty_adjustment(self):
        """Test KL penalty adjustment."""
        monitor = KLDivergenceMonitor(target_kl=0.1, adaptive=True)
        initial_penalty = monitor.kl_penalty
        
        # High KL should increase penalty
        new_penalty = monitor.update_penalty(0.5)
        assert new_penalty > initial_penalty
        assert monitor.kl_penalty == new_penalty
        
        # Low KL should decrease penalty
        monitor.kl_penalty = 0.1
        new_penalty = monitor.update_penalty(0.01)
        assert new_penalty < 0.1


class TestRewardCollapseDetector:
    """Tests for Reward Collapse Detector."""
    
    def test_initialization(self):
        """Test detector initialization."""
        detector = RewardCollapseDetector()
        assert detector.variance_threshold > 0
        assert detector.collapse_detected == False
    
    def test_variance_computation(self):
        """Test reward variance computation."""
        detector = RewardCollapseDetector()
        
        # Diverse rewards
        rewards = torch.randn(100)
        variance = detector.compute_reward_variance(rewards)
        assert variance > 0
        
        # Collapsed rewards (all same)
        rewards = torch.ones(100)
        variance = detector.compute_reward_variance(rewards)
        assert variance < 0.01
    
    def test_entropy_computation(self):
        """Test reward entropy computation."""
        detector = RewardCollapseDetector()
        
        # Diverse rewards
        rewards = torch.randn(100)
        entropy = detector.compute_reward_entropy(rewards)
        assert entropy > 0
    
    def test_collapse_detection(self):
        """Test collapse detection."""
        detector = RewardCollapseDetector(
            variance_threshold=0.01,
            collapse_patience=2
        )
        
        # Feed collapsed rewards
        for _ in range(3):
            rewards = torch.ones(100)  # All same
            collapsed = detector.detect_collapse(rewards)
        
        assert detector.collapse_detected


class TestValueDivergenceMonitor:
    """Tests for Value Divergence Monitor."""
    
    def test_initialization(self):
        """Test monitor initialization."""
        monitor = ValueDivergenceMonitor()
        assert monitor.td_error_threshold > 0
        assert monitor.divergence_detected == False
    
    def test_value_error_computation(self):
        """Test value error computation."""
        monitor = ValueDivergenceMonitor()
        
        values = torch.randn(100)
        returns = torch.randn(100)
        
        error = monitor.compute_value_error(values, returns)
        assert error.shape == values.shape
    
    def test_divergence_detection(self):
        """Test divergence detection."""
        monitor = ValueDivergenceMonitor(
            td_error_threshold=1.0,
            divergence_patience=2
        )
        
        # Feed high errors
        for _ in range(3):
            values = torch.randn(100)
            returns = values + torch.randn(100) * 10  # Large error
            advantages = torch.randn(100)
            
            diverged = monitor.check_divergence(values, returns, advantages)
        
        assert monitor.divergence_detected


class TestPPORatioMonitor:
    """Tests for PPO Ratio Monitor."""
    
    def test_initialization(self):
        """Test monitor initialization."""
        monitor = PPORatioMonitor(clip_range=0.2)
        assert monitor.clip_range == 0.2
        assert monitor.alert_triggered == False
    
    def test_ratio_computation(self):
        """Test ratio computation."""
        monitor = PPORatioMonitor()
        
        policy_logprobs = torch.randn(100)
        old_logprobs = torch.randn(100)
        
        ratio = monitor.compute_ratio(policy_logprobs, old_logprobs)
        assert ratio.shape == policy_logprobs.shape
        assert (ratio > 0).all()
    
    def test_clipping_frequency(self):
        """Test clipping frequency computation."""
        monitor = PPORatioMonitor(clip_range=0.2)
        
        # All ratios within clip range
        ratio = torch.ones(100)
        freq = monitor.compute_clipping_frequency(ratio)
        assert freq == 0.0
        
        # All ratios outside clip range
        ratio = torch.ones(100) * 2.0
        freq = monitor.compute_clipping_frequency(ratio)
        assert freq == 1.0
    
    def test_extreme_ratio_detection(self):
        """Test extreme ratio detection."""
        monitor = PPORatioMonitor(extreme_ratio_threshold=10.0)
        
        # Normal ratios
        ratio = torch.ones(100)
        assert not monitor.is_extreme_ratio(ratio)
        
        # Extreme ratios
        ratio = torch.ones(100) * 15.0
        assert monitor.is_extreme_ratio(ratio)


class TestRLHFStabilityGuard:
    """Tests for RLHF Stability Guard."""
    
    def test_initialization(self):
        """Test guard initialization."""
        # Create dummy models
        policy = torch.nn.Linear(10, 10)
        ref = torch.nn.Linear(10, 10)
        value = torch.nn.Linear(10, 1)
        reward = torch.nn.Linear(10, 1)
        
        guard = RLHFStabilityGuard(
            policy_model=policy,
            ref_model=ref,
            value_model=value,
            reward_model=reward
        )
        
        assert guard.kl_monitor is not None
        assert guard.reward_detector is not None
    
    def test_stability_check(self):
        """Test comprehensive stability check."""
        policy = torch.nn.Linear(10, 10)
        ref = torch.nn.Linear(10, 10)
        value = torch.nn.Linear(10, 1)
        reward = torch.nn.Linear(10, 1)
        
        guard = RLHFStabilityGuard(
            policy_model=policy,
            ref_model=ref,
            value_model=value,
            reward_model=reward
        )
        
        # Create dummy data
        policy_logprobs = torch.randn(4, 10)
        ref_logprobs = torch.randn(4, 10)
        rewards = torch.randn(4)
        values = torch.randn(4)
        returns = torch.randn(4)
        advantages = torch.randn(4)
        old_logprobs = torch.randn(4, 10)
        
        report = guard.check_stability(
            policy_logprobs=policy_logprobs,
            ref_logprobs=ref_logprobs,
            rewards=rewards,
            values=values,
            returns=returns,
            advantages=advantages,
            old_logprobs=old_logprobs
        )
        
        assert "kl_stats" in report
        assert "kl_explosion" in report
        assert "critical_issues" in report
        assert isinstance(report["critical_issues"], list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

