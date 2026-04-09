"""
Unit tests for advanced logging components.
"""

import torch
import pytest
from stabilityguard.logging import (
    GradientFlowTracker,
    ActivationStatsLogger,
    WeightUpdateTracker,
    CheckpointHealthScorer,
    AdvancedLogger
)


class TestGradientFlowTracker:
    """Tests for Gradient Flow Tracker."""
    
    def test_initialization(self):
        """Test tracker initialization."""
        tracker = GradientFlowTracker(track_frequency=1)
        assert tracker.track_frequency == 1
        assert tracker.step_count == 0
    
    def test_flow_tracking(self):
        """Test gradient flow tracking."""
        tracker = GradientFlowTracker(track_frequency=1)
        model = torch.nn.Linear(10, 10)
        
        # Create dummy gradients
        for param in model.parameters():
            param.grad = torch.randn_like(param)
        
        flow_data = tracker.track_flow(model)
        
        assert "step" in flow_data
        assert "layer_norms" in flow_data
        assert len(flow_data["layer_norms"]) > 0


class TestActivationStatsLogger:
    """Tests for Activation Stats Logger."""
    
    def test_initialization(self):
        """Test logger initialization."""
        logger = ActivationStatsLogger(track_frequency=10)
        assert logger.track_frequency == 10
        assert logger.step_count == 0
    
    def test_hook_registration(self):
        """Test hook registration."""
        logger = ActivationStatsLogger()
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 10),
            torch.nn.ReLU(),
            torch.nn.Linear(10, 5)
        )
        
        logger.register_hooks(model)
        assert len(logger.hooks) > 0
        
        logger.remove_hooks()
        assert len(logger.hooks) == 0


class TestWeightUpdateTracker:
    """Tests for Weight Update Tracker."""
    
    def test_initialization(self):
        """Test tracker initialization."""
        tracker = WeightUpdateTracker(track_frequency=1)
        assert tracker.track_frequency == 1
        assert tracker.step_count == 0
    
    def test_update_tracking(self):
        """Test weight update tracking."""
        tracker = WeightUpdateTracker(track_frequency=1)
        model = torch.nn.Linear(10, 10)
        
        # Save initial state
        prev_state = {n: p.data.clone() for n, p in model.named_parameters()}
        
        # Modify weights
        for param in model.parameters():
            param.data += torch.randn_like(param) * 0.01
        
        # Track updates
        updates = tracker.track_updates(model, prev_state)
        
        assert len(updates) > 0
        for name, update_norm in updates.items():
            assert update_norm >= 0


class TestCheckpointHealthScorer:
    """Tests for Checkpoint Health Scorer."""
    
    def test_initialization(self):
        """Test scorer initialization."""
        scorer = CheckpointHealthScorer(history_size=10)
        assert scorer.history_size == 10
    
    def test_checkpoint_scoring(self):
        """Test checkpoint scoring."""
        scorer = CheckpointHealthScorer()
        
        # Create dummy history
        history = [
            {"grad_norm": 5.0, "loss": 2.0, "spike_detected": False},
            {"grad_norm": 5.5, "loss": 1.8, "spike_detected": False},
            {"grad_norm": 6.0, "loss": 1.5, "spike_detected": False},
        ]
        
        score = scorer.score_checkpoint("checkpoint.pt", history)
        
        assert 0 <= score <= 100
    
    def test_scoring_with_spikes(self):
        """Test scoring with spike history."""
        scorer = CheckpointHealthScorer()
        
        # History with spikes
        history = [
            {"grad_norm": 5.0, "loss": 2.0, "spike_detected": True},
            {"grad_norm": 50.0, "loss": 5.0, "spike_detected": True},
            {"grad_norm": 5.0, "loss": 2.0, "spike_detected": False},
        ]
        
        score = scorer.score_checkpoint("checkpoint.pt", history)
        
        # Score should be lower due to spikes
        assert score < 100


class TestAdvancedLogger:
    """Tests for Advanced Logger."""
    
    def test_initialization(self):
        """Test logger initialization."""
        logger = AdvancedLogger(
            log_dir="./test_logs",
            enable_gradient_flow=True,
            enable_weight_updates=True
        )
        
        assert logger.gradient_flow is not None
        assert logger.weight_updates is not None
    
    def test_log_step(self):
        """Test step logging."""
        logger = AdvancedLogger(enable_gradient_flow=True)
        model = torch.nn.Linear(10, 10)
        
        # Create dummy gradients
        for param in model.parameters():
            param.grad = torch.randn_like(param)
        
        log_data = logger.log_step(
            step=1,
            loss=0.5,
            model=model
        )
        
        assert "step" in log_data
        assert "loss" in log_data
        assert log_data["step"] == 1
        assert log_data["loss"] == 0.5
    
    def test_comprehensive_stats(self):
        """Test comprehensive statistics."""
        logger = AdvancedLogger()
        
        stats = logger.get_comprehensive_stats()
        
        assert isinstance(stats, dict)
        assert "checkpoint_scorer" in stats


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
