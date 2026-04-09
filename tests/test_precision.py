"""
Unit tests for mixed precision components.
"""

import torch
import pytest
from stabilityguard.precision import (
    PrecisionMonitor,
    AdaptiveLossScaler,
    MixedPrecisionGuard
)


class TestPrecisionMonitor:
    """Tests for Precision Monitor."""
    
    def test_initialization(self):
        """Test monitor initialization."""
        monitor = PrecisionMonitor(precision="fp16")
        assert monitor.precision == "fp16"
        assert monitor.loss_scale > 0
    
    def test_invalid_precision(self):
        """Test invalid precision raises error."""
        with pytest.raises(ValueError):
            monitor = PrecisionMonitor(precision="invalid")
    
    def test_overflow_detection(self):
        """Test overflow detection."""
        monitor = PrecisionMonitor()
        
        # Normal gradients
        gradients = [torch.randn(10, 10) for _ in range(3)]
        assert not monitor.check_overflow(gradients)
        
        # Gradients with NaN
        gradients = [torch.randn(10, 10) for _ in range(2)]
        gradients.append(torch.tensor([[float('nan')]]))
        assert monitor.check_overflow(gradients)
        
        # Gradients with Inf
        gradients = [torch.randn(10, 10) for _ in range(2)]
        gradients.append(torch.tensor([[float('inf')]]))
        assert monitor.check_overflow(gradients)
    
    def test_underflow_detection(self):
        """Test underflow detection."""
        monitor = PrecisionMonitor(underflow_threshold=0.5)
        
        # Normal gradients
        gradients = [torch.randn(10, 10) for _ in range(3)]
        assert not monitor.check_underflow(gradients)
        
        # Mostly zeros (underflow)
        gradients = [torch.zeros(10, 10) for _ in range(3)]
        assert monitor.check_underflow(gradients)
    
    def test_gradient_range(self):
        """Test gradient range computation."""
        monitor = PrecisionMonitor()
        
        gradients = [
            torch.tensor([0.001, 0.1, 1.0, 10.0]),
            torch.tensor([0.01, 0.5, 5.0])
        ]
        
        min_val, max_val = monitor.get_gradient_range(gradients)
        
        assert min_val > 0
        assert max_val > min_val
    
    def test_precision_recommendation(self):
        """Test precision recommendation."""
        monitor = PrecisionMonitor(precision="fp16")
        
        # Simulate overflows
        for _ in range(20):
            gradients = [torch.tensor([[float('nan')]])]
            monitor.check_overflow(gradients)
        
        # Should recommend BF16 or FP32
        recommended = monitor.recommend_precision()
        assert recommended in ["bf16", "fp32"]
    
    def test_bf16_switch_recommendation(self):
        """Test BF16 switch recommendation."""
        monitor = PrecisionMonitor(precision="fp16")
        
        # Initially should not recommend switch
        assert not monitor.should_switch_to_bf16()
        
        # After many overflows, should recommend
        for _ in range(20):
            gradients = [torch.tensor([[float('nan')]])]
            monitor.check_overflow(gradients)
        
        assert monitor.should_switch_to_bf16()


class TestAdaptiveLossScaler:
    """Tests for Adaptive Loss Scaler."""
    
    def test_initialization(self):
        """Test scaler initialization."""
        scaler = AdaptiveLossScaler(init_scale=2**16)
        assert scaler.scale == 2**16
        assert scaler.overflow_count == 0
    
    def test_loss_scaling(self):
        """Test loss scaling."""
        scaler = AdaptiveLossScaler(init_scale=1000.0)
        
        loss = torch.tensor(0.5)
        scaled_loss = scaler.scale_loss(loss)
        
        assert scaled_loss.item() == 500.0
    
    def test_overflow_handling(self):
        """Test overflow handling."""
        scaler = AdaptiveLossScaler(init_scale=1000.0, scale_factor=2.0)
        initial_scale = scaler.get_scale()
        
        # Simulate overflow
        scaler.update(overflow=True)
        
        # Scale should decrease
        assert scaler.get_scale() < initial_scale
    
    def test_scale_growth(self):
        """Test scale growth after stable period."""
        scaler = AdaptiveLossScaler(
            init_scale=1000.0,
            scale_window=10,
            scale_factor=2.0
        )
        initial_scale = scaler.get_scale()
        
        # Simulate stable training
        for _ in range(15):
            scaler.update(overflow=False)
        
        # Scale should increase
        assert scaler.get_scale() > initial_scale
    
    def test_conservative_mode(self):
        """Test conservative mode near spikes."""
        scaler = AdaptiveLossScaler(
            init_scale=1000.0,
            scale_window=10,
            conservative_mode=True
        )
        
        # Simulate spike
        scaler.update(overflow=False, spike_detected=True)
        
        # Should be in conservative mode
        assert scaler.spike_detected_recently
        assert scaler.steps_since_spike == 0
    
    def test_state_dict(self):
        """Test state dict save/load."""
        scaler = AdaptiveLossScaler(init_scale=1000.0)
        scaler.update(overflow=True)
        
        state = scaler.state_dict()
        assert "scale" in state
        assert "overflow_count" in state
        
        # Create new scaler and load state
        new_scaler = AdaptiveLossScaler()
        new_scaler.load_state_dict(state)
        
        assert new_scaler.scale == scaler.scale
        assert new_scaler.overflow_count == scaler.overflow_count


class TestMixedPrecisionGuard:
    """Tests for Mixed Precision Guard."""
    
    def test_initialization(self):
        """Test guard initialization."""
        model = torch.nn.Linear(10, 10)
        guard = MixedPrecisionGuard(model, precision="fp16")
        
        assert guard.precision == "fp16"
        assert guard.precision_monitor is not None
        assert guard.loss_scaler is not None
    
    def test_stability_check(self):
        """Test stability check."""
        model = torch.nn.Linear(10, 10)
        guard = MixedPrecisionGuard(model)
        
        # Create dummy gradients
        for param in model.parameters():
            param.grad = torch.randn_like(param)
        
        report = guard.check_stability(model)
        
        assert "overflow_detected" in report
        assert "underflow_detected" in report
        assert "recommend_bf16" in report
        assert "gradient_range" in report
    
    def test_loss_scaling(self):
        """Test loss scaling."""
        model = torch.nn.Linear(10, 10)
        guard = MixedPrecisionGuard(model)
        
        loss = torch.tensor(0.5)
        scaled_loss = guard.scale_loss(loss)
        
        assert scaled_loss.item() > loss.item()
    
    def test_update(self):
        """Test guard update."""
        model = torch.nn.Linear(10, 10)
        guard = MixedPrecisionGuard(model)
        
        # Create dummy gradients
        for param in model.parameters():
            param.grad = torch.randn_like(param)
        
        # Should not raise error
        guard.update(spike_detected=False)
    
    def test_state_dict(self):
        """Test state dict save/load."""
        model = torch.nn.Linear(10, 10)
        guard = MixedPrecisionGuard(model)
        
        state = guard.state_dict()
        assert "precision" in state
        
        # Load state
        new_guard = MixedPrecisionGuard(model)
        new_guard.load_state_dict(state)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
