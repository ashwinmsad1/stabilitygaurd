"""
Integration tests for StabilityGuard v0.3.0.
Tests the interaction between different components.
"""

import torch
import pytest
from stabilityguard.core import GuardedOptimizer
from stabilityguard.rlhf import RLHFGuard
from stabilityguard.precision import MixedPrecisionGuard
from stabilityguard.logging import AdvancedLogger


class TestRLHFIntegration:
    """Integration tests for RLHF components."""
    
    def test_rlhf_with_optimizer(self):
        """Test RLHF guard with guarded optimizer."""
        model = torch.nn.Linear(10, 10)
        ref_model = torch.nn.Linear(10, 10)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        guarded_opt = GuardedOptimizer(
            optimizer,
            model,
            spike_threshold=10.0
        )
        
        rlhf_guard = RLHFGuard(
            model=model,
            ref_model=ref_model,
            kl_threshold=0.1
        )
        
        # Simulate training step
        logits = torch.randn(4, 10)
        ref_logits = torch.randn(4, 10)
        rewards = torch.randn(4)
        
        rlhf_guard.check_step(
            logits=logits,
            ref_logits=ref_logits,
            rewards=rewards,
            step=1
        )
        
        assert rlhf_guard.kl_monitor.step_count == 1


class TestPrecisionIntegration:
    """Integration tests for mixed precision components."""
    
    def test_precision_with_optimizer(self):
        """Test mixed precision guard with optimizer."""
        model = torch.nn.Linear(10, 10)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        guarded_opt = GuardedOptimizer(
            optimizer,
            model,
            spike_threshold=10.0
        )
        
        precision_guard = MixedPrecisionGuard(
            model=model,
            precision="fp16",
            init_scale=2**16
        )
        
        # Simulate training step
        loss = torch.tensor(1.0, requires_grad=True)
        scaled_loss = precision_guard.scale_loss(loss)
        
        assert scaled_loss.item() > loss.item()
        
        # Check stability
        stability_report = precision_guard.check_stability(model=model)
        
        assert isinstance(stability_report, dict)
        assert "overflow_detected" in stability_report
        assert "underflow_detected" in stability_report


class TestLoggingIntegration:
    """Integration tests for logging components."""
    
    def test_logger_with_optimizer(self):
        """Test advanced logger with optimizer."""
        model = torch.nn.Linear(10, 10)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        guarded_opt = GuardedOptimizer(
            optimizer,
            model,
            spike_threshold=10.0
        )
        
        logger = AdvancedLogger(
            enable_gradient_flow=True,
            enable_weight_updates=True
        )
        
        # Create dummy gradients
        for param in model.parameters():
            param.grad = torch.randn_like(param)
        
        # Log step
        log_data = logger.log_step(
            step=1,
            loss=0.5,
            model=model
        )
        
        assert "step" in log_data
        assert "loss" in log_data


class TestFullPipeline:
    """Test complete training pipeline with all components."""
    
    def test_complete_pipeline(self):
        """Test full pipeline with all v0.3.0 features."""
        # Setup models
        model = torch.nn.Linear(10, 10)
        ref_model = torch.nn.Linear(10, 10)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Setup components
        guarded_opt = GuardedOptimizer(
            optimizer,
            model,
            spike_threshold=10.0
        )
        
        rlhf_guard = RLHFGuard(
            model=model,
            ref_model=ref_model,
            kl_threshold=0.1
        )
        
        precision_guard = MixedPrecisionGuard(
            model=model,
            precision="fp16",
            init_scale=2**16
        )
        
        logger = AdvancedLogger(
            enable_gradient_flow=True,
            enable_weight_updates=True
        )
        
        # Simulate training steps
        for step in range(5):
            # Forward pass
            x = torch.randn(4, 10)
            logits = model(x)
            ref_logits = ref_model(x)
            rewards = torch.randn(4)
            
            # Compute loss
            loss = torch.nn.functional.mse_loss(logits, torch.randn(4, 10))
            
            # Scale loss
            scaled_loss = precision_guard.scale_loss(loss)
            
            # Backward pass
            scaled_loss.backward()
            
            # Check RLHF
            rlhf_guard.check_step(
                logits=logits,
                ref_logits=ref_logits,
                rewards=rewards,
                step=step
            )
            
            # Check precision stability
            stability_report = precision_guard.check_stability(model=model)
            
            # Optimizer step
            guarded_opt.step()
            guarded_opt.zero_grad()
            
            # Update precision guard
            precision_guard.update()
            
            # Log
            logger.log_step(
                step=step,
                loss=loss.item(),
                model=model
            )
        
        # Verify all components tracked steps
        assert rlhf_guard.kl_monitor.step_count == 5
        assert precision_guard.precision_monitor.step_count == 5
        assert logger.step_count == 5


class TestErrorHandling:
    """Test error handling across components."""
    
    def test_invalid_precision(self):
        """Test handling of invalid precision."""
        model = torch.nn.Linear(10, 10)
        with pytest.raises(ValueError):
            MixedPrecisionGuard(model=model, precision="invalid")
    
    def test_missing_ref_model(self):
        """Test RLHF without reference model."""
        model = torch.nn.Linear(10, 10)
        
        with pytest.raises(ValueError, match="ref_model is required"):
            RLHFGuard(model=model, kl_threshold=0.1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
