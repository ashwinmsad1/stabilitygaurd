"""
Integration tests for GuardedOptimizer.

Uses small synthetic models to verify end-to-end behavior:
spike detection, action execution, normal training passthrough.
"""

import pytest
import math
import torch
import torch.nn as nn
from pathlib import Path

from stabilityguard.core.guarded_optimizer import GuardedOptimizer
from stabilityguard.core.actions import GradientSpikeError


class TinyMLP(nn.Module):
    """Minimal 2-layer MLP for testing."""

    def __init__(self, in_dim=8, hidden=16, out_dim=4):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden, out_dim)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


class TestNormalTraining:
    """Verify GuardedOptimizer doesn't interfere with normal training."""

    def test_basic_training_step(self):
        model = TinyMLP()
        base_opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        optimizer = GuardedOptimizer(
            base_opt, model, spike_threshold=100.0, warmup_steps=0
        )

        # Save initial weights
        w_before = model.fc1.weight.data.clone()

        # Normal forward-backward-step
        x = torch.randn(4, 8)
        loss = model(x).sum()
        loss.backward()
        optimizer.step(loss=loss.item())
        optimizer.zero_grad()

        # Weights should have changed
        assert not torch.equal(w_before, model.fc1.weight.data)

        optimizer.close()

    def test_multiple_normal_steps(self):
        model = TinyMLP()
        base_opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        optimizer = GuardedOptimizer(
            base_opt, model, spike_threshold=100.0, warmup_steps=0
        )

        losses = []
        for _ in range(20):
            x = torch.randn(4, 8)
            target = torch.randn(4, 4)
            loss = nn.functional.mse_loss(model(x), target)
            loss.backward()
            optimizer.step(loss=loss.item())
            optimizer.zero_grad()
            losses.append(loss.item())

        # Loss should generally decrease (or at least not explode)
        assert all(math.isfinite(l) for l in losses)
        assert optimizer.step_count == 20
        assert optimizer.total_spikes == 0

        optimizer.close()

    def test_optimizer_proxies(self):
        model = TinyMLP()
        base_opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        optimizer = GuardedOptimizer(base_opt, model, spike_threshold=10.0)

        # param_groups should be accessible
        assert len(optimizer.param_groups) == 1
        assert optimizer.param_groups[0]["lr"] == 1e-3

        # state_dict / load_state_dict should work
        sd = optimizer.state_dict()
        optimizer.load_state_dict(sd)

        # defaults should be accessible
        assert "lr" in optimizer.defaults

        # repr should work
        r = repr(optimizer)
        assert "GuardedOptimizer" in r
        assert "Adam" in r

        optimizer.close()


class TestSpikeSkip:
    """Verify spike detection causes step to be skipped."""

    def test_nan_gradient_skips_step(self, tmp_path):
        model = TinyMLP()
        base_opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        optimizer = GuardedOptimizer(
            base_opt,
            model,
            spike_threshold=10.0,
            nan_action="skip",
            log_dir=str(tmp_path),
            warmup_steps=0,
        )

        # Normal step first
        x = torch.randn(4, 8)
        loss = model(x).sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Save weights before NaN step
        w_before = model.fc1.weight.data.clone()

        # Forward + backward
        x = torch.randn(4, 8)
        loss = model(x).sum()
        loss.backward()

        # Inject NaN gradient
        model.fc1.weight.grad.fill_(float("nan"))

        # Step should be skipped
        optimizer.step(loss=loss.item())
        optimizer.zero_grad()

        # Weights should NOT have changed
        assert torch.equal(w_before, model.fc1.weight.data)
        assert optimizer.total_spikes == 1
        assert optimizer.total_skips == 1

        # Spike report JSON should exist
        report_files = list(tmp_path.glob("spike_step*.json"))
        assert len(report_files) > 0

        optimizer.close()

    def test_large_spike_skips_step(self, tmp_path):
        model = TinyMLP()
        base_opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        optimizer = GuardedOptimizer(
            base_opt,
            model,
            spike_threshold=5.0,
            nan_action="skip",
            log_dir=str(tmp_path),
            warmup_steps=2,
        )

        # Warmup steps with normal gradients
        for _ in range(5):
            x = torch.randn(4, 8)
            loss = model(x).sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Save weights
        w_before = model.fc1.weight.data.clone()

        # Forward + backward with massive gradient
        x = torch.randn(4, 8)
        loss = model(x).sum()
        loss.backward()
        model.fc1.weight.grad.mul_(1000.0)  # 1000x spike

        optimizer.step(loss=loss.item())
        optimizer.zero_grad()

        # Weights should not have changed
        assert torch.equal(w_before, model.fc1.weight.data)
        assert optimizer.total_spikes >= 1

        optimizer.close()


class TestSpikeRaise:
    """Verify raise action throws GradientSpikeError."""

    def test_nan_gradient_raises(self, tmp_path):
        model = TinyMLP()
        base_opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        optimizer = GuardedOptimizer(
            base_opt,
            model,
            spike_threshold=10.0,
            nan_action="raise",
            log_dir=str(tmp_path),
            warmup_steps=0,
        )

        # Normal step first
        x = torch.randn(4, 8)
        loss = model(x).sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Forward + backward with NaN
        x = torch.randn(4, 8)
        loss = model(x).sum()
        loss.backward()
        model.fc1.weight.grad.fill_(float("nan"))

        with pytest.raises(GradientSpikeError) as exc_info:
            optimizer.step()

        assert "spike" in str(exc_info.value).lower()

        optimizer.close()


class TestSpikeRollback:
    """Verify rollback action restores model state."""

    def test_rollback_restores_weights(self, tmp_path):
        model = TinyMLP()
        base_opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        optimizer = GuardedOptimizer(
            base_opt,
            model,
            spike_threshold=10.0,
            nan_action="rollback",
            log_dir=str(tmp_path),
            warmup_steps=0,
        )

        # Do a couple normal steps (rollback needs a checkpoint)
        for _ in range(3):
            x = torch.randn(4, 8)
            loss = model(x).sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Weights after step 3 (the checkpoint state)
        w_checkpoint = model.fc1.weight.data.clone()

        # Do one more normal step → new checkpoint
        x = torch.randn(4, 8)
        loss = model(x).sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        w_after_step4 = model.fc1.weight.data.clone()

        # Now spike → should rollback to step 4's checkpoint
        x = torch.randn(4, 8)
        loss = model(x).sum()
        loss.backward()
        model.fc1.weight.grad.fill_(float("nan"))
        optimizer.step()
        optimizer.zero_grad()

        # After rollback, weights should equal the checkpoint (step 4 state)
        assert torch.allclose(model.fc1.weight.data, w_after_step4, atol=1e-6)

        optimizer.close()
