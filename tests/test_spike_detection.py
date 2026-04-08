"""
Tests for the SpikeDetector — EMA baseline tracking and spike detection.
"""

import math
import pytest
from stabilityguard.core.spike_detector import SpikeDetector, SpikeInfo


class TestEMABaseline:
    """Verify EMA baselines are updated correctly."""

    def test_initial_baseline_set_to_first_value(self):
        detector = SpikeDetector(threshold=10.0, warmup_steps=0)
        norms = {"layer1": 5.0, "layer2": 3.0}

        detector.check(norms, set())

        assert detector.baselines["layer1"] == pytest.approx(5.0, rel=0.01)
        assert detector.baselines["layer2"] == pytest.approx(3.0, rel=0.01)

    def test_ema_updates_slowly(self):
        detector = SpikeDetector(threshold=10.0, ema_alpha=0.01, warmup_steps=0)

        # First step: baseline = 1.0
        detector.check({"layer1": 1.0}, set())
        assert detector.baselines["layer1"] == pytest.approx(1.0, rel=0.01)

        # Second step: value = 2.0, baseline should barely move
        detector.check({"layer1": 2.0}, set())
        expected = 0.99 * 1.0 + 0.01 * 2.0  # 1.01
        assert detector.baselines["layer1"] == pytest.approx(expected, rel=0.01)

    def test_ema_converges_over_many_steps(self):
        detector = SpikeDetector(threshold=10.0, ema_alpha=0.1, warmup_steps=0)

        # Run 100 steps at norm=1.0, then switch to norm=5.0
        for _ in range(100):
            detector.check({"layer1": 1.0}, set())

        assert detector.baselines["layer1"] == pytest.approx(1.0, rel=0.05)

        # Now run 100 steps at norm=5.0 — baseline should converge
        for _ in range(100):
            detector.check({"layer1": 5.0}, set())

        assert detector.baselines["layer1"] == pytest.approx(5.0, rel=0.1)


class TestSpikeDetection:
    """Verify spike detection triggers correctly."""

    def test_no_spike_on_normal_gradients(self):
        detector = SpikeDetector(threshold=10.0, warmup_steps=2)

        # Warmup
        for i in range(5):
            spike, info, _ = detector.check({"layer1": 1.0}, set())

        # Normal gradient — should not trigger
        spike, info, _ = detector.check({"layer1": 1.5}, set())
        assert spike is False
        assert info is None

    def test_spike_triggers_on_threshold_exceeded(self):
        detector = SpikeDetector(threshold=5.0, warmup_steps=2)

        # Warmup with normal values
        for _ in range(5):
            detector.check({"layer1": 1.0}, set())

        # Spike: 50x the baseline
        spike, info, _ = detector.check({"layer1": 50.0}, set())
        assert spike is True
        assert info is not None
        assert info.layer == "layer1"
        assert info.ratio > 5.0

    def test_warmup_suppresses_spikes(self):
        detector = SpikeDetector(threshold=5.0, warmup_steps=10)

        # During warmup — even large values should not trigger
        for i in range(10):
            spike, info, _ = detector.check({"layer1": 100.0 * (i + 1)}, set())
            assert spike is False

    def test_multiple_layers_worst_spike_selected(self):
        detector = SpikeDetector(threshold=5.0, warmup_steps=2)

        # Warmup
        for _ in range(5):
            detector.check({"layer1": 1.0, "layer2": 1.0}, set())

        # Both layers spike, but layer2 is worse
        spike, info, _ = detector.check({"layer1": 10.0, "layer2": 100.0}, set())
        assert spike is True
        assert info.layer == "layer2"
        assert info.ratio > 5.0  # above threshold

    def test_no_spike_below_threshold(self):
        detector = SpikeDetector(threshold=10.0, warmup_steps=2)

        # Warmup
        for _ in range(3):
            detector.check({"layer1": 1.0}, set())

        # 3x is below 10x threshold
        spike, info, _ = detector.check({"layer1": 3.0}, set())
        assert spike is False

    def test_reset_clears_state(self):
        detector = SpikeDetector(threshold=10.0, warmup_steps=0)
        detector.check({"layer1": 1.0}, set())
        assert len(detector.baselines) > 0
        assert detector.step_count > 0

        detector.reset()
        assert len(detector.baselines) == 0
        assert detector.step_count == 0
