"""
Unit tests for distributed training components.
"""

import torch
import pytest
from unittest.mock import Mock, patch


class TestDistributedSpikeDetector:
    """Tests for Distributed Spike Detector."""
    
    @patch('torch.distributed.is_initialized', return_value=True)
    @patch('torch.distributed.get_rank', return_value=0)
    @patch('torch.distributed.get_world_size', return_value=4)
    def test_initialization(self, mock_world_size, mock_rank, mock_init):
        """Test detector initialization."""
        from stabilityguard.distributed import DistributedSpikeDetector
        
        detector = DistributedSpikeDetector(
            world_size=4,
            rank=0,
            spike_threshold=10.0
        )
        
        assert detector.world_size == 4
        assert detector.rank == 0
        assert detector.spike_threshold == 10.0
    
    @patch('torch.distributed.is_initialized', return_value=True)
    @patch('torch.distributed.get_rank', return_value=0)
    @patch('torch.distributed.get_world_size', return_value=4)
    @patch('torch.distributed.all_gather_into_tensor')
    def test_spike_detection(self, mock_all_gather, mock_world_size, mock_rank, mock_init):
        """Test spike detection across ranks."""
        from stabilityguard.distributed import DistributedSpikeDetector
        
        detector = DistributedSpikeDetector(world_size=4, rank=0)
        
        # Mock all-gather to return norms from all ranks
        def mock_gather(output, input):
            output.copy_(torch.tensor([5.0, 5.0, 15.0, 5.0]))  # Rank 2 spiked
        
        mock_all_gather.side_effect = mock_gather
        
        result = detector.detect_distributed_spike(
            local_grad_norm=5.0,
            threshold=10.0
        )
        
        assert result['spike_detected'] == True
        assert result['spike_rank'] == 2


class TestFSDPStabilityGuard:
    """Tests for FSDP Stability Guard."""
    
    def test_initialization_without_fsdp(self):
        """Test that guard requires FSDP model."""
        from stabilityguard.distributed import FSDPStabilityGuard
        
        model = torch.nn.Linear(10, 10)
        
        with pytest.raises(TypeError):
            guard = FSDPStabilityGuard(model)


class TestDeepSpeedStabilityGuard:
    """Tests for DeepSpeed Stability Guard."""
    
    def test_initialization_without_deepspeed(self):
        """Test that guard requires DeepSpeed engine."""
        from stabilityguard.distributed import DeepSpeedStabilityGuard
        
        model = torch.nn.Linear(10, 10)
        
        with pytest.raises((TypeError, ImportError)):
            guard = DeepSpeedStabilityGuard(model)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
