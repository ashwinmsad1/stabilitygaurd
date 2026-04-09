"""
Example: Distributed Training with StabilityGuard

This example demonstrates how to use StabilityGuard's distributed training
features with PyTorch DDP (DistributedDataParallel).
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from stabilityguard.distributed import DistributedGuardedOptimizer


class SimpleModel(nn.Module):
    """Simple model for demonstration."""
    def __init__(self, input_size=784, hidden_size=256, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def setup_distributed(rank, world_size):
    """Initialize distributed training."""
    import os
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Initialize process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup_distributed():
    """Clean up distributed training."""
    dist.destroy_process_group()


def train_distributed(rank, world_size):
    """Training function for each process."""
    print(f"Running on rank {rank}/{world_size}")
    
    # Setup distributed
    setup_distributed(rank, world_size)
    
    # Create model and move to GPU
    model = SimpleModel().to(rank)
    ddp_model = DDP(model, device_ids=[rank])
    
    # Setup optimizer with StabilityGuard
    base_optimizer = AdamW(ddp_model.parameters(), lr=1e-3)
    optimizer = DistributedGuardedOptimizer(
        base_optimizer,
        ddp_model,
        spike_threshold=10.0,
        nan_action="skip",
        backend="nccl"
    )
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    if rank == 0:
        print("=" * 60)
        print("Distributed Training with StabilityGuard")
        print(f"World Size: {world_size}")
        print("=" * 60)
    
    # Training loop
    for step in range(100):
        # Generate dummy batch
        batch_size = 32
        inputs = torch.randn(batch_size, 784).to(rank)
        targets = torch.randint(0, 10, (batch_size,)).to(rank)
        
        # Forward pass
        outputs = ddp_model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass
        loss.backward()
        
        # Optimizer step (with distributed spike detection)
        optimizer.step()
        optimizer.zero_grad()
        
        # Periodic logging (only on rank 0)
        if rank == 0 and step % 10 == 0:
            print(f"\nStep {step}:")
            print(f"  Loss: {loss.item():.4f}")
            
            # Get distributed statistics
            stats = optimizer.get_distributed_stats()
            if stats:
                print(f"  Total Spikes: {stats.get('total_spikes', 0)}")
                print(f"  Spikes by Rank: {stats.get('spikes_by_rank', {})}")
    
    if rank == 0:
        print("\n" + "=" * 60)
        print("Training Complete!")
        print("=" * 60)
    
    # Cleanup
    cleanup_distributed()


def main():
    """Main function to launch distributed training."""
    world_size = torch.cuda.device_count()
    
    if world_size < 2:
        print("This example requires at least 2 GPUs.")
        print("For single GPU training, use the basic GuardedOptimizer.")
        return
    
    # Launch distributed training
    import torch.multiprocessing as mp
    mp.spawn(
        train_distributed,
        args=(world_size,),
        nprocs=world_size,
        join=True
    )


if __name__ == "__main__":
    main()

# Made with Bob
