"""
Example: Mixed Precision Training with StabilityGuard

This example demonstrates how to use StabilityGuard's mixed precision
monitoring features for FP16/BF16 training.
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from stabilityguard.precision import MixedPrecisionGuard
from stabilityguard.core import GuardedOptimizer


class TransformerModel(nn.Module):
    """Simple transformer model for demonstration."""
    def __init__(self, vocab_size=10000, d_model=512, nhead=8, num_layers=6):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, 512, d_model))
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, batch_first=True),
            num_layers=num_layers
        )
        self.fc_out = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        seq_len = x.size(1)
        x = self.embedding(x) + self.pos_encoding[:, :seq_len, :]
        x = self.transformer(x)
        x = self.fc_out(x)
        return x


def train_fp16():
    """Train with FP16 mixed precision."""
    print("=" * 60)
    print("FP16 Mixed Precision Training with StabilityGuard")
    print("=" * 60)
    
    # Setup model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerModel().to(device)
    
    # Setup optimizer
    optimizer = AdamW(model.parameters(), lr=1e-4)
    guarded_opt = GuardedOptimizer(
        optimizer,
        model,
        spike_threshold=10.0,
        nan_action="skip"
    )
    
    # Setup mixed precision guard
    precision_guard = MixedPrecisionGuard(
        model=model,
        precision="fp16",
        initial_scale=2**16,
        growth_interval=2000
    )
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    for step in range(100):
        # Generate dummy batch
        batch_size = 8
        seq_len = 128
        inputs = torch.randint(0, 10000, (batch_size, seq_len)).to(device)
        targets = torch.randint(0, 10000, (batch_size, seq_len)).to(device)
        
        # Forward pass with autocast
        with torch.cuda.amp.autocast(dtype=torch.float16):
            outputs = model(inputs)
            loss = criterion(outputs.view(-1, 10000), targets.view(-1))
        
        # Scale loss
        scaled_loss = precision_guard.scale_loss(loss)
        
        # Backward pass
        scaled_loss.backward()
        
        # Unscale gradients and check for issues
        precision_guard.unscale_gradients(guarded_opt)
        
        # Check for precision issues
        issues = precision_guard.check_precision(model, guarded_opt, step)
        if issues:
            print(f"\nStep {step}: Precision Issues:")
            for issue in issues:
                print(f"  - {issue}")
        
        # Optimizer step
        guarded_opt.step()
        guarded_opt.zero_grad()
        
        # Update loss scale
        precision_guard.update_scale(model, guarded_opt, step)
        
        # Periodic logging
        if step % 10 == 0:
            stats = precision_guard.get_stats()
            print(f"\nStep {step}:")
            print(f"  Loss: {loss.item():.4f}")
            print(f"  Loss Scale: {stats['current_scale']:.0f}")
            print(f"  Overflows: {stats['overflow_count']}")
            print(f"  Underflows: {stats['underflow_count']}")
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    
    # Final statistics
    final_stats = precision_guard.get_stats()
    print("\nFinal Precision Statistics:")
    print(f"  Total Overflows: {final_stats['overflow_count']}")
    print(f"  Total Underflows: {final_stats['underflow_count']}")
    print(f"  Final Loss Scale: {final_stats['current_scale']:.0f}")
    print(f"  Scale Updates: {final_stats.get('scale_updates', 0)}")


def train_bf16():
    """Train with BF16 mixed precision."""
    print("\n" + "=" * 60)
    print("BF16 Mixed Precision Training with StabilityGuard")
    print("=" * 60)
    
    # Setup model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerModel().to(device)
    
    # Setup optimizer
    optimizer = AdamW(model.parameters(), lr=1e-4)
    guarded_opt = GuardedOptimizer(
        optimizer,
        model,
        spike_threshold=10.0,
        nan_action="skip"
    )
    
    # Setup mixed precision guard (BF16 doesn't need loss scaling)
    precision_guard = MixedPrecisionGuard(
        model=model,
        precision="bf16"
    )
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    for step in range(50):
        # Generate dummy batch
        batch_size = 8
        seq_len = 128
        inputs = torch.randint(0, 10000, (batch_size, seq_len)).to(device)
        targets = torch.randint(0, 10000, (batch_size, seq_len)).to(device)
        
        # Forward pass with autocast
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            outputs = model(inputs)
            loss = criterion(outputs.view(-1, 10000), targets.view(-1))
        
        # Backward pass (no scaling needed for BF16)
        loss.backward()
        
        # Check for precision issues
        issues = precision_guard.check_precision(model, guarded_opt, step)
        if issues:
            print(f"\nStep {step}: Precision Issues:")
            for issue in issues:
                print(f"  - {issue}")
        
        # Optimizer step
        guarded_opt.step()
        guarded_opt.zero_grad()
        
        # Periodic logging
        if step % 10 == 0:
            print(f"\nStep {step}:")
            print(f"  Loss: {loss.item():.4f}")
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)


def main():
    """Main function."""
    # Train with FP16
    train_fp16()
    
    # Train with BF16 (if supported)
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        train_bf16()
    else:
        print("\nBF16 not supported on this device, skipping BF16 example.")


if __name__ == "__main__":
    main()


