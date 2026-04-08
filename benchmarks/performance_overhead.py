"""
Benchmark script to measure StabilityGuard's performance overhead.

Tests:
1. Baseline PyTorch training (no StabilityGuard)
2. StabilityGuard with no spike (normal operation)
3. StabilityGuard with spike detected (worst case)

Expected results:
- No spike overhead: < 0.2ms per step
- Spike detected overhead: ~3ms per step (includes JSON write)
"""

import time
import torch
import torch.nn as nn
from torch.optim import AdamW
import statistics

# Import StabilityGuard
import sys
sys.path.insert(0, '..')
from stabilityguard import GuardedOptimizer


class BenchmarkModel(nn.Module):
    """Small transformer-like model for benchmarking."""
    def __init__(self, d_model=512, n_layers=6):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=d_model, nhead=8, batch_first=True)
            for _ in range(n_layers)
        ])
        self.fc = nn.Linear(d_model, d_model)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.fc(x)


def benchmark_baseline(model, optimizer, batch_size=32, seq_len=128, steps=100):
    """Benchmark baseline PyTorch training (no StabilityGuard)."""
    times = []
    
    for _ in range(steps):
        x = torch.randn(batch_size, seq_len, 512)
        
        start = time.perf_counter()
        
        loss = model(x).sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms
    
    return times


def benchmark_guarded_no_spike(model, optimizer, batch_size=32, seq_len=128, steps=100):
    """Benchmark StabilityGuard with no spikes (normal operation)."""
    times = []
    
    for _ in range(steps):
        x = torch.randn(batch_size, seq_len, 512)
        
        start = time.perf_counter()
        
        loss = model(x).sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms
    
    return times


def benchmark_guarded_with_spike(model, optimizer, batch_size=32, seq_len=128, steps=10):
    """Benchmark StabilityGuard with spike detected (worst case)."""
    times = []
    
    for i in range(steps):
        x = torch.randn(batch_size, seq_len, 512)
        
        start = time.perf_counter()
        
        loss = model(x).sum()
        loss.backward()
        
        # Inject spike every other step
        if i % 2 == 0:
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.fill_(float('nan'))
        
        optimizer.step()
        optimizer.zero_grad()
        
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms
    
    return times


def print_stats(name, times):
    """Print statistics for benchmark results."""
    print(f"\n{name}:")
    print(f"  Mean:   {statistics.mean(times):.3f} ms")
    print(f"  Median: {statistics.median(times):.3f} ms")
    print(f"  Stdev:  {statistics.stdev(times):.3f} ms")
    print(f"  Min:    {min(times):.3f} ms")
    print(f"  Max:    {max(times):.3f} ms")


def main():
    print("=" * 60)
    print("StabilityGuard Performance Benchmark")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Test 1: Baseline PyTorch
    print("\n[1/3] Benchmarking baseline PyTorch (no StabilityGuard)...")
    model1 = BenchmarkModel().to(device)
    optimizer1 = AdamW(model1.parameters(), lr=1e-4)
    baseline_times = benchmark_baseline(model1, optimizer1, steps=100)
    print_stats("Baseline PyTorch", baseline_times)
    
    # Test 2: StabilityGuard with no spikes
    print("\n[2/3] Benchmarking StabilityGuard (no spikes)...")
    model2 = BenchmarkModel().to(device)
    base_opt2 = AdamW(model2.parameters(), lr=1e-4)
    optimizer2 = GuardedOptimizer(
        base_opt2, model2,
        spike_threshold=10.0,
        nan_action="skip",
        log_every=1000,  # Disable periodic logging
        verbose=False
    )
    guarded_times = benchmark_guarded_no_spike(model2, optimizer2, steps=100)
    print_stats("StabilityGuard (no spikes)", guarded_times)
    
    # Test 3: StabilityGuard with spikes
    print("\n[3/3] Benchmarking StabilityGuard (with spikes)...")
    model3 = BenchmarkModel().to(device)
    base_opt3 = AdamW(model3.parameters(), lr=1e-4)
    optimizer3 = GuardedOptimizer(
        base_opt3, model3,
        spike_threshold=10.0,
        nan_action="skip",
        log_every=1000,
        verbose=False
    )
    spike_times = benchmark_guarded_with_spike(model3, optimizer3, steps=10)
    print_stats("StabilityGuard (with spikes)", spike_times)
    
    # Calculate overhead
    baseline_mean = statistics.mean(baseline_times)
    guarded_mean = statistics.mean(guarded_times)
    overhead = guarded_mean - baseline_mean
    overhead_pct = (overhead / baseline_mean) * 100
    
    print("\n" + "=" * 60)
    print("OVERHEAD ANALYSIS")
    print("=" * 60)
    print(f"Baseline mean:           {baseline_mean:.3f} ms")
    print(f"StabilityGuard mean:     {guarded_mean:.3f} ms")
    print(f"Absolute overhead:       {overhead:.3f} ms")
    print(f"Relative overhead:       {overhead_pct:.2f}%")
    
    # Verdict
    print("\n" + "=" * 60)
    print("VERDICT")
    print("=" * 60)
    if overhead < 0.2:
        print(f"✅ PASS: Overhead ({overhead:.3f} ms) is < 0.2 ms")
    elif overhead < 1.0:
        print(f"⚠️  ACCEPTABLE: Overhead ({overhead:.3f} ms) is < 1.0 ms")
    else:
        print(f"❌ FAIL: Overhead ({overhead:.3f} ms) is > 1.0 ms")
    
    print("\nNote: Spike detection overhead (~3ms) only occurs when spikes are detected,")
    print("which should be rare in healthy training runs.")


if __name__ == "__main__":
    main()

# Made with Bob
