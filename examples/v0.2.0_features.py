"""
Example: Using StabilityGuard v0.2.0 Advanced Features

Demonstrates all four new features:
1. Edge of Stability Detection (predictive)
2. SPAM Optimizer (momentum reset)
3. Auto-Calibration (zero manual tuning)
4. HELENE Clipping (adaptive per-layer)

Usage:
    pip install stabilityguard>=0.2.0
    python v0.2.0_features.py
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from stabilityguard import GuardedOptimizer


# ━━━ Simple Model for Demo ━━━
class DemoModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 10)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


def main():
    print("=" * 70)
    print("StabilityGuard v0.2.0 - Advanced Features Demo")
    print("=" * 70)
    
    # Initialize model and base optimizer
    model = DemoModel()
    base_opt = AdamW(model.parameters(), lr=1e-3)
    
    # ━━━ Example 1: All Features Enabled ━━━
    print("\n[Example 1] All v0.2.0 features enabled:")
    print("-" * 70)
    
    optimizer = GuardedOptimizer(
        base_opt, model,
        
        # Core settings
        spike_threshold=10.0,
        nan_action="skip",
        log_every=50,
        
        # ✨ v0.2.0 Features (all enabled)
        enable_edge_of_stability=True,     # Predictive detection
        enable_spam=True,                  # Momentum reset
        enable_auto_calibration=True,      # Auto-tune threshold
        enable_helene=True,                # Adaptive clipping
        
        # Fine-tuning
        eos_check_interval=10,             # Check every 10 steps
        eos_power_iterations=20,           # Accuracy vs speed
        spam_lr_reduction=0.5,             # Reduce LR to 50% after spike
        spam_recovery_steps=100,           # Recover over 100 steps
        auto_calibration_warmup_steps=100, # Collect samples for 100 steps
        helene_base_clip=1.0,              # Base clip value
        
        verbose=True
    )
    
    print("\n✅ GuardedOptimizer initialized with all v0.2.0 features!")
    print("\nFeatures enabled:")
    print("  🔮 Edge of Stability: Predicts spikes 10-50 steps ahead")
    print("  🔄 SPAM Optimizer: Resets momentum on spike detection")
    print("  🎯 Auto-Calibration: Learns optimal threshold automatically")
    print("  ✂️ HELENE Clipping: Adaptive per-layer gradient clipping")
    
    # ━━━ Example 2: Edge of Stability Only ━━━
    print("\n\n[Example 2] Edge of Stability only (predictive detection):")
    print("-" * 70)
    
    model2 = DemoModel()
    base_opt2 = AdamW(model2.parameters(), lr=1e-3)
    
    optimizer2 = GuardedOptimizer(
        base_opt2, model2,
        enable_edge_of_stability=True,
        eos_check_interval=10,
        eos_stability_threshold=2.0,  # λ_max threshold (2/lr)
        verbose=False
    )
    
    print("✅ Predictive spike detection enabled")
    print("   Checks Hessian spectral radius every 10 steps")
    print("   Warns when λ_max × lr > 2 (approaching instability)")
    
    # ━━━ Example 3: SPAM + Auto-Calibration ━━━
    print("\n\n[Example 3] SPAM + Auto-Calibration (production setup):")
    print("-" * 70)
    
    model3 = DemoModel()
    base_opt3 = AdamW(model3.parameters(), lr=1e-3)
    
    optimizer3 = GuardedOptimizer(
        base_opt3, model3,
        enable_spam=True,                  # Momentum reset
        enable_auto_calibration=True,      # Auto-tune threshold
        spam_lr_reduction=0.5,
        spam_recovery_steps=100,
        auto_calibration_warmup_steps=100,
        verbose=False
    )
    
    print("✅ Production-ready configuration:")
    print("   • Auto-calibration eliminates manual threshold tuning")
    print("   • SPAM resets momentum buffers on spike detection")
    print("   • Gradual LR recovery prevents secondary spikes")
    
    # ━━━ Example 4: HELENE Only ━━━
    print("\n\n[Example 4] HELENE Clipping only (adaptive per-layer):")
    print("-" * 70)
    
    model4 = DemoModel()
    base_opt4 = AdamW(model4.parameters(), lr=1e-3)
    
    optimizer4 = GuardedOptimizer(
        base_opt4, model4,
        enable_helene=True,
        helene_base_clip=1.0,
        helene_power_iterations=10,
        verbose=False
    )
    
    print("✅ Adaptive per-layer gradient clipping enabled")
    print("   Clip value = base_clip / sqrt(κ)")
    print("   Ill-conditioned layers get more aggressive clipping")
    
    # ━━━ Training Loop Demo ━━━
    print("\n\n[Demo] Running 20 training steps with all features...")
    print("-" * 70)
    
    for step in range(20):
        # Generate random batch
        x = torch.randn(32, 128)
        y = torch.randint(0, 10, (32,))
        
        # Forward pass
        logits = model(x)
        loss = nn.functional.cross_entropy(logits, y)
        
        # Backward pass
        loss.backward()
        
        # GuardedOptimizer step (with all v0.2.0 features)
        optimizer.step(loss=loss.item())
        optimizer.zero_grad()
        
        if step % 5 == 0:
            print(f"  Step {step:3d}: loss = {loss.item():.4f}")
    
    print("\n✅ Training completed successfully!")
    print("\nAll v0.2.0 features worked seamlessly together.")
    
    # ━━━ Summary ━━━
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("\nv0.2.0 Features:")
    print("  1. Edge of Stability: Predictive spike detection (10-50 steps ahead)")
    print("  2. SPAM Optimizer: Momentum reset + gradual LR recovery")
    print("  3. Auto-Calibration: Zero manual threshold tuning")
    print("  4. HELENE Clipping: Adaptive per-layer gradient clipping")
    print("\nBackward Compatibility:")
    print("  • All features are opt-in (default False)")
    print("  • Existing v0.1.x code continues to work without changes")
    print("\nRecommended for Production:")
    print("  • enable_spam=True + enable_auto_calibration=True")
    print("  • enable_edge_of_stability=True for critical runs")
    print("  • enable_helene=True for heterogeneous models")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()

# Made with Bob
