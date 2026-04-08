# ⚡ StabilityGuard

**Advanced PyTorch training stability with predictive spike detection.**  
Catch gradient explosions before they happen. Auto-tune thresholds. Recover gracefully.

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

---

## The Problem

Training large neural networks fails catastrophically when **gradient spikes** or **NaN explosions** occur. A single corrupted gradient can poison your optimizer's state (momentum buffers, Adam's second moments), forcing you to restart from the last checkpoint—often thousands of steps back.

**The failure cascade:**
1. A gradient spike hits one layer (10-1000× larger than normal)
2. Optimizer state gets corrupted with NaN/Inf values
3. Subsequent steps propagate NaN through the entire model
4. By the time PyTorch throws `RuntimeError: Function returned nan`, it's too late—your checkpoint is poisoned

**What PyTorch tells you:**
```
RuntimeError: Function 'AddmmBackward0' returned nan values in its 0th output
```

**What you actually need to know:**
- Which layer caused the spike?
- When did it start (exact step number)?
- What was the gradient magnitude vs. normal baseline?
- Can we catch it *before* it corrupts the optimizer?

**The cost:** For a 7B LLM on 8×A100 GPUs, a single NaN explosion at 40% training wastes ~72 GPU-hours. You lose days of compute because one gradient went haywire.

StabilityGuard solves this by monitoring every layer's gradients in real-time and catching spikes *before* they corrupt your training run.

## The Fix

```bash
pip install stabilityguard
```

**Before StabilityGuard** (standard PyTorch):
```python
from torch.optim import AdamW

optimizer = AdamW(model.parameters(), lr=2e-4)

for batch in dataloader:
    loss = model(batch)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

**After StabilityGuard** (one-line change):
```python
from torch.optim import AdamW
from stabilityguard import GuardedOptimizer  # ← the only change

base_opt = AdamW(model.parameters(), lr=2e-4)
optimizer = GuardedOptimizer(base_opt, model,
    spike_threshold=10.0,      # gradient norm ratio to trigger alert
    nan_action="skip",         # "skip" | "rollback" | "raise"
    log_every=50               # steps between diagnostic summaries
)

for batch in dataloader:
    loss = model(batch)
    loss.backward()
    optimizer.step()           # GuardedOptimizer intercepts here
    optimizer.zero_grad()
```

## 🚀 New in v0.2.0: Advanced Features

### 1. 🔮 Edge of Stability (Predictive Detection)
**Predict spikes 10-50 steps before they happen** using Hessian spectral radius estimation.

```python
optimizer = GuardedOptimizer(base_opt, model,
    enable_edge_of_stability=True,
    eos_check_interval=10
)
```

### 2. 🔄 SPAM Optimizer (Momentum Reset)
**Automatically reset momentum buffers** when spikes are detected, preventing corruption propagation.

```python
optimizer = GuardedOptimizer(base_opt, model,
    enable_spam=True,
    spam_lr_reduction=0.5,
    spam_recovery_steps=100
)
```

### 3. 🎯 Auto-Calibration (Zero Manual Tuning)
**Eliminate manual threshold tuning** by learning optimal thresholds from your data.

```python
optimizer = GuardedOptimizer(base_opt, model,
    enable_auto_calibration=True,
    auto_calibration_warmup_steps=100
)
```

### 4. ✂️ HELENE Clipping (Adaptive Per-Layer)
**Per-layer adaptive gradient clipping** based on local Hessian conditioning.

```python
optimizer = GuardedOptimizer(base_opt, model,
    enable_helene=True,
    helene_base_clip=1.0
)
```

**All v0.2.0 features are opt-in (default `False`) for backward compatibility.**

See `examples/v0.2.0_features.py` for detailed usage examples.

## What You See

When a spike hits at step 847:

```
╔══════════════════════════════════════════════════════════════╗
║  ⚠ STABILITYGUARD — SPIKE DETECTED @ step 847                ║
╠══════════════════════════════════════════════════════════════╣
║  Trigger layer  : transformer.h.11.mlp.c_proj                ║
║  Grad norm      : 847.3  (baseline: 1.2, ratio: 706.1x)      ║
║  Action taken   : optimizer.step() SKIPPED                   ║
║  Loss (pre-skip): 14.71                                      ║
╚══════════════════════════════════════════════════════════════╝
stabilityguard.log written → ./sg_logs/spike_step847.json
```

**Your model weights are untouched. Training continues.**

## How It Works

1. **Backward hooks** on every `nn.Module` capture per-layer gradient L2 norms
2. **EMA baselines** track normal gradient behavior (α=0.01, ~100 step lookback)
3. **Spike detection** fires when `current_norm / ema_baseline > threshold`
4. **NaN/Inf short-circuit** catches corrupted gradients via `torch.isfinite`
5. **Actions** execute automatically: skip the step, rollback to checkpoint, or raise

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `spike_threshold` | `10.0` | Ratio of current norm to EMA baseline that triggers a spike |
| `nan_action` | `"skip"` | Action on spike: `"skip"`, `"rollback"`, or `"raise"` |
| `log_every` | `50` | Steps between periodic diagnostic summaries |
| `log_dir` | `"./sg_logs"` | Directory for JSON spike reports |
| `ema_alpha` | `0.01` | EMA smoothing factor (smaller = slower adaptation) |
| `warmup_steps` | `10` | Steps before spike detection activates |
| `verbose` | `True` | Print diagnostic summaries to stdout |

### v0.2.0 Advanced Features

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enable_edge_of_stability` | `False` | Enable predictive spike detection |
| `enable_spam` | `False` | Enable momentum reset on spikes |
| `enable_auto_calibration` | `False` | Enable automatic threshold tuning |
| `enable_helene` | `False` | Enable adaptive per-layer clipping |
| `eos_check_interval` | `10` | Steps between Edge of Stability checks |
| `eos_power_iterations` | `20` | Accuracy of λ_max estimation |
| `spam_lr_reduction` | `0.5` | LR reduction factor after spike |
| `spam_recovery_steps` | `100` | Steps to recover LR after spike |
| `auto_calibration_warmup_steps` | `100` | Warmup steps for auto-calibration |
| `helene_base_clip` | `1.0` | Base gradient clip value for HELENE |

## Actions

| Action | Behavior |
|--------|----------|
| `skip` | Skip `optimizer.step()` — corrupted gradients discarded, weights unchanged |
| `rollback` | Restore model + optimizer to the last clean checkpoint |
| `raise` | Throw `GradientSpikeError` for interactive debugging |

## Integrations

### Weights & Biases
```python
from stabilityguard.integrations.wandb import WandBBridge
bridge = WandBBridge()
# Metrics logged under sg/ namespace automatically
```

### MLflow
```python
from stabilityguard.integrations.mlflow import MLflowBridge
bridge = MLflowBridge()
```

### HuggingFace Transformers
```python
from stabilityguard.integrations.huggingface import StabilityGuardCallback

trainer = Trainer(
    model=model,
    args=training_args,
    callbacks=[StabilityGuardCallback(spike_threshold=10.0)],
)
```

## Performance

| Metric | Value |
|--------|-------|
| Per-step overhead (GPU) | **Low** (estimated 1-5% on modern GPUs) |
| Per-step overhead (CPU) | **High** (~60%, not recommended for production) |
| Per-step overhead (spike detected) | **~3ms** (includes JSON write) |
| Recommended for | **GPU-based training** (production workloads) |
| External dependencies | **0** (only PyTorch) |
| License | **MIT** |

### v0.2.0 Feature Overhead

| Feature | Overhead | When |
|---------|----------|------|
| Edge of Stability | ~40 backward passes | Every `eos_check_interval` steps |
| SPAM | Negligible | Only on spike detection |
| Auto-Calibration | Negligible | Only during warmup |
| HELENE | ~20 backward passes | Every step (if enabled) |

## Understanding Spike Logs

StabilityGuard writes detailed JSON logs for every spike detected to `./sg_logs/`:

```json
{
  "snapshot": {
    "step": 847,                   // When spike occurred
    "spike_layer": "fc2",          // Which layer caused it
    "spike_ratio": 12.8,           // Severity (current/baseline)
    "action": "skip",              // Action taken
    "loss": "tensor(14.71, ...)"   // Loss at spike time
  },
  "layer_norms": {
    "fc2": 15.36,                  // Current gradient norm
    "fc2.weight": 14.82
  },
  "ema_baselines": {
    "fc2": 1.20,                   // Expected "normal" norm
    "fc2.weight": 1.15
  },
  "nan_layers": []                 // NaN corruption check
}
```

### Spike Severity Scale

| spike_ratio | Severity | Action |
|-------------|----------|--------|
| 1.0-2.0 | Normal | No spike |
| 2.0-5.0 | Moderate ⚠️ | Monitor |
| 5.0-10.0 | Severe 🔥 | Investigate |
| >10.0 | Critical 💥 | Fix immediately |

### Quick Analysis Commands

```bash
# Count total spikes
ls sg_logs/ | wc -l

# Find most problematic layer
grep -h "spike_layer" sg_logs/*.json | sort | uniq -c | sort -rn

# Check for NaN corruption
grep -l "nan_layers.*\[" sg_logs/*.json
```

## Install from source

```bash
git clone https://github.com/ashwinmsad1/stabilityguard.git
cd stabilityguard
pip install -e ".[test]"
pytest tests/ -v
```

## Documentation

- [v0.2.0 Release Summary](../V0.2.0_RELEASE_SUMMARY.md) - Detailed feature overview
- [v0.2.0 Roadmap](../ROADMAP_v0.2.0.md) - Technical specifications
- [Long-term Roadmap](../LONG_TERM_ROADMAP.md) - Vision through v2.0.0
- [Hessian-Vector Products Explained](../HESSIAN_VECTOR_PRODUCTS_EXPLAINED.md) - Computational cost analysis
- [Changelog](CHANGELOG.md) - Version history

## License

MIT — use it anywhere, no restrictions.
