# ⚡ StabilityGuard

**One-line circuit breaker for PyTorch training.**  
Add one line. See exactly which layer is about to explode.

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

---

## The Problem

Training large models? Gradient spikes and NaN explosions silently corrupt your model weights. You lose hours of GPU time, and the error message tells you nothing about *which layer* caused it.

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
| Per-step overhead (no spike) | **< 0.2ms** on A100 |
| Per-step overhead (spike detected) | **~3ms** (includes JSON write) |
| External dependencies | **0** (only PyTorch) |
| License | **MIT** |

## Install from source

```bash
git clone https://github.com/stabilityguard/stabilityguard.git
cd stabilityguard
pip install -e ".[test]"
pytest tests/ -v
```

## License

MIT — use it anywhere, no restrictions.
