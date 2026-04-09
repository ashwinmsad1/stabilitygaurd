# Changelog

All notable changes to StabilityGuard will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2026-04-09

### 🚀 Major Features Added

#### 1. RLHF/PPO Stability Guard
- **Industry-first RLHF training stability monitoring**
- **KL Divergence Monitor**: Tracks KL divergence between policy and reference model
  - Automatic penalty adjustment when KL exceeds threshold
  - Configurable target KL and adjustment rate
- **Reward Collapse Detector**: Identifies degenerate reward model behavior
  - Entropy-based collapse detection
  - Variance and range monitoring
- **Value Divergence Monitor**: Tracks critic network stability
  - Detects value function divergence
  - Monitors prediction variance
- **PPO Ratio Monitor**: Analyzes policy update magnitudes
  - Tracks clipping frequency
  - Identifies excessive policy changes
- **Unified RLHF Guard**: Single interface for all RLHF monitoring

#### 2. Distributed Training Support
- **Multi-GPU gradient spike detection** with rank attribution
- **Distributed Spike Detector**: Coordinates spike detection across all ranks
  - All-reduce operations for global spike detection
  - Identifies which GPU caused the spike
  - Synchronized recovery across all processes
- **FSDP Stability Guard**: Specialized monitoring for Fully Sharded Data Parallel
  - Tracks parameter sharding health
  - Monitors gradient synchronization
  - Detects communication bottlenecks
- **DeepSpeed Stability Guard**: Support for ZeRO optimization stages
  - ZeRO-1, ZeRO-2, ZeRO-3 compatibility
  - Optimizer state monitoring
  - Gradient partition tracking
- **Distributed Guarded Optimizer**: Extends GuardedOptimizer for distributed training

#### 3. Mixed Precision Stability
- **Comprehensive FP16/BF16/FP8 training monitoring**
- **Precision Monitor**: Detects numerical issues in mixed precision
  - Overflow and underflow detection
  - Dynamic range tracking
  - Precision-specific recommendations
- **Adaptive Loss Scaler**: Stability-aware loss scaling
  - Conservative scaling near gradient spikes
  - Automatic scale adjustment
  - Overflow recovery
- **Mixed Precision Guard**: Unified interface for precision monitoring
  - Automatic precision selection
  - Real-time issue detection
  - Actionable recommendations

#### 4. Advanced Logging System
- **Comprehensive training diagnostics**
- **Gradient Flow Tracker**: Per-layer gradient monitoring
  - Identifies vanishing/exploding gradients
  - Detects gradient flow bottlenecks
  - Layer-wise statistics
- **Activation Stats Logger**: Tracks activation distributions
  - Dead neuron detection
  - Saturation monitoring
  - Layer-wise activation analysis
- **Weight Update Tracker**: Monitors parameter changes
  - Update magnitude tracking
  - Learning rate effectiveness
  - Parameter stability analysis
- **Checkpoint Health Scorer**: Evaluates checkpoint quality
  - Stability-based scoring (0-100)
  - Historical trend analysis
  - Checkpoint ranking
- **Advanced Logger**: Unified logging interface
  - Configurable component selection
  - Comprehensive statistics
  - Integration with existing loggers

### ✨ Improvements
- All v0.3.0 features are **modular and composable**
- Zero additional dependencies (PyTorch only)
- Comprehensive type hints and docstrings
- Production-ready error handling
- Extensive test coverage (1,000+ lines of tests)

### 📚 Documentation
- Added `ROADMAP_v0.3.0.md` with detailed specifications
- Added `V0.3.0_IMPLEMENTATION_COMPLETE.md` with implementation summary
- Comprehensive docstrings for all new modules
- Usage examples for each component

### 🧪 Testing
- Unit tests for all RLHF components (289 lines)
- Unit tests for distributed training (87 lines)
- Unit tests for mixed precision (232 lines)
- Unit tests for logging system (172 lines)
- Integration tests for component interaction (223 lines)
- Total test coverage: 1,003 lines

### 🔧 Technical Details
- **RLHF**: KL divergence computed using log probabilities
- **Distributed**: All-reduce operations for spike coordination
- **Precision**: Dynamic loss scaling with stability awareness
- **Logging**: Minimal overhead with configurable frequency

### 📦 Module Structure
```
stabilityguard/
├── rlhf/              # RLHF monitoring (1,261 lines)
│   ├── kl_monitor.py
│   ├── reward_collapse.py
│   ├── value_divergence.py
│   ├── ppo_ratio.py
│   └── rlhf_guard.py
├── distributed/       # Distributed training (1,215 lines)
│   ├── spike_detector.py
│   ├── fsdp_guard.py
│   ├── deepspeed_guard.py
│   └── distributed_optimizer.py
├── precision/         # Mixed precision (875 lines)
│   ├── precision_monitor.py
│   ├── loss_scaler.py
│   └── mixed_precision_guard.py
└── logging/          # Advanced logging (467 lines)
    ├── gradient_flow.py
    ├── activation_stats.py
    ├── weight_updates.py
    ├── checkpoint_scorer.py
    └── advanced_logger.py
```

### 🎯 Use Cases
- **RLHF Training**: Monitor ChatGPT-style fine-tuning stability
- **Multi-GPU Training**: Coordinate spike detection across GPUs
- **Mixed Precision**: Safely train with FP16/BF16/FP8
- **Production Monitoring**: Comprehensive training diagnostics

## [0.2.0] - 2026-04-08

### 🚀 Major Features Added

#### 1. Edge of Stability Detection
- **Predictive spike detection** using Hessian spectral radius estimation
- Detects instability 10-50 steps before gradient explosion occurs
- Uses power iteration method for efficient λ_max computation
- Configurable with `enable_edge_of_stability=True`
- Parameters: `eos_check_interval`, `eos_power_iterations`, `eos_stability_threshold`

#### 2. SPAM Optimizer (Spike-Aware Momentum)
- **Automatic momentum buffer reset** on spike detection
- Prevents gradient corruption from propagating through Adam's exponential moving averages
- **Gradual learning rate recovery** after spike handling
- Configurable with `enable_spam=True`
- Parameters: `spam_lr_reduction`, `spam_recovery_steps`

#### 3. Auto-Calibration
- **Statistical threshold tuning** during warmup period
- Fits log-normal distribution to gradient norms
- Sets threshold at 99th percentile automatically
- Eliminates manual threshold tuning
- Configurable with `enable_auto_calibration=True`
- Parameters: `auto_calibration_warmup_steps`, `auto_calibration_percentile`

#### 4. HELENE Clipping (Hessian-Aware Adaptive Clipping)
- **Per-layer adaptive gradient clipping** based on local Hessian conditioning
- Adjusts clip values dynamically: `clip = base_clip / sqrt(κ)`
- More aggressive clipping for ill-conditioned layers
- Configurable with `enable_helene=True`
- Parameters: `helene_base_clip`, `helene_power_iterations`

### ✨ Improvements
- All v0.2.0 features are **opt-in** (default `False`) for backward compatibility
- Comprehensive integration into `GuardedOptimizer.step()` method
- CPU-friendly test suite for validation without GPU
- Detailed documentation for each feature

### 📚 Documentation
- Added `ROADMAP_v0.2.0.md` with detailed feature specifications
- Added `LONG_TERM_ROADMAP.md` with vision through v2.0.0
- Added `HESSIAN_VECTOR_PRODUCTS_EXPLAINED.md` explaining computational costs
- Created CPU-friendly test suite: `test_v0.2.0_features_cpu.py`

### 🔧 Technical Details
- Edge of Stability: 2 backward passes per Hessian-vector product
- SPAM: Resets `exp_avg` and `exp_avg_sq` buffers in Adam optimizer
- Auto-Calibration: Collects samples during warmup, fits distribution
- HELENE: Estimates per-layer conditioning number using power iteration

### 🧪 Testing
- All 5 v0.2.0 features tested and validated
- Tests pass on CPU with small models (1.3K parameters)
- Optimized for fast testing (5 power iterations vs 20)

## [0.1.3] - 2026-04-07

### 🐛 Bug Fixes
- Fixed false positive spike detection in early training steps
- Improved baseline gradient norm tracking accuracy

### 📊 Performance
- Documented honest performance overhead: 1-5% GPU time
- Optimized spike detection algorithm

### 📚 Documentation
- Added performance benchmarks
- Updated README with realistic expectations
- Added troubleshooting guide

## [0.1.2] - 2026-04-06

### ✨ Features
- Added WandB integration
- Added MLflow integration
- Added HuggingFace Trainer integration

### 🔧 Improvements
- Better logging format
- Improved error messages

## [0.1.1] - 2026-04-05

### 🐛 Bug Fixes
- Fixed snapshot restoration on CPU
- Fixed NaN handling in mixed precision training

## [0.1.0] - 2026-04-04

### 🎉 Initial Release
- Basic gradient spike detection
- NaN/Inf gradient handling
- Automatic model snapshot and restoration
- JSON logging of spike events
- Support for PyTorch 2.0+

---

## Version Numbering

- **Major version (X.0.0)**: Breaking API changes
- **Minor version (0.X.0)**: New features, backward compatible
- **Patch version (0.0.X)**: Bug fixes, backward compatible

## Links

- [PyPI Package](https://pypi.org/project/stabilityguard/)
- [GitHub Repository](https://github.com/ashwinmsad1/stabilityguard)
- [Documentation](https://github.com/ashwinmsad1/stabilityguard#readme)