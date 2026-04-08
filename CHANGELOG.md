# Changelog

All notable changes to StabilityGuard will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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