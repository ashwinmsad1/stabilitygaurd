# Changelog

All notable changes to StabilityGuard will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-04-08

### Added
- **Core Features**
  - `GuardedOptimizer`: Drop-in wrapper for any PyTorch optimizer with automatic gradient spike detection
  - `SpikeDetector`: EMA-based baseline tracking with configurable threshold detection
  - `GradientHookManager`: Per-layer gradient norm monitoring via backward hooks
  - Three remediation strategies: `skip`, `rollback`, and `raise`
  
- **Detection Capabilities**
  - Exponential Moving Average (EMA) baseline with α=0.01 for adaptive threshold tracking
  - Per-layer gradient norm monitoring for granular spike localization
  - NaN/Inf detection with dual-path verification (hooks + parameter scan)
  - Configurable warmup period to avoid false positives during initialization
  - Spike metadata capture including layer names, norms, and model state hash

- **Integrations**
  - Weights & Biases (W&B) logging integration
  - MLflow experiment tracking integration
  - HuggingFace Trainer callback for seamless integration with transformers

- **Utilities**
  - Comprehensive logging system with structured output
  - Model state checkpointing and rollback capabilities
  - Gradient snapshot system for forensic analysis
  - Spike report generation with top-10 worst layers

- **Examples**
  - GPT-2 pretraining example with spike injection demo
  - LLaMA fine-tuning example with HuggingFace integration

- **Testing**
  - Complete test suite with 22 tests covering:
    - Spike detection accuracy
    - NaN/Inf handling
    - All three remediation actions
    - Hook lifecycle management
    - Optimizer proxy behavior

### Technical Details
- **Dependencies**: PyTorch ≥2.0 (core), optional integrations for wandb/mlflow/transformers
- **Python Support**: Python 3.8+
- **License**: MIT
- **Architecture**: Non-invasive hook-based design with zero training loop modifications required

### Documentation
- Comprehensive README with quick start guide
- Architecture documentation with mathematical foundations
- API documentation via docstrings
- Example scripts for common use cases

---

## [Unreleased]

### Planned Features
- Hessian spectral radius estimation for Edge of Stability monitoring
- SPAM optimizer (momentum reset on spike detection)
- HELENE layer-wise gradient clipping with Hessian conditioning
- PPO/RLHF divergence constraint monitoring
- Real-time dashboard for training stability metrics
- Automatic hyperparameter tuning based on spike patterns

---

[0.1.0]: https://github.com/yourusername/stabilityguard/releases/tag/v0.1.0