# Contributing to StabilityGuard

Thank you for your interest in contributing to StabilityGuard! This project aims to make LLM training more stable and reliable, and we welcome contributions from the community.

## 🎯 How You Can Help

### 1. Report Issues
- **Bug Reports**: Found a bug? Open an issue with:
  - Clear description of the problem
  - Minimal reproducible example
  - Your environment (PyTorch version, Python version, OS)
  - Expected vs. actual behavior
  
- **Feature Requests**: Have an idea? We'd love to hear it!
  - Describe the use case
  - Explain why it would be valuable
  - Suggest potential implementation approaches

### 2. Improve Documentation
- Fix typos or clarify confusing sections
- Add examples or tutorials
- Improve docstrings
- Translate documentation

### 3. Contribute Code
- Fix bugs
- Implement new features
- Improve performance
- Add tests
- Enhance integrations (TensorBoard, Neptune, etc.)

## 🚀 Getting Started

### Development Setup

1. **Fork and Clone**
   ```bash
   git clone https://github.com/yourusername/stabilityguard.git
   cd stabilityguard
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install in Development Mode**
   ```bash
   pip install -e ".[dev]"
   ```

4. **Install Development Dependencies**
   ```bash
   pip install pytest pytest-cov black isort mypy
   ```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=stabilityguard --cov-report=html

# Run specific test file
pytest tests/test_spike_detection.py

# Run with verbose output
pytest -v
```

### Code Style

We follow standard Python conventions:

```bash
# Format code with black
black stabilityguard/ tests/

# Sort imports with isort
isort stabilityguard/ tests/

# Type checking with mypy
mypy stabilityguard/
```

## 📝 Pull Request Process

1. **Create a Branch**
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/your-bug-fix
   ```

2. **Make Your Changes**
   - Write clear, concise commit messages
   - Add tests for new functionality
   - Update documentation as needed
   - Ensure all tests pass

3. **Commit Guidelines**
   ```
   feat: add SPAM optimizer integration
   fix: resolve NaN detection in mixed precision
   docs: improve quickstart guide
   test: add edge cases for rollback action
   refactor: simplify hook registration logic
   ```

4. **Push and Create PR**
   ```bash
   git push origin feature/your-feature-name
   ```
   - Open a pull request on GitHub
   - Provide clear description of changes
   - Reference any related issues
   - Wait for review and address feedback

5. **PR Checklist**
   - [ ] Tests pass locally
   - [ ] New tests added for new features
   - [ ] Documentation updated
   - [ ] Code follows style guidelines
   - [ ] Commit messages are clear
   - [ ] No merge conflicts

## 🎨 Code Guidelines

### Python Style
- Follow PEP 8
- Use type hints where appropriate
- Write docstrings for public APIs (Google style)
- Keep functions focused and small
- Prefer composition over inheritance

### Example Docstring
```python
def detect_spike(
    grad_norms: Dict[str, float],
    threshold: float = 10.0
) -> Optional[SpikeReport]:
    """Detect gradient spikes using EMA baseline comparison.
    
    Args:
        grad_norms: Dictionary mapping layer names to gradient norms
        threshold: Multiplier for spike detection (default: 10.0)
        
    Returns:
        SpikeReport if spike detected, None otherwise
        
    Example:
        >>> norms = {"layer1": 0.5, "layer2": 100.0}
        >>> report = detect_spike(norms, threshold=10.0)
        >>> print(report.worst_layer)
        'layer2'
    """
```

### Testing Guidelines
- Write unit tests for new functions
- Add integration tests for new features
- Test edge cases and error conditions
- Use fixtures for common test setup
- Mock external dependencies (W&B, MLflow)

## 🌟 Feature Priorities

We're particularly interested in contributions for:

### High Priority
- [ ] Hessian spectral radius estimation
- [ ] SPAM optimizer implementation
- [ ] Real-time training dashboard
- [ ] Additional ML framework integrations (JAX, TensorFlow)

### Medium Priority
- [ ] HELENE layer-wise clipping
- [ ] PPO/RLHF divergence monitoring
- [ ] Automatic hyperparameter tuning
- [ ] Performance optimizations

### Community Requests
- [ ] TensorBoard integration
- [ ] Neptune.ai integration
- [ ] Distributed training support
- [ ] Model-specific presets (GPT, LLaMA, etc.)

## 🐛 Bug Bounty

While we don't have a formal bug bounty program, we deeply appreciate bug reports and fixes:

- **Critical bugs** (data corruption, crashes): Acknowledged in CHANGELOG
- **Security issues**: Please email directly (see SECURITY.md)
- **Feature contributions**: Credit in release notes

## 📚 Resources

- **Documentation**: [README.md](README.md)
- **Architecture**: [stabilityguard_architecture.html](../stabilityguard_architecture.html)
- **Examples**: [examples/](examples/)
- **Tests**: [tests/](tests/)

## 💬 Communication

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: Questions and general discussion
- **Pull Requests**: Code contributions

## 📜 Code of Conduct

### Our Pledge
We are committed to providing a welcoming and inclusive environment for all contributors, regardless of:
- Experience level
- Gender identity and expression
- Sexual orientation
- Disability
- Personal appearance
- Body size
- Race
- Ethnicity
- Age
- Religion
- Nationality

### Expected Behavior
- Be respectful and considerate
- Welcome newcomers and help them learn
- Accept constructive criticism gracefully
- Focus on what's best for the community
- Show empathy towards others

### Unacceptable Behavior
- Harassment or discriminatory language
- Trolling or insulting comments
- Personal or political attacks
- Publishing others' private information
- Other conduct inappropriate in a professional setting

## 🙏 Recognition

Contributors will be:
- Listed in CHANGELOG.md for significant contributions
- Mentioned in release notes
- Added to CONTRIBUTORS.md (coming soon)

## ❓ Questions?

Don't hesitate to ask! Open an issue with the `question` label, and we'll help you get started.

---

**Thank you for making StabilityGuard better!** 🚀