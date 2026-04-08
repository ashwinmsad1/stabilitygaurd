"""
Model utility functions.

Helpers for inspecting PyTorch model architecture: enumerating
named modules, computing architecture hashes, and parameter counts.
"""

import hashlib
from typing import Dict, Iterator, Tuple

import torch.nn as nn


def get_all_named_modules(model: nn.Module) -> Dict[str, nn.Module]:
    """Get a flat dictionary of all named modules in the model.

    Args:
        model: PyTorch model.

    Returns:
        Dict mapping qualified name → module instance.
    """
    return {name: module for name, module in model.named_modules() if name}


def compute_model_hash(model: nn.Module) -> str:
    """Compute a SHA-256 hash of the model architecture representation.

    Useful for identifying the exact model architecture in spike reports,
    allowing correlation across different training runs of the same model.

    Args:
        model: PyTorch model.

    Returns:
        First 16 characters of the SHA-256 hex digest.
    """
    return hashlib.sha256(repr(model).encode()).hexdigest()[:16]


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Count total parameters in a model.

    Args:
        model: PyTorch model.
        trainable_only: If True, only count parameters with requires_grad=True.

    Returns:
        Total number of parameters.
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def get_leaf_modules(model: nn.Module) -> Iterator[Tuple[str, nn.Module]]:
    """Yield leaf modules (modules with no children).

    Leaf modules are the actual compute layers (Linear, Conv2d, etc.)
    as opposed to container modules (Sequential, ModuleList, etc.).

    Args:
        model: PyTorch model.

    Yields:
        (name, module) tuples for leaf modules only.
    """
    for name, module in model.named_modules():
        if not list(module.children()):
            yield name, module
