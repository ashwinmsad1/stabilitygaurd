"""
Edge of Stability Detection

Implements Hessian spectral radius (λ_max) estimation to predict gradient spikes
before they occur. Based on Cohen et al. (2021) "Gradient Descent on Neural Networks
Typically Occurs at the Edge of Stability".

Key Insight: Training becomes unstable when 2 * learning_rate * λ_max > 2
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)

# Constants for numerical stability
EPSILON_HV_NORM = 1e-8  # Minimum Hessian-vector product norm


class EdgeOfStabilityDetector:
    """
    Detects when training approaches the "edge of stability" by estimating
    the largest eigenvalue (λ_max) of the Hessian matrix.
    
    The stability condition is: 2 * lr * λ_max < 2
    When this is violated, gradient descent becomes unstable and spikes occur.
    
    Args:
        model: PyTorch model to monitor
        power_iterations: Number of power iterations for λ_max estimation (default: 20)
        estimation_frequency: Estimate λ_max every N steps (default: 10)
        stability_threshold: Warn when sharpness > threshold (default: 1.8, i.e., 90% of limit)
        warmup_steps: Skip estimation during warmup (default: 10)
        verbose: Print warnings when approaching instability (default: True)
    
    Example:
        >>> detector = EdgeOfStabilityDetector(model, power_iterations=20)
        >>> for step, batch in enumerate(dataloader):
        >>>     loss = model(batch)
        >>>     loss.backward()
        >>>     
        >>>     # Check stability before optimizer step
        >>>     lambda_max, sharpness, is_unstable = detector.check_stability(
        >>>         loss, optimizer.param_groups[0]['lr'], step
        >>>     )
        >>>     
        >>>     if is_unstable:
        >>>         print(f"Warning: Approaching instability! λ_max={lambda_max:.4f}")
        >>>     
        >>>     optimizer.step()
    """
    
    def __init__(
        self,
        model: nn.Module,
        power_iterations: int = 20,
        estimation_frequency: int = 10,
        stability_threshold: float = 1.8,
        warmup_steps: int = 10,
        verbose: bool = True
    ):
        self.model = model
        self.power_iterations = power_iterations
        self.estimation_frequency = estimation_frequency
        self.stability_threshold = stability_threshold
        self.warmup_steps = warmup_steps
        self.verbose = verbose
        
        # State tracking
        self.step_count = 0
        self.lambda_max_history = []
        self.sharpness_history = []
        self.last_lambda_max = None
        self.last_sharpness = None
        
        # Cache for random vector (reused across iterations)
        self._random_vector = None
        
    def estimate_lambda_max(self, loss: torch.Tensor) -> float:
        """
        Estimate the largest eigenvalue of the Hessian using power iteration.
        
        Power iteration algorithm:
        1. Start with random vector v
        2. Repeat: v = H @ v / ||H @ v||  (where H is Hessian)
        3. λ_max ≈ v^T @ H @ v
        
        We compute H @ v efficiently using Hessian-vector products:
        H @ v = ∇_θ (∇_θ L · v)
        
        Args:
            loss: Scalar loss tensor (must have requires_grad=True)
            
        Returns:
            Estimated largest eigenvalue λ_max
        """
        # Get model parameters
        params = [p for p in self.model.parameters() if p.requires_grad]
        
        if len(params) == 0:
            logger.warning("No trainable parameters found")
            return 0.0
        
        # Initialize random vector if needed
        if self._random_vector is None:
            self._random_vector = [torch.randn_like(p) for p in params]
        
        # Normalize random vector
        v = self._random_vector
        v_norm = torch.sqrt(sum(torch.sum(vi ** 2) for vi in v))
        v = [vi / v_norm for vi in v]
        
        # Power iteration
        for _ in range(self.power_iterations):
            # Compute Hessian-vector product: H @ v
            # This is ∇_θ (∇_θ L · v)
            
            # First, compute gradients of loss
            grads = torch.autograd.grad(
                loss, params, create_graph=True, retain_graph=True
            )
            
            # Compute dot product: ∇_θ L · v
            grad_v_product = sum(
                torch.sum(g * vi) for g, vi in zip(grads, v)
            )
            
            # Compute gradient of this dot product: ∇_θ (∇_θ L · v) = H @ v
            hv = torch.autograd.grad(
                grad_v_product, params, retain_graph=True
            )
            
            # Normalize: v = H @ v / ||H @ v||
            hv_norm = torch.sqrt(sum(torch.sum(hvi ** 2) for hvi in hv))
            
            if hv_norm < EPSILON_HV_NORM:
                # Avoid division by zero
                logger.warning(
                    f"Hessian-vector product norm too small: {hv_norm:.2e}. "
                    "This may indicate numerical instability or flat loss landscape."
                )
                return 0.0
            
            v = [hvi / hv_norm for hvi in hv]
        
        # Estimate λ_max = v^T @ H @ v
        # We already have H @ v from last iteration
        lambda_max = sum(
            torch.sum(vi * hvi) for vi, hvi in zip(v, hv)
        ).item()
        
        # Update cached random vector for next iteration
        self._random_vector = v
        
        return abs(lambda_max)  # Take absolute value
    
    def compute_sharpness(self, learning_rate: float, lambda_max: float) -> float:
        """
        Compute sharpness metric: 2 * lr * λ_max
        
        Stability condition: sharpness < 2
        - sharpness < 1.0: Very stable
        - 1.0 < sharpness < 1.8: Stable
        - 1.8 < sharpness < 2.0: Approaching instability (warning zone)
        - sharpness > 2.0: Unstable (spikes likely)
        
        Args:
            learning_rate: Current learning rate
            lambda_max: Largest eigenvalue of Hessian
            
        Returns:
            Sharpness metric
        """
        return 2.0 * learning_rate * lambda_max
    
    def check_stability(
        self,
        loss: torch.Tensor,
        learning_rate: float,
        step: int
    ) -> Tuple[Optional[float], Optional[float], bool]:
        """
        Check if training is approaching the edge of stability.
        
        Args:
            loss: Current loss (must have gradients)
            learning_rate: Current learning rate
            step: Current training step
            
        Returns:
            Tuple of (lambda_max, sharpness, is_unstable)
            - lambda_max: Estimated largest eigenvalue (None if not estimated this step)
            - sharpness: 2 * lr * λ_max (None if not estimated this step)
            - is_unstable: True if approaching instability threshold
        """
        self.step_count += 1
        
        # Skip during warmup
        if step < self.warmup_steps:
            return None, None, False
        
        # Only estimate every N steps (expensive operation)
        if step % self.estimation_frequency != 0:
            # Return cached values
            return self.last_lambda_max, self.last_sharpness, False
        
        try:
            # Estimate λ_max
            lambda_max = self.estimate_lambda_max(loss)
            
            # Compute sharpness
            sharpness = self.compute_sharpness(learning_rate, lambda_max)
            
            # Update history
            self.lambda_max_history.append(lambda_max)
            self.sharpness_history.append(sharpness)
            
            # Cache values
            self.last_lambda_max = lambda_max
            self.last_sharpness = sharpness
            
            # Check if unstable
            is_unstable = sharpness > self.stability_threshold
            
            if is_unstable and self.verbose:
                logger.warning(
                    f"EDGE OF STABILITY WARNING @ step {step}\n"
                    f"   lambda_max = {lambda_max:.6f}\n"
                    f"   Sharpness = {sharpness:.4f} (threshold: {self.stability_threshold})\n"
                    f"   Recommendation: Reduce learning rate to {learning_rate * 0.5:.2e}"
                )
            
            return lambda_max, sharpness, is_unstable
            
        except RuntimeError as e:
            # Handle CUDA OOM, autograd errors, etc.
            if "out of memory" in str(e).lower():
                logger.error(
                    f"CUDA out of memory during λ_max estimation at step {step}. "
                    "Consider reducing power_iterations or estimation_frequency."
                )
            else:
                logger.error(f"Runtime error estimating λ_max: {e}", exc_info=True)
            return None, None, False
        except ValueError as e:
            logger.error(f"Value error in λ_max estimation: {e}", exc_info=True)
            return None, None, False
        except Exception as e:
            # Unexpected errors should be logged with full traceback
            logger.critical(
                f"Unexpected error in edge detection at step {step}: {e}",
                exc_info=True
            )
            # Re-raise unexpected errors for debugging
            raise
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about stability over training.
        
        Returns:
            Dictionary with statistics:
            - mean_lambda_max: Average λ_max
            - max_lambda_max: Maximum λ_max observed
            - mean_sharpness: Average sharpness
            - max_sharpness: Maximum sharpness observed
            - unstable_steps: Number of steps with sharpness > threshold
        """
        if len(self.lambda_max_history) == 0:
            return {
                "mean_lambda_max": 0.0,
                "max_lambda_max": 0.0,
                "mean_sharpness": 0.0,
                "max_sharpness": 0.0,
                "unstable_steps": 0
            }
        
        return {
            "mean_lambda_max": sum(self.lambda_max_history) / len(self.lambda_max_history),
            "max_lambda_max": max(self.lambda_max_history),
            "mean_sharpness": sum(self.sharpness_history) / len(self.sharpness_history),
            "max_sharpness": max(self.sharpness_history),
            "unstable_steps": sum(1 for s in self.sharpness_history if s > self.stability_threshold)
        }
    
    def recommend_learning_rate(self, current_lr: float, target_sharpness: float = 1.5) -> float:
        """
        Recommend a learning rate that achieves target sharpness.
        
        Given: sharpness = 2 * lr * λ_max
        Solve for lr: lr = target_sharpness / (2 * λ_max)
        
        Args:
            current_lr: Current learning rate
            target_sharpness: Desired sharpness (default: 1.5, safely below 2.0)
            
        Returns:
            Recommended learning rate
        """
        if self.last_lambda_max is None or self.last_lambda_max < 1e-10:
            return current_lr
        
        recommended_lr = target_sharpness / (2.0 * self.last_lambda_max)
        return recommended_lr

