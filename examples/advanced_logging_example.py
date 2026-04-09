"""
Example: Advanced Logging with StabilityGuard

This example demonstrates how to use StabilityGuard's advanced logging
features for comprehensive training diagnostics.
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from stabilityguard.logging import AdvancedLogger
from stabilityguard.core import GuardedOptimizer


class CNNModel(nn.Module):
    """Simple CNN model for demonstration."""
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def main():
    print("=" * 60)
    print("Advanced Logging with StabilityGuard")
    print("=" * 60)
    
    # Setup model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNModel().to(device)
    
    # Setup optimizer
    optimizer = AdamW(model.parameters(), lr=1e-3)
    guarded_opt = GuardedOptimizer(
        optimizer,
        model,
        spike_threshold=10.0,
        nan_action="skip"
    )
    
    # Setup advanced logger
    logger = AdvancedLogger(
        log_dir="./advanced_logs",
        enable_gradient_flow=True,
        enable_activation_stats=True,
        enable_weight_updates=True,
        enable_checkpoint_scoring=True,
        gradient_flow_frequency=5,
        activation_stats_frequency=10,
        weight_update_frequency=5
    )
    
    # Register activation hooks
    logger.register_activation_hooks(model)
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Training history for checkpoint scoring
    training_history = []
    
    print("\nStarting training with comprehensive logging...")
    
    # Training loop
    for step in range(100):
        # Generate dummy batch
        batch_size = 32
        inputs = torch.randn(batch_size, 1, 28, 28).to(device)
        targets = torch.randint(0, 10, (batch_size,)).to(device)
        
        # Save previous weights for update tracking
        prev_weights = {
            name: param.data.clone()
            for name, param in model.named_parameters()
        }
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass
        loss.backward()
        
        # Optimizer step
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        guarded_opt.step()
        guarded_opt.zero_grad()
        
        # Log comprehensive statistics
        log_data = logger.log_step(
            step=step,
            loss=loss.item(),
            model=model,
            optimizer=guarded_opt,
            prev_weights=prev_weights,
            grad_norm=grad_norm
        )
        
        # Add to training history
        training_history.append({
            "step": step,
            "loss": loss.item(),
            "grad_norm": grad_norm,
            "spike_detected": False
        })
        
        # Periodic detailed logging
        if step % 10 == 0:
            print(f"\n{'='*60}")
            print(f"Step {step}")
            print(f"{'='*60}")
            print(f"Loss: {loss.item():.4f}")
            print(f"Gradient Norm: {grad_norm:.4f}")
            
            # Gradient flow statistics
            if "gradient_flow" in log_data:
                flow = log_data["gradient_flow"]
                print(f"\nGradient Flow:")
                print(f"  Layers tracked: {len(flow.get('layer_norms', {}))}")
                if flow.get('bottleneck_layers'):
                    print(f"  Bottleneck layers: {flow['bottleneck_layers'][:3]}")
            
            # Activation statistics
            if "activation_stats" in log_data:
                act_stats = log_data["activation_stats"]
                print(f"\nActivation Statistics:")
                print(f"  Layers monitored: {len(act_stats.get('layer_stats', {}))}")
                if act_stats.get('dead_neurons'):
                    print(f"  Dead neurons detected: {act_stats['dead_neurons']}")
            
            # Weight update statistics
            if "weight_updates" in log_data:
                updates = log_data["weight_updates"]
                print(f"\nWeight Updates:")
                print(f"  Parameters updated: {len(updates.get('update_norms', {}))}")
                if updates.get('largest_updates'):
                    print(f"  Largest updates: {updates['largest_updates'][:3]}")
        
        # Score checkpoint every 20 steps
        if step > 0 and step % 20 == 0:
            checkpoint_path = f"checkpoint_step_{step}.pt"
            score = logger.score_checkpoint(
                checkpoint_path,
                training_history[-20:]  # Last 20 steps
            )
            print(f"\nCheckpoint Health Score: {score:.1f}/100")
            
            if score < 50:
                print("  WARNING: Low checkpoint health!")
            elif score < 75:
                print("  INFO: Checkpoint health is moderate")
            else:
                print("  GOOD: Checkpoint health is good")
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    
    # Get comprehensive statistics
    final_stats = logger.get_comprehensive_stats()
    
    print("\nFinal Statistics:")
    print(f"  Total steps logged: {final_stats.get('total_steps', 0)}")
    print(f"  Gradient flow checks: {final_stats.get('gradient_flow_checks', 0)}")
    print(f"  Activation stats collected: {final_stats.get('activation_stats_collected', 0)}")
    print(f"  Weight updates tracked: {final_stats.get('weight_updates_tracked', 0)}")
    print(f"  Checkpoints scored: {final_stats.get('checkpoints_scored', 0)}")
    
    # Score final checkpoint
    final_score = logger.score_checkpoint(
        "final_checkpoint.pt",
        training_history[-50:]  # Last 50 steps
    )
    print(f"\nFinal Checkpoint Health Score: {final_score:.1f}/100")
    
    # Cleanup
    logger.remove_activation_hooks()
    
    print("\nLogs saved to: ./advanced_logs/")


if __name__ == "__main__":
    main()

# Made with Bob
