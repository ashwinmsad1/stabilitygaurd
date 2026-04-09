"""
Example: RLHF/PPO Training with StabilityGuard

This example demonstrates how to use StabilityGuard's RLHF monitoring
features during Reinforcement Learning from Human Feedback training.
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from stabilityguard.rlhf import RLHFGuard
from stabilityguard.core import GuardedOptimizer


class PolicyModel(nn.Module):
    """Simple policy model for demonstration."""
    def __init__(self, vocab_size=1000, hidden_size=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_size, nhead=8, batch_first=True),
            num_layers=4
        )
        self.lm_head = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, input_ids):
        x = self.embedding(input_ids)
        x = self.transformer(x)
        logits = self.lm_head(x)
        return logits


class ValueModel(nn.Module):
    """Simple value model (critic) for demonstration."""
    def __init__(self, vocab_size=1000, hidden_size=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_size, nhead=8, batch_first=True),
            num_layers=2
        )
        self.value_head = nn.Linear(hidden_size, 1)
    
    def forward(self, input_ids):
        x = self.embedding(input_ids)
        x = self.transformer(x)
        values = self.value_head(x).squeeze(-1)
        return values


def main():
    # Setup models
    policy_model = PolicyModel()
    ref_model = PolicyModel()  # Reference model (frozen)
    value_model = ValueModel()
    
    # Copy weights to reference model
    ref_model.load_state_dict(policy_model.state_dict())
    ref_model.eval()
    
    # Setup optimizers with StabilityGuard
    policy_optimizer = AdamW(policy_model.parameters(), lr=1e-5)
    value_optimizer = AdamW(value_model.parameters(), lr=1e-4)
    
    guarded_policy_opt = GuardedOptimizer(
        policy_optimizer,
        policy_model,
        spike_threshold=10.0,
        nan_action="skip"
    )
    
    guarded_value_opt = GuardedOptimizer(
        value_optimizer,
        value_model,
        spike_threshold=10.0,
        nan_action="skip"
    )
    
    # Setup RLHF Guard
    rlhf_guard = RLHFGuard(
        model=policy_model,
        ref_model=ref_model,
        kl_threshold=0.1,              # Maximum KL divergence
        reward_collapse_threshold=0.01, # Minimum reward entropy
        value_divergence_threshold=10.0, # Maximum value divergence
        ppo_clip_threshold=0.2         # PPO clipping threshold
    )
    
    print("=" * 60)
    print("RLHF Training with StabilityGuard")
    print("=" * 60)
    
    # Simulate RLHF training loop
    for step in range(100):
        # Generate dummy batch
        batch_size = 4
        seq_len = 32
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        
        # Forward pass - policy
        policy_logits = policy_model(input_ids)
        with torch.no_grad():
            ref_logits = ref_model(input_ids)
        
        # Forward pass - value
        values = value_model(input_ids)
        
        # Simulate rewards (normally from reward model)
        rewards = torch.randn(batch_size, seq_len)
        
        # Simulate old and new log probabilities for PPO
        old_logprobs = torch.randn(batch_size, seq_len)
        new_logprobs = torch.randn(batch_size, seq_len)
        
        # Check RLHF stability
        issues = rlhf_guard.check_step(
            logits=policy_logits,
            ref_logits=ref_logits,
            rewards=rewards,
            values=values,
            old_logprobs=old_logprobs,
            new_logprobs=new_logprobs,
            step=step
        )
        
        if issues:
            print(f"\nStep {step}: RLHF Issues Detected:")
            for issue in issues:
                print(f"  - {issue}")
        
        # Compute losses (simplified)
        policy_loss = -policy_logits.mean()
        value_loss = (values - rewards).pow(2).mean()
        
        # Backward and optimize
        policy_loss.backward()
        guarded_policy_opt.step()
        guarded_policy_opt.zero_grad()
        
        value_loss.backward()
        guarded_value_opt.step()
        guarded_value_opt.zero_grad()
        
        # Periodic logging
        if step % 10 == 0:
            stats = rlhf_guard.get_stats()
            print(f"\nStep {step}:")
            print(f"  KL Divergence: {stats['kl_divergence']:.4f}")
            print(f"  Reward Entropy: {stats['reward_entropy']:.4f}")
            print(f"  Value Variance: {stats['value_variance']:.4f}")
            print(f"  PPO Clip Freq: {stats['ppo_clip_frequency']:.2%}")
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    
    # Final statistics
    final_stats = rlhf_guard.get_stats()
    print("\nFinal RLHF Statistics:")
    print(f"  Total KL Violations: {final_stats.get('kl_violations', 0)}")
    print(f"  Total Reward Collapses: {final_stats.get('reward_collapses', 0)}")
    print(f"  Total Value Divergences: {final_stats.get('value_divergences', 0)}")
    print(f"  Average PPO Clip Frequency: {final_stats['ppo_clip_frequency']:.2%}")


if __name__ == "__main__":
    main()

# Made with Bob
