"""
Example: GPT-2 pretraining with StabilityGuard.

Demonstrates wrapping a standard PyTorch training loop with
GuardedOptimizer for gradient spike detection.

Usage:
    pip install stabilityguard
    python gpt2_pretrain.py
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from stabilityguard import GuardedOptimizer


# Simple Transformer Block (GPT-2 style)
class TransformerBlock(nn.Module):
    def __init__(self, d_model=256, nhead=4, dim_ff=512):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Linear(dim_ff, d_model),
        )
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = self.ln1(x + attn_out)
        x = self.ln2(x + self.ff(x))
        return x


class MiniGPT(nn.Module):
    def __init__(self, vocab_size=1000, d_model=256, n_layers=4, nhead=4):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(128, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, nhead) for _ in range(n_layers)
        ])
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        x = self.embed(x) + self.pos_embed(pos)
        for block in self.blocks:
            x = block(x)
        return self.head(x)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on {device}")

    # Model setup
    model = MiniGPT(vocab_size=1000, d_model=256, n_layers=4).to(device)
    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model parameters: {param_count:.1f}M")

    # Standard optimizer
    base_opt = AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

    # Wrap with StabilityGuard (the one-line change)
    optimizer = GuardedOptimizer(
        base_opt,
        model,
        spike_threshold=10.0,    # 10x deviation = spike
        nan_action="skip",       # skip corrupted steps
        log_every=20,            # summary every 20 steps
    )

    # Training loop
    criterion = nn.CrossEntropyLoss()

    for step in range(200):
        # Synthetic data (replace with real data in production)
        x = torch.randint(0, 1000, (8, 64), device=device)
        target = torch.randint(0, 1000, (8, 64), device=device)

        # Forward
        logits = model(x)
        loss = criterion(logits.view(-1, 1000), target.view(-1))

        # Backward
        loss.backward()

        # Step — GuardedOptimizer intercepts here
        optimizer.step(loss=loss.item())
        optimizer.zero_grad()

        if step % 20 == 0:
            print(f"Step {step}: loss={loss.item():.4f}")

    print(f"\nTraining complete - {optimizer.total_spikes} spikes detected, "
          f"{optimizer.total_skips} steps skipped")
    optimizer.close()


if __name__ == "__main__":
    main()
