"""
Example: Llama-style fine-tuning with StabilityGuard.

Demonstrates using GuardedOptimizer with rollback action for
supervised fine-tuning (SFT) of a Llama-style model.

Usage:
    pip install stabilityguard
    python llama_finetune.py
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from stabilityguard import GuardedOptimizer


class RMSNorm(nn.Module):
    """Llama-style RMSNorm."""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight


class LlamaBlock(nn.Module):
    """Simplified Llama transformer block."""
    def __init__(self, d_model=256, nhead=4, dim_ff=512):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.norm1 = RMSNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff, bias=False),
            nn.SiLU(),
            nn.Linear(dim_ff, d_model, bias=False),
        )
        self.norm2 = RMSNorm(d_model)

    def forward(self, x):
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h)
        x = x + attn_out
        x = x + self.ff(self.norm2(x))
        return x


class MiniLlama(nn.Module):
    def __init__(self, vocab_size=2000, d_model=256, n_layers=6, nhead=4):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([
            LlamaBlock(d_model, nhead) for _ in range(n_layers)
        ])
        self.norm = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x):
        x = self.embed(x)
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x))


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Model
    model = MiniLlama(vocab_size=2000, d_model=256, n_layers=6).to(device)
    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"MiniLlama: {param_count:.1f}M parameters")

    # Optimizer with rollback protection
    base_opt = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    optimizer = GuardedOptimizer(
        base_opt,
        model,
        spike_threshold=8.0,     # more aggressive for fine-tuning
        nan_action="rollback",   # rollback to checkpoint on spike
        log_every=25,
        warmup_steps=5,
    )

    # Synthetic SFT training loop
    criterion = nn.CrossEntropyLoss()

    for step in range(100):
        # Simulate instruction-following data
        x = torch.randint(0, 2000, (4, 128), device=device)
        target = torch.randint(0, 2000, (4, 128), device=device)

        logits = model(x)
        loss = criterion(logits.view(-1, 2000), target.view(-1))
        loss.backward()

        optimizer.step(loss=loss.item())
        optimizer.zero_grad()

        if step % 25 == 0:
            print(f"[SFT] Step {step}: loss={loss.item():.4f}")

    print(f"\nFine-tuning complete - {optimizer.total_spikes} spikes, "
          f"{optimizer.total_skips} rollbacks")
    optimizer.close()


if __name__ == "__main__":
    main()
