"""TinyCore — the tiny 'consciousness' of ModuleMind.

A miniature transformer in the spirit of minimind, but far smaller: it
trains and runs on a weak CPU. On its own it knows nothing specific —
knowledge comes exclusively from connected LoRA modules (see lora.py, hub.py).
"""
import math

import torch
import torch.nn as nn

from .config import CoreConfig


class Attention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        H, Dh = self.n_heads, self.d_head

        def split(t):
            return t.view(B, T, H, Dh).transpose(1, 2)  # [B, H, T, Dh]

        q, k, v = split(self.q(x)), split(self.k(x)), split(self.v(x))
        att = (q @ k.transpose(-2, -1)) / math.sqrt(Dh)
        mask = torch.triu(
            torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1
        )
        att = att.masked_fill(mask, float("-inf"))
        att = att.softmax(dim=-1)
        out = (att @ v).transpose(1, 2).reshape(B, T, D)
        return self.o(out)


class MLP(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.fc = nn.Linear(d_model, 4 * d_model)
        self.proj = nn.Linear(4 * d_model, d_model)

    def forward(self, x):
        return self.proj(torch.nn.functional.gelu(self.fc(x)))


class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = Attention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TinyCore(nn.Module):
    """Tiny autoregressive transformer (the language core)."""

    def __init__(self, config: CoreConfig):
        super().__init__()
        self.config = config
        d = config.d_model
        self.tok_emb = nn.Embedding(config.vocab_size, d)
        self.pos_emb = nn.Embedding(config.max_seq, d)
        self.blocks = nn.ModuleList(
            [Block(d, config.n_heads) for _ in range(config.n_layers)]
        )
        self.ln_f = nn.LayerNorm(d)
        self.lm_head = nn.Linear(d, config.vocab_size, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        T = idx.shape[1]
        if T > self.config.max_seq:
            raise ValueError(
                f"sequence length {T} exceeds max_seq={self.config.max_seq}"
            )
        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)[None, :, :]
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.ln_f(x))

    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """Greedy generation: appends max_new_tokens chars to each row."""
        for _ in range(max_new_tokens):
            ctx = idx[:, -self.config.max_seq:]
            next_id = self(ctx)[:, -1].argmax(dim=-1, keepdim=True)
            idx = torch.cat([idx, next_id], dim=1)
        return idx
