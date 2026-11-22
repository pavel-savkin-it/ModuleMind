"""LoRA adapters — the 'neuron' mechanism of ModuleMind.

A module = small trainable matrices A (rank x in) and B (out x rank) placed
on top of selected core layers. A fresh module is initialized so its
contribution is exactly zero: connecting an untrained module does not
change the core's behavior at all.
"""
import logging

import numpy as np
import torch
import torch.nn as nn

log = logging.getLogger("modulemind.lora")


class LoRALinear(nn.Module):
    """Wrapper over nn.Linear: base(x) + scaling * (x @ A^T @ B^T)."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError("LoRA can only wrap nn.Linear")
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.rank = rank
        self.scaling = alpha / rank
        self.lora_A = nn.Parameter(torch.randn(rank, base.in_features) * 0.02)
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank))

    def forward(self, x):
        return self.base(x) + self.scaling * (x @ self.lora_A.T @ self.lora_B.T)


def _replace(root: nn.Module, path: str, new_module: nn.Module):
    parent = root
    parts = path.split(".")
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new_module)


def _matches(name: str, targets) -> bool:
    return any(name == t or name.endswith("." + t) for t in targets)


def inject_lora(model, targets, rank: int, alpha=None):
    """Place LoRA on top of the layers named in targets.

    Returns dict {layer_path: LoRALinear}. Freezes every non-LoRA parameter:
    the core is untouchable, gradients live only in the adapters.
    """
    if alpha is None:
        alpha = float(rank)
    handles = {}
    for name, mod in list(model.named_modules()):
        if isinstance(mod, nn.Linear) and _matches(name, targets):
            lora = LoRALinear(mod, rank, alpha)
            _replace(model, name, lora)
            handles[name] = lora
    if not handles:
        raise ValueError(f"LoRA: none of the targets found in the core: {targets}")
    for n, p in model.named_parameters():
        if "lora" not in n:
            p.requires_grad_(False)
    log.info("LoRA injected into %d layers: %s", len(handles), sorted(handles))
    return handles


def remove_lora(model, handles) -> int:
    """Unwrap the given LoRA wrappers, restoring the original layers."""
    for path, lora in handles.items():
        _replace(model, path, lora.base)
    log.info("LoRA removed from %d layers", len(handles))
    return len(handles)


def detach_lora(model) -> int:
    """Remove ALL LoRA wrappers from the model. Returns how many."""
    n = 0
    for name, mod in list(model.named_modules()):
        if isinstance(mod, LoRALinear):
            _replace(model, name, mod.base)
            n += 1
    return n


def lora_state_dict(handles) -> dict:
    """Adapter weights -> {'<path>.A'/'<path>.B': float32 ndarray}."""
    out = {}
    for path, h in handles.items():
        out[f"{path}.A"] = h.lora_A.detach().cpu().numpy().astype(np.float32)
        out[f"{path}.B"] = h.lora_B.detach().cpu().numpy().astype(np.float32)
    return out


def load_lora_state(handles, state):
    """Load weights into existing adapters with strict shape checks."""
    for path, h in handles.items():
        for key, param, shape in (
            (f"{path}.A", h.lora_A, h.lora_A.shape),
            (f"{path}.B", h.lora_B, h.lora_B.shape),
        ):
            if key not in state:
                raise KeyError(f"weights miss key '{key}'")
            arr = np.asarray(state[key])
            if tuple(arr.shape) != tuple(shape):
                raise ValueError(
                    f"{key}: shape {arr.shape} does not match expected {tuple(shape)}"
                )
            with torch.no_grad():
                param.copy_(torch.from_numpy(np.ascontiguousarray(arr)))
    log.info("loaded weights for %d LoRA layers", len(handles))
