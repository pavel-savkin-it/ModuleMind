"""Core checkpoints: release artifacts of a trained mind.

A checkpoint stores the full core state plus its config, so any machine can
rebuild the exact same core bit-for-bit (modules are tied to it).
"""
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from .config import CoreConfig
from .tiny_core import TinyCore


def save_core(core: TinyCore, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        n: p.detach().cpu().numpy().astype(np.float32)
        for n, p in core.state_dict().items()
    }
    np.savez_compressed(
        path, __config__=np.array(json.dumps(asdict(core.config))), **arrays
    )


def load_core(path) -> TinyCore:
    data = np.load(Path(path), allow_pickle=False)
    cfg = json.loads(str(data["__config__"]))
    core = TinyCore(CoreConfig(**cfg))
    state = {
        k: torch.from_numpy(np.ascontiguousarray(data[k]))
        for k in data.files if k != "__config__"
    }
    core.load_state_dict(state)
    return core
