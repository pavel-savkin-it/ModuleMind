# Neuron module format (v0.2)

A module is a directory:

```
modules/<name>/
├── manifest.json          the manifest (below)
└── weights/lora.npz       adapter weights (float32, a small file)
```

Directory name = module name: lowercase latin letters, digits, `_`.

## manifest.json

```json
{
  "name": "demo_math",
  "version": "0.2.0",
  "type": "lora_adapter",
  "description": "What the module adds to the core (concise, English)",
  "targets": ["attn.q", "attn.v", "lm_head"],
  "rank": 8,
  "alpha": 16.0,
  "weights": "weights/lora.npz",
  "core_compat": {"d_model": 96, "n_layers": 2, "vocab_size": 41}
}
```

| Field | Req. | Rules |
|---|---|---|
| `name` | yes | `^[a-z0-9_]+$`, equals the directory name |
| `version` | yes | semantic version string like `1.2.0` |
| `type` | yes | only `lora_adapter` for now |
| `description` | yes | what the AI 'knows' with this module |
| `targets` | yes | non-empty list of core layer names (`attn.q`, `attn.v`, `attn.k`, `attn.o`, `mlp.fc`, `mlp.proj`, `lm_head`) |
| `rank` | yes | integer 1..256 (adapter size) |
| `alpha` | no | scale (defaults to rank); scaling = alpha/rank |
| `weights` | no | path INSIDE the module directory; no `..` |
| `core_compat` | no | expected core parameters; a mismatch logs a warning |

A broken manifest never crashes the system: ModuleHub skips the module and
logs a warning. Several modules may stack on the same `targets` — their
effects add up; disconnecting one leaves the others intact.

## weights/lora.npz

Keys: `<layer path>.A` and `<layer path>.B`.

- `A` — shape `(rank, in_features)`; `B` — shape `(out_features, rank)`.
- All arrays are `float32`.
- `B = 0` → the module changes nothing (safe to connect).

Module contribution: `base(x) + (alpha/rank) · x · Aᵀ · Bᵀ`.

## Core compatibility

Weights must match the current core by shape, otherwise `ModuleError`
(e.g., weights made for `d_model=96` will not load into a `d_model=32` core).
IMPORTANT: matching shapes is necessary but not sufficient — the adapter was
trained under one exact core instance (see ARCHITECTURE.md, invariant 4).
State the release core (`core_v1`) and how to rebuild it in your PR.

## Editing weights by an AI agent (e.g., the ChatGPT agent)

Weights are a plain numpy archive — editable without torch:

```python
import numpy as np

npz = np.load("modules/demo_math/weights/lora.npz")
w = {k: npz[k].copy() for k in npz.files}     # keys: *.A, *.B
w["lm_head.B"] *= 1.1                          # example edit
np.savez_compressed("modules/demo_math/weights/lora.npz", **w)
```

Agent rules: only `float32`, only existing keys and shapes, and after ANY edit
run `python -m pytest tests/test_e2e.py -v` — the acceptance tests must stay
green.
