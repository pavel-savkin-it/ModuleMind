"""Training the english module: the model speaks simple English.

ONLY the LoRA adapter is trained (the core stays frozen). The module is
trained under the release core checkpoints/core_v1.npz — modules are tied
to the exact core weights.

Run from the repo root (CPU, a few minutes):
    python -m modules.english.train
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modulemind.core.lora import inject_lora, lora_state_dict  # noqa: E402
from modulemind.core.checkpoint import load_core               # noqa: E402
from modulemind.core.tokenizer import CharTokenizer            # noqa: E402
from modulemind.core.tiny_core import TinyCore                 # noqa: E402
from modulemind.core.config import CoreConfig                  # noqa: E402
from modulemind.core.log_setup import get_logger               # noqa: E402
from modulemind.data.toy_corpus import (                       # noqa: E402
    ALPHABET, make_english, make_holdout,
)

MODULE_DIR = Path(__file__).resolve().parent
TARGETS = ["attn.q", "attn.v", "attn.o"]
DEFAULT_CORE = ROOT / "checkpoints" / "core_v1.npz"
HOLDOUT = make_holdout()


def _load_default_core(tok):
    if DEFAULT_CORE.exists():
        return load_core(DEFAULT_CORE), "core_v1"
    return TinyCore(CoreConfig(vocab_size=tok.vocab_size, d_model=64,
                               n_layers=2, n_heads=4, max_seq=48, seed=0)), "fresh"


def _holdout_loss(core, tok):
    """Mean next-char CE over the fresh holdout lines (reserved subjects)."""
    total, ntok = 0.0, 0
    with torch.no_grad():
        for line in HOLDOUT:
            ids = tok.encode(line)
            x = torch.tensor([ids[:-1]], dtype=torch.long)
            y = torch.tensor(ids[1:], dtype=torch.long)
            logits = core(x)[0]
            total += float(F.cross_entropy(logits, y, reduction="sum"))
            ntok += y.numel()
    return total / max(ntok, 1)


def train_english(steps=2500, batch=32, lr=3e-3, seed=0, rank=16, alpha=16.0,
                  out_dir=None, log_to_console=True, core=None):
    """Train the english adapter; return metrics and save weights."""
    log = get_logger("modulemind.english_train", log_dir=str(ROOT / "logs"))
    torch.manual_seed(seed)

    tok = CharTokenizer(ALPHABET)
    if core is None:
        core, core_kind = _load_default_core(tok)
    else:
        core_kind = "provided"
    loss_bare = _holdout_loss(core, tok)

    handles = inject_lora(core, TARGETS, rank, alpha)
    params = [p for n, p in core.named_parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr)

    data = [tok.encode(line) for line in make_english(800, seed=7)]
    first_loss = last_loss = None
    for step in range(1, steps + 1):
        pick = torch.randint(0, len(data), (batch,))
        seqs = [data[i] for i in pick.tolist()]
        L = max(len(s) for s in seqs) - 1
        x = torch.zeros(len(seqs), L, dtype=torch.long)
        y = torch.zeros(len(seqs), L, dtype=torch.long)
        for r, s in enumerate(seqs):
            x[r, :len(s) - 1] = torch.tensor(s[:-1])
            y[r, :len(s) - 1] = torch.tensor(s[1:])
        logits = core(x).reshape(-1, core.config.vocab_size)
        loss = F.cross_entropy(logits, y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if first_loss is None:
            first_loss = float(loss.detach())
        last_loss = float(loss.detach())
        if step % 100 == 0 or step == 1:
            log.info("step %d/%d loss=%.4f", step, steps, last_loss)
            if log_to_console:
                print(f"  step {step}/{steps} loss={last_loss:.4f}")

    loss_module = _holdout_loss(core, tok)
    out_dir = Path(out_dir) if out_dir else MODULE_DIR / "weights"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "lora.npz", **lora_state_dict(handles))

    result = {
        "steps": steps, "rank": rank, "core": core_kind,
        "train_first_loss": round(first_loss, 4),
        "train_last_loss": round(last_loss, 4),
        "holdout_loss_bare": round(loss_bare, 4),
        "holdout_loss_module": round(loss_module, 4),
        "weights": str(out_dir / "lora.npz"),
    }
    log.info("training finished: %s", json.dumps(result, ensure_ascii=False))
    return result


def main():
    ap = argparse.ArgumentParser(description="Train the english module")
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=16.0)
    args = ap.parse_args()
    res = train_english(steps=args.steps, batch=args.batch, lr=args.lr,
                        seed=args.seed, rank=args.rank, alpha=args.alpha)
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
