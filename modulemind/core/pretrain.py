"""Stage 1: from-scratch pretraining of the TinyCore (the first model).

Trains the FULL core on next-character prediction over a NEUTRAL filler
corpus (word salad: letters and words, no grammar, no math). The core
becomes a machine that can work with text but knows NOTHING specific —
all knowledge arrives later via LoRA modules.

Produces the release artifact checkpoints/core_v1.npz. Run from the repo
root (CPU is fine, ~10 minutes; or on Colab):
    python -m modulemind.core.pretrain
"""
import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modulemind.core.config import CoreConfig            # noqa: E402
from modulemind.core.tiny_core import TinyCore           # noqa: E402
from modulemind.core.tokenizer import CharTokenizer      # noqa: E402
from modulemind.core.checkpoint import save_core         # noqa: E402
from modulemind.core.log_setup import get_logger         # noqa: E402
from modulemind.data.toy_corpus import ALPHABET, make_filler  # noqa: E402


def pretrain_core(core, lines, steps=3000, batch=32, lr=3e-3, seed=0,
                  log_to_console=True):
    """Train the FULL core on next-char prediction. Returns metrics dict."""
    log = get_logger("modulemind.pretrain", log_dir=str(ROOT / "logs"))
    torch.manual_seed(seed)
    tok = CharTokenizer(ALPHABET)
    if core.config.vocab_size != tok.vocab_size:
        raise ValueError(
            f"core vocab_size={core.config.vocab_size} but corpus alphabet "
            f"needs {tok.vocab_size}"
        )
    data = [tok.encode(line) for line in lines]
    opt = torch.optim.AdamW(core.parameters(), lr=lr)

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
        if step % 200 == 0 or step == 1:
            log.info("pretrain step %d/%d loss=%.4f", step, steps, last_loss)
            if log_to_console:
                print(f"  pretrain step {step}/{steps} loss={last_loss:.4f}")

    return {
        "steps": steps, "lines": len(lines),
        "first_loss": round(first_loss, 4), "last_loss": round(last_loss, 4),
    }


def main():
    ap = argparse.ArgumentParser(description="Pretrain core v1 from scratch")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--n-filler", type=int, default=800)
    ap.add_argument("--d-model", type=int, default=96)
    ap.add_argument("--out", default="checkpoints/core_v1.npz")
    args = ap.parse_args()

    tok = CharTokenizer(ALPHABET)
    core = TinyCore(CoreConfig(
        vocab_size=tok.vocab_size, d_model=args.d_model,
        n_layers=2, n_heads=4, max_seq=48, seed=args.seed,
    ))
    n_params = sum(p.numel() for p in core.parameters())

    lines = make_filler(args.n_filler, seed=args.seed)
    print(f"core: {n_params} params, corpus: {len(lines)} filler lines")
    res = pretrain_core(core, lines, steps=args.steps, batch=args.batch,
                        lr=args.lr, seed=args.seed)

    save_core(core, ROOT / args.out)
    meta = {
        "release": "core_v1", "params": n_params,
        "config": {"vocab_size": tok.vocab_size, "d_model": args.d_model,
                   "n_layers": 2, "n_heads": 4, "max_seq": 48,
                   "seed": args.seed},
        "corpus": {"filler_lines": args.n_filler, "alphabet": ALPHABET},
        **res,
    }
    meta_path = ROOT / args.out.replace(".npz", ".json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
