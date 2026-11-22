"""Обучение демо-модуля demo_math: таблица сложения 0..4.

Обучается ТОЛЬКО LoRA-адаптер (ядро заморожено) — поэтому это и есть
«знание модуля». Всё считается на CPU за секунды.

Запуск из корня репозитория:
    python -m modules.demo_math.train --steps 400
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modulemind.core.config import CoreConfig            # noqa: E402
from modulemind.core.tokenizer import CharTokenizer      # noqa: E402
from modulemind.core.tiny_core import TinyCore           # noqa: E402
from modulemind.core.lora import inject_lora, lora_state_dict  # noqa: E402
from modulemind.core.log_setup import get_logger         # noqa: E402

MODULE_DIR = Path(__file__).resolve().parent
ALPHABET = "0123456789+= "
TARGETS = ["attn.q", "attn.v", "lm_head"]


def make_facts(max_digit: int = 4):
    """Все факты «a+b=c» для a, b из 0..max_digit (ответ — одна цифра)."""
    return [f"{a}+{b}={a + b}" for a in range(max_digit + 1)
            for b in range(max_digit + 1)]


def _answer_accuracy(core, tok, facts):
    """Доля фактов, где ядро сразу после «=» выдаёт верную цифру."""
    ok = 0
    for f in facts:
        ids = torch.tensor([tok.encode(f.split("=")[0] + "=")])
        with torch.no_grad():
            pred = int(core(ids)[0, -1].argmax())
        ok += int(tok.decode([pred]) == f.split("=")[1])
    return ok / len(facts)


def train_demo(steps=400, batch=32, lr=1e-2, seed=0, max_digit=4,
               rank=8, alpha=16.0, out_dir=None, log_to_console=True):
    """Обучить адаптер на фактах сложения; вернуть метрики и сохранить веса."""
    log = get_logger("modulemind.demo_train", log_dir=str(ROOT / "logs"))
    torch.manual_seed(seed)

    tok = CharTokenizer(ALPHABET)
    core = TinyCore(CoreConfig(vocab_size=tok.vocab_size, d_model=64,
                               n_layers=2, n_heads=4, max_seq=16))
    facts = make_facts(max_digit)
    baseline_acc = _answer_accuracy(core, tok, facts)

    handles = inject_lora(core, TARGETS, rank, alpha)
    params = [p for n, p in core.named_parameters() if "lora" in n]
    opt = torch.optim.AdamW(params, lr=lr)

    data = [tok.encode(f) for f in facts]
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
        logits = core(x).reshape(-1, tok.vocab_size)
        loss = torch.nn.functional.cross_entropy(logits, y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if first_loss is None:
            first_loss = float(loss.detach())
        last_loss = float(loss.detach())
        if step % 50 == 0 or step == 1:
            log.info("шаг %d/%d loss=%.4f", step, steps, last_loss)
            if log_to_console:
                print(f"  шаг {step}/{steps} loss={last_loss:.4f}")

    answer_acc = _answer_accuracy(core, tok, facts)
    out_dir = Path(out_dir) if out_dir else MODULE_DIR / "weights"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "lora.npz", **lora_state_dict(handles))

    result = {
        "steps": steps, "rank": rank, "facts": len(facts),
        "first_loss": round(first_loss, 4), "last_loss": round(last_loss, 4),
        "baseline_acc": round(baseline_acc, 3), "answer_acc": round(answer_acc, 3),
        "weights": str(out_dir / "lora.npz"),
    }
    log.info("обучение завершено: %s", json.dumps(result, ensure_ascii=False))
    return result


def main():
    ap = argparse.ArgumentParser(description="Обучение демо-модуля demo_math")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-digit", type=int, default=4)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=16.0)
    args = ap.parse_args()
    res = train_demo(steps=args.steps, batch=args.batch, lr=args.lr,
                     seed=args.seed, max_digit=args.max_digit,
                     rank=args.rank, alpha=args.alpha)
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
