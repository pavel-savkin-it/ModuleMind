# ModuleMind Architecture (v0.2 — stage 1a: the first model)

## Components

```
modulemind/core/
  config.py      CoreConfig — core sizes (vocab, d_model, layers, heads)
  tokenizer.py   CharTokenizer — char-level tokenizer (+UNK symbol)
  tiny_core.py   TinyCore — mini transformer: emb → 2 blocks → lm_head
  lora.py        LoRALinear + inject/remove/detach + weights save/load
  hub.py         ModuleHub — discovery, validation, connect/disconnect
  checkpoint.py  save_core/load_core — release artifacts (npz)
  pretrain.py    stage 1: from-scratch pretraining of the core
  log_setup.py   get_logger — logs/modulemind.log + console (WARNING+)
modulemind/data/
  toy_corpus.py  deterministic toy corpora (English + arithmetic)
modules/
  demo_math/     math module (addition 0..4) — the community template
  english/       english module (simple sentences)
checkpoints/
  core_v1.npz    THE first core trained from scratch (release artifact)
```

## Request flow

1. Text → `CharTokenizer.encode` → char ids.
2. `TinyCore.forward` → logits for the next character.
3. Inside every targeted layer: `y = base(x) + (alpha/rank)·x·Aᵀ·Bᵀ` for EACH
   connected module (LoRA).
4. Answer: argmax at the last position → `decode`.

## Key decisions and invariants

1. **The core is untouchable.** `inject_lora` freezes all core parameters;
   gradients are allowed only for `lora_A`/`lora_B`. Knowledge lives only in
   modules — so modules can be plugged and removed without a trace (the e2e
   test checks logits match bit-for-bit after disconnecting).
2. **An untrained module = zero effect.** `lora_B` starts as zeros, so a
   freshly connected untrained module does not change the core at all.
3. **Modules stack.** Several modules may wrap the same core layer; their
   contributions add up. Disconnecting one module rebuilds the chain —
   the others keep working with their weights intact. (A per-layer
   exclusivity mode is reserved for the future.)
4. **Module weights are tied to the exact core.** An adapter is trained under
   one specific core instance (same config AND same weights). The release
   artifact `checkpoints/core_v1.npz` is the single source: rebuilt only via
   `python -m modulemind.core.pretrain` (deterministic: same seed 2026, same
   corpus). Manifests carry `core_compat` (d_model/n_layers/vocab_size); a
   full weight-hash fingerprint is planned for stage 2.
5. **Tests before code.** Every component is covered by mini-tests first
   (~40 tests, under a minute on CPU). A new feature starts as a red test.
6. **Everything is logged.** connect/disconnect/training/errors →
   `logs/modulemind.log` (1 MB rotation × 3). The log always shows what happened.

## Why LoRA (not whole layers, not a router)

| Option | Pros | Cons |
|---|---|---|
| **LoRA (chosen)** | tiny weights, trainable on free Colab, one shared core | a module is tied to its core |
| Layer modules | stronger influence | bigger weights, costlier training |
| Router + micro-nets | full independence | skills do not mix |

Reference: [minimind](https://github.com/jingyaogong/minimind) — a mini LM
trained from scratch on cheap hardware. Our core is the same idea, ~235K
parameters, with pluggable knowledge.

## The first model: what was proven (stage 1a, artifacts trained on Colab)

Measured on the release artifacts (checkpoints/core_v1.npz):

- bare core: english holdout loss 5.98, math facts 0/5 — **the core knows
  nothing** (it was pretrained on neutral word salad only);
- +english module: holdout loss **5.98 → 0.93** — the grammar knowledge
  comes from the module; math facts stay 0/5 (no knowledge leakage);
- +math module: facts **5/5**; disconnecting removes the knowledge without a
  trace (logits match bit-for-bit);
- modules stack mechanically and switch instantly: after removing english
  from a stack, math still answers 5/5;
- KNOWN LIMITATION: with BOTH modules stacked simultaneously their effects
  interfere (english loss rises to ~10.4, math answers degrade) because the
  modules were trained independently. Co-training / routing is stage 3.

e2e tests assert all of the above; `demo.py` prints it live.

## Honest limitations of stage 1a

- The corpus is tiny template English — the core is a proof, not a chatbot.
- Char-level tokenizer; BPE comes with stage 1b/2.
- demo_math is memorized addition (0..4), not general arithmetic.
- Modules are tied to core_v1 — changing the core requires retraining modules.
