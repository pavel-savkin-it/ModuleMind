# ModuleMind

**An AI that knows exactly what you plug into it.**

The **core** ("the mind") is a tiny neural network trained from scratch.
On its own it knows NOTHING — it cannot speak, cannot count.

Every skill lives in a separate **module** — a small file of weights:

- plug in the `english` module → the AI speaks English
- plug in the `demo_math` module → the AI does addition
- unplug it → the knowledge is gone, without a trace

Think of a game console: the console (core) does nothing without a
cartridge (module) — and anyone can build a new cartridge.

```
query → CORE (frozen, knows nothing) → [ connected modules ] → answer
```

---

## Choose your path

### Path 1 — Just try it (5 minutes, CPU is enough)

**Step 1.** Get the code and install dependencies:

```bash
git clone https://github.com/pavel-savkin-it/ModuleMind.git
cd ModuleMind
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

**Step 2.** Run the tests — 42 mini-tests, ~7 seconds, ALL must pass:

```bash
python -m pytest -v
# expected last line:  42 passed
```

**Step 3.** Run the live demo:

```bash
python demo.py
```

You will see the SAME core in 4 states (see the table below).

### Path 2 — Train your own module (the community path)

Guides: **[docs/MODULE_GUIDE.md](docs/MODULE_GUIDE.md)** (step by step) and
**[docs/COLAB_TRAINING.md](docs/COLAB_TRAINING.md)** (free Colab/Kaggle).

Short version:

1. Pick **ONE** topic for your module ("multiplication table", "capitals"…).
2. Collect data: 20–1000 examples, one per line (`2+1=3` or plain text).
3. Copy the template: `cp -r modules/demo_math modules/my_topic`
4. Train on free Colab with the template script (~10 minutes, no GPU needed).
5. Add a mini-test (copy `tests/test_demo_math.py`), run `pytest` — all green.
6. Open a PR with your module folder.

### Path 3 — Rebuild the core from scratch (advanced)

```bash
python -m modulemind.core.pretrain     # ~10 min CPU → checkpoints/core_v1.npz
python -m modules.demo_math.train      # retrain modules under YOUR core
python -m modules.english.train
python -m pytest -v                    # all green again
```

**Important:** module weights are tied to the EXACT core weights.
If you rebuild the core, you MUST retrain the modules under it.

---

## What the demo shows

`python demo.py` runs one and the same core in 4 states:

| State | "2+1=" | English sample |
|---|---|---|
| bare core (nothing connected) | `?` — knows nothing | `the cat cat` (word salad) |
| + `english` module | `?` — english adds no math | `the cat runs in the box` |
| + `demo_math` module | `3` — correct | — |
| after disconnecting everything | `?` — knowledge removed without a trace | — |

---

## How it works (30 seconds)

- The core is a tiny transformer (~235K parameters). It is **frozen**:
  module training never changes a single core weight.
- A **module** = `manifest.json` (a few fields) + `weights/lora.npz`
  (a LoRA adapter, ~30–70 KB). Training touches ONLY the module weights.
- **ModuleHub** connects/disconnects modules at runtime; every action is
  logged to `logs/modulemind.log`.
- Full details: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Project structure

```
modulemind/core/     the core: tiny_core, lora, hub, tokenizer, checkpoint, pretrain
modulemind/data/     deterministic toy corpora
modules/             pluggable modules — demo_math is THE template to copy
checkpoints/         core_v1.npz — the release core (trained, committed)
tests/               42 mini-tests (written BEFORE the code, always)
demo.py              live proof: run it after anything changes
docs/                guides for humans and AI agents
logs/                modulemind.log — what happened and when
```

## FAQ

**Can I connect two modules at once?**
Yes, they stack mechanically. But skills trained independently interfere
(measured: english quality drops when math is stacked) — switch modules
instead; instant and lossless. Co-training is planned (roadmap).

**I rebuilt the core and modules give garbage.**
Module weights are tied to the exact core weights. Retrain each module
under your new core (`python -m modules.<name>.train`).

**Where do I see what the AI "knows" right now?**
`hub.describe()` — the list of connected modules with their descriptions.

**Who trains the modules?**
The community, on free Colab/Kaggle, following docs/MODULE_GUIDE.md.
AI agents may also edit module weights directly — rules in
[docs/MODULE_FORMAT.md](docs/MODULE_FORMAT.md).

## Documentation

| Document | About |
|---|---|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | how it all works and why, with measured numbers |
| [docs/MODULE_FORMAT.md](docs/MODULE_FORMAT.md) | module format — for humans and AI agents |
| [docs/MODULE_GUIDE.md](docs/MODULE_GUIDE.md) | build and train your own module, step by step |
| [docs/COLAB_TRAINING.md](docs/COLAB_TRAINING.md) | training on free Colab/Kaggle |
| [docs/HANDOFF.md](docs/HANDOFF.md) | continuation document (kept locally, not in the repo) |

## Status

- [x] Stage 0: MVP — core + LoRA modules + demo_math (31 tests)
- [x] Stage 1a: FIRST core trained from scratch (core_v1) + english module —
      the whole idea proven: knowledge comes only from modules
- [ ] Stage 1b: real pretraining on Colab (real corpus, BPE, bigger core)
- [ ] Stage 2: a real English module on the pretrained core
- [ ] Stage 3: co-trained modules (skills that work simultaneously), CI for PRs

Reference: [minimind](https://github.com/jingyaogong/minimind) — same path,
smaller, with pluggable knowledge.
