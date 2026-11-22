# ModuleMind

**A modular AI: a core-'consciousness' + pluggable neuron modules.**

On its own the core knows nothing specific. Connect the 'math' module — the AI
becomes a mathematician. Connect the 'english' module — it speaks English.
Disconnect — the knowledge is gone. **The AI knows exactly what you plugged in.**

```
query → CORE (TinyCore, frozen) → [LoRA modules: ✅ math ✅ english …] → answer
```

- **Core (TinyCore)** — a tiny transformer, untouchable: gradients are allowed
  only inside modules.
- **Neuron module** = `manifest.json` + a small weights file `lora.npz`
  (a LoRA adapter). The community trains modules on Colab/Kaggle following our
  [guide](docs/MODULE_GUIDE.md).
- **ModuleHub** — connects/disconnects modules; every action is logged.

## The first model (proof it all works)

`checkpoints/core_v1.npz` — our first core, **trained from scratch on CPU**
(~235K params) on a NEUTRAL word-salad corpus: it learns to work with text
but knows **neither grammar nor math**. Knowledge comes only from modules:

| Module | What the AI gets | Proof |
|---|---|---|
| `demo_math` | answers `a+b=` for digits 0..4 | bare core: 0/5 → with module: 5/5 |
| `english` | speaks simple English sentences | fresh-holdout loss drops measurably |

Modules stack: with both connected the AI speaks English AND does math;
disconnecting one leaves the other intact.

Live proof:

```bash
python demo.py        # bare core → word salad; +english → sentences; +math → sums
python -m pytest -v   # all mini-tests incl. acceptance e2e
```

## Quick start (CPU, a weak PC is fine)

```bash
git clone git@github.com:pavel-savkin-it/ModuleMind.git
cd ModuleMind
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# rebuild the first core from scratch (~10 min CPU) — or use the committed one
python -m modulemind.core.pretrain

# (re)train the modules under your core
python -m modules.demo_math.train
python -m modules.english.train

python -m pytest -v
```

## Connect a module — 6 lines

```python
import torch
from modulemind.core import ModuleHub
from modulemind.core.checkpoint import load_core
from modulemind.core.tokenizer import CharTokenizer

core = load_core("checkpoints/core_v1.npz")
tok = CharTokenizer("0123456789+= ")
hub = ModuleHub(core, "modules")

hub.connect("demo_math")            # ← the AI became a mathematician
print(hub.describe())               # what the AI knows right now
ids = torch.tensor([tok.encode("2+1=")])
print(tok.decode([int(core(ids)[0, -1].argmax())]))   # → "3"
```

## Structure

```
modulemind/core/     core: tiny_core, lora, hub, tokenizer, checkpoint, pretrain
modulemind/data/     toy corpora (deterministic)
modules/             pluggable modules (english, demo_math — the template)
checkpoints/         core_v1.npz — the release core (+ meta json)
tests/               mini-tests (tests are always written BEFORE the code)
docs/                architecture, module format, community guide, Colab, HANDOFF
logs/                modulemind.log (not committed)
```

## Documentation

| Document | About |
|---|---|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | how it all works and why |
| [docs/MODULE_FORMAT.md](docs/MODULE_FORMAT.md) | module format (for humans and AI agents) |
| [docs/MODULE_GUIDE.md](docs/MODULE_GUIDE.md) | how the community builds and trains a module |
| [docs/COLAB_TRAINING.md](docs/COLAB_TRAINING.md) | training modules on Colab/Kaggle |
| [docs/HANDOFF.md](docs/HANDOFF.md) | continuation document (if the agent session drops) |

## Status

- [x] Stage 0: MVP — core + LoRA modules + demo_math (random core, 31 tests)
- [x] Stage 1a: the FIRST core trained from scratch (core_v1) + english module —
      the full idea proven locally on CPU
- [ ] Stage 1b: large-scale pretraining on Colab/Kaggle (real corpus, bigger core)
- [ ] Stage 2: a real English-language module on the pretrained core (BPE tokenizer)
- [ ] Stage 3: CI for community PRs, merging multiple modules

Reference approach: [minimind](https://github.com/jingyaogong/minimind) — we walk
the same path but smaller, with pluggable knowledge.
