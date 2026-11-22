# ModuleMind

## The idea in 10 seconds

We are building a small AI brain.

The brain starts **empty**. It knows NOTHING.

Knowledge comes in small packages. We call them **modules**.

- Connect a module → the AI learns ONE new thing.
- Disconnect it → the AI forgets that thing. Completely.

Example:

- Module "math" is connected → the AI answers `2+1=3`.
- Nothing is connected → the AI answers with garbage.

That is the whole idea.

---

## What is in this repository?

Three things:

1. **The core** — a small program (a mini neural network). This is the
   empty brain. It is already trained and saved in `checkpoints/`.
2. **Two example modules**:
   - `modules/demo_math` — the skill "add numbers 0..4"
   - `modules/english` — the skill "speak simple English"
3. **Code that connects and disconnects modules** at any moment.

A module is just a small file (~30–70 KB). You can make your own.

---

## Try it now (5 minutes, no GPU, weak PC is fine)

Open a terminal. Copy-paste these commands one by one.

**Step 1 — download and install (one time):**

```bash
git clone https://github.com/pavel-savkin-it/ModuleMind.git
cd ModuleMind
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Step 2 — run the tests:**

```bash
python -m pytest -v
```

You should see the line `42 passed` at the bottom.
If you see red errors — tell us, something is broken.

**Step 3 — run the demo:**

```bash
python demo.py
```

This is the fun part. You will see the SAME brain in 4 situations:

| Situation | AI answers "2+1=" | AI speaks English? |
|---|---|---|
| nothing connected | `?` — garbage, it knows nothing | no, gibberish |
| math module connected | `3` — correct! | still gibberish |
| english module connected | `?` — math module is not connected! | yes: `the cat runs in the box` |
| everything disconnected again | `?` — forgot everything | no, gibberish |

One brain. Different modules. Different knowledge.
**This is ModuleMind.**

---

## I want to make my own module

You need: a free Google account (for Google Colab).
No powerful computer. No GPU.

Full guides: `docs/MODULE_GUIDE.md` and `docs/COLAB_TRAINING.md`.

In short:

1. Pick **ONE** thing your module will know. Just one.
   Good examples: "multiplication table", "capital cities", "colors".
2. Write examples in a file. One per line, like this:

   ```
   2*3=6
   4*5=20
   ```

   20–1000 lines is enough.
3. Copy our example module and rename it:

   ```bash
   cp -r modules/demo_math modules/my_skill
   ```

4. Open Google Colab (free in browser), paste our ready-made training cell
   (see `docs/COLAB_TRAINING.md`), wait ~10 minutes.
5. Download one small weights file and put it into your module folder.
6. Done. Connect your module — the AI now knows your skill.

---

## Honest FAQ

**Can two modules be connected at the same time?**
Technically yes. But then the two skills start to interfere (we measured
it). For now it is better to switch modules — it takes a millisecond
and is 100% safe.

**What if I delete a module file?**
The AI forgets that skill. The core is not harmed.

**What if I train a new core?**
Then retrain all modules under the new core. A module only works with the
core it was trained with. Like a cartridge made for one console model.

**Is it a chatbot like ChatGPT?**
Not yet. Today it is a small working PROOF of the idea (42 tests, all
green). Next step: a bigger core trained on real texts.

---

## For developers and AI agents

```
modulemind/core/     core code: tiny_core, lora, hub, tokenizer, checkpoint, pretrain
modulemind/data/     training data generators (deterministic)
modules/             modules. demo_math is the template — copy it
checkpoints/         the trained core (release artifact)
tests/               42 mini-tests. Tests are ALWAYS written before the code
demo.py              live demo. Run it after any change
docs/                all guides
logs/                what happened and when (modulemind.log)
```

Deep documents:

| File | What is inside |
|---|---|
| `docs/ARCHITECTURE.md` | how it works inside, with measured numbers |
| `docs/MODULE_FORMAT.md` | module file format (humans and AI agents can edit weights) |
| `docs/MODULE_GUIDE.md` | make your own module, step by step |
| `docs/COLAB_TRAINING.md` | training on free Colab/Kaggle |
| `docs/HANDOFF.md` | internal continuation document (not public) |

## Status

- [x] Working proof: empty core + modules that add/remove knowledge (42 tests)
- [ ] Bigger core trained on real texts
- [ ] Modules that work together at the same time
- [ ] Automatic checks for community contributions

Reference project: [minimind](https://github.com/jingyaogong/minimind).
