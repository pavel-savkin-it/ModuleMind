# Community guide: how to build your own neuron module

Your goal: train a small adapter that adds ONE clear piece of knowledge or
skill to the core. One module = one topic ("multiplication table", "colors",
"english words"...). Everything runs on CPU or free Colab.

## Step 1. Pick a task and collect data

- The task must reduce to 'query → answer' pairs (like demo_math: `2+1=3`)
  or to plain text in one style (like english: simple sentences).
- Collect 20–1000 pairs (or a small text corpus) — one line per example.
- Use the smallest alphabet that covers your data.

## Step 2. Copy the template

```bash
cp -r modules/demo_math modules/my_topic
rm modules/my_topic/weights/lora.npz
```

Edit `manifest.json` (name, description, rank). Do not change `targets`
until you understand what they mean (see docs/ARCHITECTURE.md).

## Step 3. Train

Use `modules/demo_math/train.py` or `modules/english/train.py` as the
pattern: replace the facts/corpus with your data and keep the core frozen.
Train under the release core `checkpoints/core_v1.npz` (the default) —
modules are tied to the exact core weights.

Train locally (small tasks) or on Colab/Kaggle — see
[COLAB_TRAINING.md](COLAB_TRAINING.md). Readiness criterion: accuracy on
your own examples ≥ 0.9.

## Step 4. Verify (mandatory)

```bash
python -m pytest -v          # all tests must stay green
```

Add a mini-test modeled on `tests/test_demo_math.py`: train your module
inside the test (short run) and assert the metric improves. Tests are written
BEFORE the code — that is the project process.

## Step 5. Describe and open a PR

- `manifest.json`: an honest `description` (what exactly the AI knows with
  the module), `core_compat`, the release core you trained under.
- In the PR: a before/after table (metric without the module vs with it).
- Review checks: manifest valid, tests green, weights npz < 5 MB.

## Rules of good taste

1. One module — one topic. Do not try to teach everything at once.
2. Say it honestly in `description` if the module answers garbage outside
   its topic (the core has no way to 'reject' out-of-domain queries yet).
3. The module weights must come from YOUR data. No third-party corpora or models.
4. Small files: manifest ≤ 2 KB, weights ≤ 5 MB.
