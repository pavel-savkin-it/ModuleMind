"""Live proof of ModuleMind: the same core, different knowledge.

Loads the release core (checkpoints/core_v1.npz), then shows what the AI
does without modules, with the english module, and with the math module.
Disconnecting removes the knowledge without a trace.

Run from the repo root:
    python demo.py
"""
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modulemind.core.checkpoint import load_core        # noqa: E402
from modulemind.core.hub import ModuleHub               # noqa: E402
from modulemind.core.tokenizer import CharTokenizer     # noqa: E402
from modulemind.data.toy_corpus import ALPHABET         # noqa: E402

CHECKPOINT = ROOT / "checkpoints" / "core_v1.npz"
MODULES = ROOT / "modules"
ALPHABET_MATH = "0123456789+= "
FACTS = ["2+1=", "0+0=", "4+4=", "3+2=", "1+3="]
RIGHT = {"2+1=": "3", "0+0=": "0", "4+4=": "8", "3+2=": "5", "1+3=": "4"}
PROMPT = "the cat"


def _sample(core, tok, prompt, n=40):
    ids = torch.tensor([tok.encode(prompt)], dtype=torch.long)
    out = core.generate(ids, max_new_tokens=n)[0].tolist()
    return tok.decode(out)


def _facts(core, tok):
    answers = {}
    for f in FACTS:
        ids = torch.tensor([tok.encode(f)], dtype=torch.long)
        with torch.no_grad():
            answers[f] = tok.decode([int(core(ids)[0, -1].argmax())])
    return answers


def run_demo():
    core = load_core(CHECKPOINT)
    tok_en = CharTokenizer(ALPHABET)
    tok_math = CharTokenizer(ALPHABET_MATH)
    hub = ModuleHub(core, MODULES, log_dir=str(ROOT / "logs"))

    report = {
        "core": "core_v1",
        "bare": {
            "connected": [],
            "english_sample": _sample(core, tok_en, PROMPT),
            "math_answers": _facts(core, tok_math),
        },
    }

    hub.connect("english")
    report["english_connected"] = {
        "connected": hub.connected(),
        "knowledge": [f"{d['name']}: {d['description']}" for d in hub.describe()],
        "english_sample": _sample(core, tok_en, PROMPT),
        "math_answers": _facts(core, tok_math),
    }
    hub.disconnect("english")

    hub.connect("demo_math")
    answers = _facts(core, tok_math)
    report["math_connected"] = {
        "connected": hub.connected(),
        "math_answers": answers,
        "all_correct": answers == RIGHT,
    }
    hub.disconnect("demo_math")

    report["after_disconnect"] = {"connected": hub.connected()}
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


if __name__ == "__main__":
    run_demo()
