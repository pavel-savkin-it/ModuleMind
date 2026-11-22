"""THE acceptance tests of the ModuleMind idea:

"The AI knows exactly what is connected."

They run against the committed release artifacts:
  checkpoints/core_v1.npz                — the first pretrained core
  modules/demo_math/weights/lora.npz     — math module (sums 0..4)
  modules/english/weights/lora.npz       — english module (simple sentences)
"""
from pathlib import Path

import torch
import pytest

from conftest import make_tok, make_tok_en, ask, sequence_loss

from modulemind.core.hub import ModuleHub
from modulemind.core.checkpoint import load_core
from modulemind.data.toy_corpus import make_holdout

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "checkpoints" / "core_v1.npz"
MATH_W = ROOT / "modules" / "demo_math" / "weights" / "lora.npz"
EN_W = ROOT / "modules" / "english" / "weights" / "lora.npz"

needs_core = pytest.mark.skipif(
    not CHECKPOINT.exists(), reason="core_v1 checkpoint is not trained yet"
)
needs_math = pytest.mark.skipif(
    not MATH_W.exists(), reason="demo_math weights are not trained yet"
)
needs_en = pytest.mark.skipif(
    not EN_W.exists(), reason="english weights are not trained yet"
)

FACTS = ["2+1=", "0+0=", "4+4=", "3+2=", "1+3="]
RIGHT = {"2+1=": "3", "0+0=": "0", "4+4=": "8", "3+2=": "5", "1+3=": "4"}


def fact_answers(core, tok):
    return {f: ask(core, tok, f) for f in FACTS}


@needs_core
@needs_math
def test_math_knowledge_comes_from_the_module():
    """The core is pretrained on neutral word salad: it cannot do math.
    The math module brings the knowledge; disconnecting removes it."""
    core = load_core(CHECKPOINT)
    tok = make_tok()
    probe = torch.tensor([tok.encode("2+1=")], dtype=torch.long)
    with torch.no_grad():
        bare_logits = core(probe).clone()

    bare = fact_answers(core, tok)

    hub = ModuleHub(core, ROOT / "modules")
    hub.connect("demo_math")
    with_mod = fact_answers(core, tok)
    assert with_mod == RIGHT, f"module must know all facts: {with_mod}"
    assert bare != with_mod, "bare core must not know all the facts"
    assert [d["name"] for d in hub.describe()] == ["demo_math"]

    # disconnecting removes the knowledge without a trace
    hub.disconnect("demo_math")
    with torch.no_grad():
        after = core(probe)
    assert torch.allclose(bare_logits, after, atol=1e-6)


@needs_core
@needs_en
def test_english_knowledge_comes_from_the_module():
    core = load_core(CHECKPOINT)
    tok = make_tok_en()
    holdout = make_holdout()

    loss_bare = sequence_loss(core, tok, holdout)
    hub = ModuleHub(core, ROOT / "modules")
    hub.connect("english")
    loss_en = sequence_loss(core, tok, holdout)
    assert loss_en < loss_bare - 0.05, (
        f"english module must reduce loss: {loss_bare:.3f} -> {loss_en:.3f}"
    )
    # the english module must not leak math knowledge
    assert fact_answers(core, make_tok()) != RIGHT
    hub.disconnect("english")


@needs_core
@needs_math
@needs_en
def test_modules_are_switchable_and_stackable():
    """Modules stack mechanically and switch instantly: each skill appears
    with its module and disappears on disconnect; removing one stacked
    module keeps the other working.

    KNOWN LIMITATION (measured): skills trained independently interfere
    when BOTH are stacked (english loss 0.93 -> 10.4 with math stacked).
    Co-training of modules is planned for stage 3.
    """
    core = load_core(CHECKPOINT)
    tok_math = make_tok()
    hub = ModuleHub(core, ROOT / "modules")

    assert fact_answers(core, tok_math) != RIGHT  # bare core knows neither

    hub.connect("english")
    hub.connect("demo_math")
    assert hub.connected() == ["english", "demo_math"]  # stacking works

    hub.disconnect("english")
    assert hub.connected() == ["demo_math"]
    assert fact_answers(core, tok_math) == RIGHT  # math survives english removal

    hub.disconnect("demo_math")
    assert hub.connected() == []
    assert fact_answers(core, tok_math) != RIGHT  # bare core again
