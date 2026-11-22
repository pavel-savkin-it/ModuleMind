"""ГЛАВНЫЙ приёмочный тест идеи ModuleMind:

«ИИ знает ровно то, что к нему подключили».
"""
from pathlib import Path

import torch
import pytest

from conftest import make_core, make_tok, ask

from modulemind.core.hub import ModuleHub

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "modules" / "demo_math" / "weights" / "lora.npz"

pytestmark = pytest.mark.skipif(
    not WEIGHTS.exists(), reason="веса demo_math ещё не обучены"
)


def test_knows_exactly_what_is_connected():
    # ВАЖНО: веса demo_math обучены под ядро seed=0 (см. train.py) —
    # модуль подходит только к точно такому же ядру (bit-в-bit).
    core = make_core(d_model=64, seed=0)
    tok = make_tok()
    probe = "2+1="
    ids = torch.tensor([tok.encode(probe)], dtype=torch.long)

    with torch.no_grad():
        bare_logits = core(ids).clone()

    # 1) без модулей ядро не знает ответа
    assert ask(core, tok, probe) != "3"

    hub = ModuleHub(core, ROOT / "modules")

    # 2) подключили модуль математика — ядро отвечает верно
    hub.connect("demo_math")
    assert hub.connected() == ["demo_math"]
    assert ask(core, tok, "2+1=") == "3"
    assert ask(core, tok, "0+0=") == "0"
    assert ask(core, tok, "4+4=") == "8"

    # 3) отключили модуль — знание исчезло, ядро байт-в-байт как было
    hub.disconnect("demo_math")
    assert ask(core, tok, probe) != "3"
    with torch.no_grad():
        after = core(ids)
    assert torch.allclose(bare_logits, after, atol=1e-6)
