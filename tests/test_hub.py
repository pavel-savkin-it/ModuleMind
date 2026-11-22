"""Мини-тесты ModuleHub: обнаружение, подключение и отключение модулей."""
import json
import shutil
from pathlib import Path

import torch
import pytest

from conftest import make_core

from modulemind.core.hub import ModuleHub, ModuleError

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "modules" / "demo_math"
WEIGHTS = DEMO / "weights" / "lora.npz"

# тесты, которым нужны обученные веса демо-модуля
trained = pytest.mark.skipif(not WEIGHTS.exists(), reason="веса demo_math ещё не обучены")


@pytest.fixture
def modules_copy(tmp_path):
    d = tmp_path / "modules"
    shutil.copytree(DEMO, d / "demo_math")
    return d


def test_scan_discovers_demo(core_demo_compat, modules_copy):
    hub = ModuleHub(core_demo_compat, modules_copy)
    assert "demo_math" in hub.scan()


def test_connect_disconnect(core_demo_compat, modules_copy):
    hub = ModuleHub(core_demo_compat, modules_copy)
    assert hub.connected() == []
    hub.connect("demo_math")
    assert hub.connected() == ["demo_math"]
    info = hub.describe()
    assert info[0]["name"] == "demo_math"
    assert info[0]["description"]
    hub.disconnect("demo_math")
    assert hub.connected() == []


def test_connect_unknown_raises(core_demo_compat, modules_copy):
    hub = ModuleHub(core_demo_compat, modules_copy)
    with pytest.raises(ModuleError):
        hub.connect("no_such_module")


def test_double_connect_raises(core_demo_compat, modules_copy):
    hub = ModuleHub(core_demo_compat, modules_copy)
    hub.connect("demo_math")
    with pytest.raises(ModuleError):
        hub.connect("demo_math")


def test_disconnect_unknown_raises(core_demo_compat, modules_copy):
    hub = ModuleHub(core_demo_compat, modules_copy)
    with pytest.raises(ModuleError):
        hub.disconnect("demo_math")


def test_invalid_manifest_skipped(core_demo_compat, modules_copy):
    mf = modules_copy / "demo_math" / "manifest.json"
    data = json.loads(mf.read_text(encoding="utf-8"))
    del data["targets"]
    mf.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    hub = ModuleHub(core_demo_compat, modules_copy)
    assert hub.scan() == []  # битый манифест пропущен без падения
    with pytest.raises(ModuleError):
        hub.connect("demo_math")


def test_modules_can_stack_on_the_same_layers(core_demo_compat, modules_copy):
    """Two modules may share the same target layers; disconnecting one
    leaves the other connected and intact."""
    second = modules_copy / "demo_math_2"
    shutil.copytree(DEMO, second)
    mf = second / "manifest.json"
    data = json.loads(mf.read_text(encoding="utf-8"))
    data["name"] = "demo_math_2"
    mf.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    hub = ModuleHub(core_demo_compat, modules_copy)
    hub.connect("demo_math")
    hub.connect("demo_math_2")
    assert hub.connected() == ["demo_math", "demo_math_2"]
    hub.disconnect("demo_math")
    assert hub.connected() == ["demo_math_2"]
    hub.disconnect("demo_math_2")
    assert hub.connected() == []


def test_module_without_weights_is_identity(core_demo_compat, tmp_path):
    """A module without a weights file connects, but the core behaves as bare."""
    bare = make_core(vocab_size=41, d_model=96, max_seq=48)  # same seed as core_demo_compat
    d = tmp_path / "modules"
    shutil.copytree(DEMO, d / "demo_math")
    shutil.rmtree(d / "demo_math" / "weights", ignore_errors=True)
    hub = ModuleHub(core_demo_compat, d)
    hub.connect("demo_math")
    idx = torch.zeros((1, 6), dtype=torch.long)
    with torch.no_grad():
        assert torch.allclose(bare(idx), core_demo_compat(idx), atol=1e-6)


@trained
def test_incompatible_weights_raise(core, modules_copy):
    """Веса под d_model=64 не должны молча грузиться в ядро d_model=32."""
    hub = ModuleHub(core, modules_copy)
    with pytest.raises(ModuleError):
        hub.connect("demo_math")
