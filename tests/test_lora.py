"""Мини-тесты LoRA-адаптеров (механизм «нейрона»)."""
import torch

from conftest import make_core

from modulemind.core.lora import LoRALinear, inject_lora, remove_lora, detach_lora


def test_untrained_lora_is_identity():
    """Свежеподключённый (необученный) модуль не меняет поведение ядра."""
    bare = make_core(seed=7)
    lora = make_core(seed=7)
    inject_lora(lora, targets=["attn.q", "attn.v", "lm_head"], rank=4)
    idx = torch.randint(0, 14, (1, 8))
    assert torch.allclose(bare(idx), lora(idx), atol=1e-6)


def test_weights_change_output():
    bare = make_core(seed=7)
    lora = make_core(seed=7)
    handles = inject_lora(lora, targets=["attn.q"], rank=4)
    with torch.no_grad():
        list(handles.values())[0].lora_B.normal_(0.0, 0.5)
    idx = torch.randint(0, 14, (1, 8))
    assert not torch.allclose(bare(idx), lora(idx))


def test_only_targets_wrapped():
    core = make_core()
    inject_lora(core, targets=["attn.q"], rank=4)
    names = [n for n, m in core.named_modules() if isinstance(m, LoRALinear)]
    assert sorted(names) == ["blocks.0.attn.q", "blocks.1.attn.q"]


def test_remove_lora_restores_module():
    core = make_core(seed=7)
    idx = torch.zeros((1, 4), dtype=torch.long)
    with torch.no_grad():
        bare_logits = core(idx)
    handles = inject_lora(core, targets=["lm_head"], rank=4)
    remove_lora(core, handles)
    names = [n for n, m in core.named_modules() if isinstance(m, LoRALinear)]
    assert names == []
    with torch.no_grad():
        assert torch.allclose(bare_logits, core(idx))


def test_detach_lora_removes_all():
    core = make_core(seed=7)
    inject_lora(core, targets=["attn.q", "attn.v"], rank=4)
    assert detach_lora(core) == 4
    assert all(not isinstance(m, LoRALinear) for m in core.modules())


def test_modules_stack_on_the_same_layer():
    """Two adapters on one layer add up; removing the top restores the bottom."""
    core = make_core(seed=7)
    idx = torch.zeros((1, 4), dtype=torch.long)
    with torch.no_grad():
        bare = core(idx).clone()

    h1 = inject_lora(core, targets=["lm_head"], rank=4)
    with torch.no_grad():
        h1["lm_head"].lora_B.normal_(0.0, 0.5)
    with torch.no_grad():
        one = core(idx).clone()
    assert not torch.allclose(bare, one)

    h2 = inject_lora(core, targets=["lm_head"], rank=4)  # stacks on top of h1
    with torch.no_grad():
        h2["lm_head"].lora_B.normal_(0.0, 0.5)
    with torch.no_grad():
        two = core(idx).clone()
    assert not torch.allclose(one, two)

    remove_lora(core, h2)  # top module leaves, bottom keeps working
    with torch.no_grad():
        assert torch.allclose(one, core(idx))

    assert detach_lora(core) == 1  # only the bottom wrapper remains
    with torch.no_grad():
        assert torch.allclose(bare, core(idx))


def test_base_weights_are_frozen():
    core = make_core()
    inject_lora(core, targets=["lm_head"], rank=4)
    for n, p in core.named_parameters():
        if "lora" not in n:
            assert not p.requires_grad, f"базовый вес заморожен, но требует градиента: {n}"
