"""Мини-тесты ядра TinyCore."""
import torch

from conftest import make_core


def test_forward_shape():
    core = make_core()
    idx = torch.randint(0, 14, (2, 8))
    assert core(idx).shape == (2, 8, 14)


def test_param_count_tiny():
    core = make_core()
    n = sum(p.numel() for p in core.parameters())
    # ядро обязано быть крошечным: комфортно для CPU и слабых ПК
    assert 1_000 < n < 500_000, f"слишком много параметров: {n}"


def test_generate_grows_sequence():
    core = make_core()
    idx = torch.zeros((1, 3), dtype=torch.long)
    assert core.generate(idx, max_new_tokens=4).shape == (1, 7)


def test_same_seed_same_weights():
    a = make_core(seed=123)
    b = make_core(seed=123)
    idx = torch.randint(0, 14, (1, 6))
    assert torch.allclose(a(idx), b(idx))


def test_positions_within_limit():
    core = make_core(max_seq=8)
    idx = torch.zeros((1, 8), dtype=torch.long)
    assert core(idx).shape == (1, 8, 14)
