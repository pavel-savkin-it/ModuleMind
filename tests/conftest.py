"""Общие фикстуры и помощники для мини-тестов ModuleMind."""
import torch
import pytest

from modulemind.core.config import CoreConfig
from modulemind.core.tokenizer import CharTokenizer
from modulemind.core.tiny_core import TinyCore

# На слабых машинах один поток быстрее для крошечных моделей
torch.set_num_threads(1)

ALPHABET = "0123456789+= "


def make_core(vocab_size=14, d_model=32, n_layers=2, n_heads=4, max_seq=16, seed=42):
    """Создать крошечное ядро с фиксированным семенем (детерминированные тесты)."""
    torch.manual_seed(seed)
    cfg = CoreConfig(
        vocab_size=vocab_size, d_model=d_model,
        n_layers=n_layers, n_heads=n_heads, max_seq=max_seq,
    )
    return TinyCore(cfg)


def make_tok():
    return CharTokenizer(ALPHABET)


def ask(core, tok, text):
    """Задать ядру вопрос одним символом ответа (argmax по последней позиции)."""
    ids = torch.tensor([tok.encode(text)], dtype=torch.long)
    with torch.no_grad():
        logits = core(ids)
    return tok.decode([int(logits[0, -1].argmax())])


@pytest.fixture
def tok():
    return make_tok()


@pytest.fixture
def core():
    return make_core()


@pytest.fixture
def core_demo_compat():
    """Ядро, совместимое с весами demo_math (d_model=64)."""
    return make_core(d_model=64)
