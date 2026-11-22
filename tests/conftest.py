"""Shared fixtures and helpers for ModuleMind mini-tests."""
import torch
import torch.nn.functional as F
import pytest

from modulemind.core.config import CoreConfig
from modulemind.core.tokenizer import CharTokenizer
from modulemind.core.tiny_core import TinyCore

# Single thread is faster for tiny models on weak machines
torch.set_num_threads(1)

ALPHABET = "0123456789+= "
ALPHABET_EN = "abcdefghijklmnopqrstuvwxyz .0123456789+="


def make_core(vocab_size=14, d_model=32, n_layers=2, n_heads=4, max_seq=16, seed=42):
    """Create a tiny core with a fixed seed (deterministic tests)."""
    torch.manual_seed(seed)
    cfg = CoreConfig(
        vocab_size=vocab_size, d_model=d_model,
        n_layers=n_layers, n_heads=n_heads, max_seq=max_seq,
    )
    return TinyCore(cfg)


def make_tok():
    return CharTokenizer(ALPHABET)


def make_tok_en():
    return CharTokenizer(ALPHABET_EN)


def ask(core, tok, text):
    """Ask the core a one-character answer (argmax at the last position)."""
    ids = torch.tensor([tok.encode(text)], dtype=torch.long)
    with torch.no_grad():
        logits = core(ids)
    return tok.decode([int(logits[0, -1].argmax())])


def sequence_loss(core, tok, lines):
    """Mean cross-entropy per next-character over a list of lines."""
    total, ntok = 0.0, 0
    with torch.no_grad():
        for line in lines:
            ids = tok.encode(line)
            x = torch.tensor([ids[:-1]], dtype=torch.long)
            y = torch.tensor(ids[1:], dtype=torch.long)
            logits = core(x)[0]
            total += float(F.cross_entropy(logits, y, reduction="sum"))
            ntok += y.numel()
    return total / max(ntok, 1)


@pytest.fixture
def tok():
    return make_tok()


@pytest.fixture
def tok_en():
    return make_tok_en()


@pytest.fixture
def core():
    return make_core()


@pytest.fixture
def core_demo_compat():
    """Core compatible with the committed demo_math weights (d_model=96)."""
    return make_core(vocab_size=41, d_model=96, max_seq=48)
