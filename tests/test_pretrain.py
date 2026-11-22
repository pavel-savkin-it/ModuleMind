"""Mini-tests for the first from-scratch pretraining (core v1)."""
from conftest import make_core, make_tok_en, sequence_loss

from modulemind.core.pretrain import pretrain_core
from modulemind.core.tokenizer import CharTokenizer
from modulemind.data.toy_corpus import (
    ALPHABET, make_english, make_math, make_filler, make_holdout,
)

MAX_SEQ = 48


def test_corpus_deterministic():
    assert make_english(30, seed=1) == make_english(30, seed=1)
    assert make_english(30, seed=1) != make_english(30, seed=2)
    assert make_filler(30, seed=1) == make_filler(30, seed=1)


def test_filler_has_no_knowledge():
    """The filler must contain neither grammar ('the ... is') nor math."""
    for line in make_filler(300, seed=5):
        assert "=" not in line and "+" not in line
        assert not any(line.startswith(s) for s in ("the ", "i ", "this "))


def test_corpus_within_alphabet_and_limit():
    tok = CharTokenizer(ALPHABET)
    lines = make_english(100, seed=3) + make_math() + make_holdout() \
        + make_filler(100)
    for line in lines:
        assert tok.decode(tok.encode(line)) == line  # no UNK symbols
        assert len(tok.encode(line)) <= MAX_SEQ


def test_holdout_is_fresh():
    train = set(make_english(400, seed=7))
    holdout = make_holdout()
    overlap = sum(line in train for line in holdout)
    assert overlap == 0  # reserved subjects guarantee disjointness


def test_pretrain_reduces_loss():
    core = make_core(vocab_size=41, d_model=32, max_seq=MAX_SEQ, seed=0)
    res = pretrain_core(
        core, make_filler(200, seed=1),
        steps=60, batch=16, lr=1e-2, log_to_console=False,
    )
    assert res["last_loss"] < res["first_loss"]


def test_pretrained_core_works_with_text_better():
    """Even filler pretraining must beat a random core on text handling."""
    tok = make_tok_en()
    holdout = make_holdout()

    random_core = make_core(vocab_size=41, d_model=32, max_seq=MAX_SEQ, seed=0)
    trained = make_core(vocab_size=41, d_model=32, max_seq=MAX_SEQ, seed=0)
    pretrain_core(trained, make_filler(200, seed=1),
                  steps=60, batch=16, lr=1e-2, log_to_console=False)

    assert sequence_loss(trained, tok, holdout) < sequence_loss(random_core, tok, holdout)
