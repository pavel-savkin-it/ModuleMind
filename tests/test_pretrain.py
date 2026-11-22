"""Mini-tests for the first from-scratch pretraining (core v1)."""
from conftest import make_core, make_tok_en

from modulemind.core.pretrain import pretrain_core
from modulemind.core.tokenizer import CharTokenizer
from modulemind.data.toy_corpus import (
    ALPHABET, make_english, make_math, make_holdout,
)

MAX_SEQ = 48


def test_corpus_deterministic():
    assert make_english(30, seed=1) == make_english(30, seed=1)
    assert make_english(30, seed=1) != make_english(30, seed=2)


def test_corpus_within_alphabet_and_limit():
    tok = CharTokenizer(ALPHABET)
    lines = make_english(100, seed=3) + make_math() + make_holdout()
    for line in lines:
        assert tok.decode(tok.encode(line)) == line  # no UNK symbols
        assert len(tok.encode(line)) <= MAX_SEQ


def test_holdout_is_fresh():
    train = set(make_english(400, seed=7))
    holdout = make_holdout()
    overlap = sum(line in train for line in holdout)
    assert overlap < 0.5 * len(holdout)


def test_pretrain_reduces_loss():
    core = make_core(vocab_size=41, d_model=32, max_seq=MAX_SEQ, seed=0)
    res = pretrain_core(
        core, make_english(200, seed=1) + make_math(),
        steps=60, batch=16, lr=1e-2, log_to_console=False,
    )
    assert res["last_loss"] < res["first_loss"]


def test_pretrained_core_speaks_better():
    """Even a short pretraining must beat a random core on English."""
    tok = make_tok_en()
    holdout = make_holdout()

    random_core = make_core(vocab_size=41, d_model=32, max_seq=MAX_SEQ, seed=0)
    trained = make_core(vocab_size=41, d_model=32, max_seq=MAX_SEQ, seed=0)
    pretrain_core(trained, make_english(200, seed=1) + make_math(),
                  steps=60, batch=16, lr=1e-2, log_to_console=False)

    from conftest import sequence_loss
    assert sequence_loss(trained, tok, holdout) < sequence_loss(random_core, tok, holdout)
