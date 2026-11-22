"""Мини-тесты посимвольного токенизатора."""
import pytest

from conftest import ALPHABET

from modulemind.core.tokenizer import CharTokenizer


def test_roundtrip():
    tok = CharTokenizer(ALPHABET)
    text = "2+3= 10+4="
    assert tok.decode(tok.encode(text)) == text


def test_vocab_size():
    tok = CharTokenizer(ALPHABET)
    assert tok.vocab_size == len(set(ALPHABET)) + 1  # +1 символ UNK


def test_unknown_char_maps_to_unk():
    tok = CharTokenizer(ALPHABET)
    ids = tok.encode("ab")
    assert len(ids) == 2
    assert tok.decode(ids) == "??"


def test_encode_is_list_of_ints():
    tok = CharTokenizer(ALPHABET)
    ids = tok.encode("2+2")
    assert all(isinstance(i, int) for i in ids)
    assert all(0 <= i < tok.vocab_size for i in ids)


def test_unk_char_in_alphabet_rejected():
    with pytest.raises(ValueError):
        CharTokenizer("01?")
