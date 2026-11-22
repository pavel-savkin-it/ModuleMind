"""Toy corpora for core v1 and the english module (deterministic, tiny).

English template sentences + arithmetic lines. The holdout split uses two
reserved subjects ("the lion", "the tiger") that NEVER appear in training
data, so its lines are guaranteed fresh.
"""
import random

ALPHABET = "abcdefghijklmnopqrstuvwxyz .0123456789+="
MAX_LEN = 40  # every line fits max_seq=48 with room to spare

_RESERVED_SUBJECTS = ["the lion", "the tiger"]
_SUBJECTS = ["the cat", "the dog", "a bird", "my friend", "the boy",
             "the girl", "the man", "a mouse", "my sister", "the teacher"]
_VERBS = ["sits", "runs", "jumps", "sleeps", "eats", "plays", "walks", "drinks"]
_PLACES = ["on the mat", "in the garden", "under the tree", "near the house",
           "in the box", "on the grass", "by the door", "in the room",
           "under the bed", "on the table"]
_STARTS = ["i see", "i like", "this is", "look at", "here is", "we have"]
_ITEMS = ["a red ball", "a big dog", "a small cat", "a green tree",
          "a nice house", "my old friend", "a blue car", "a warm cake"]
_NOUNS = ["cats", "dogs", "balls", "trees", "cars", "books", "birds", "apples"]


def _english_line(rnd: random.Random, subjects) -> str:
    kind = rnd.random()
    if kind < 0.45:
        return f"{rnd.choice(subjects)} {rnd.choice(_VERBS)} {rnd.choice(_PLACES)}."
    if kind < 0.8:
        return f"{rnd.choice(_STARTS)} {rnd.choice(_ITEMS)}."
    return f"i have {rnd.randint(1, 20)} {rnd.choice(_NOUNS)}."


def make_english(n=800, seed=7):
    """Unique simple English sentences for training (no reserved subjects)."""
    rnd = random.Random(seed)
    lines, seen = [], set()
    while len(lines) < n:
        line = _english_line(rnd, _SUBJECTS)
        if line not in seen:
            seen.add(line)
            lines.append(line)
    return lines


def make_holdout(n=40, seed=123):
    """Fresh English sentences (reserved subjects only, disjoint from train)."""
    rnd = random.Random(seed)
    lines, seen = [], set()
    while len(lines) < n:
        line = _english_line(rnd, _RESERVED_SUBJECTS)
        if line not in seen:
            seen.add(line)
            lines.append(line)
    return lines


def make_math(max_digit=9):
    """All 'a+b=c' lines for digits 0..max_digit (100 facts by default)."""
    return [f"{a}+{b}={a + b}"
            for a in range(max_digit + 1) for b in range(max_digit + 1)]
