"""Configuration of the ModuleMind core."""
from dataclasses import dataclass


@dataclass
class CoreConfig:
    """Parameters of the tiny transformer core.

    Defaults fit weak machines: the demo module trains in seconds on CPU
    (see modules/demo_math/train.py).
    """

    vocab_size: int = 64
    d_model: int = 64
    n_layers: int = 2
    n_heads: int = 4
    max_seq: int = 32
    seed: int = 42

    def __post_init__(self):
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model={self.d_model} is not divisible by n_heads={self.n_heads}"
            )
        if self.max_seq < 2:
            raise ValueError("max_seq must be >= 2")
