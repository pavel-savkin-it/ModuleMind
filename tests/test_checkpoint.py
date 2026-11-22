"""Mini-tests for core checkpoints (release artifacts of the mind)."""
import torch

from conftest import make_core

from modulemind.core.checkpoint import save_core, load_core


def test_roundtrip(tmp_path):
    c1 = make_core(seed=9)
    idx = torch.randint(0, 14, (1, 8))
    with torch.no_grad():
        ref = c1(idx).clone()

    path = tmp_path / "core.npz"
    save_core(c1, path)
    c2 = load_core(path)

    with torch.no_grad():
        assert torch.allclose(ref, c2(idx))
    assert c2.config.d_model == c1.config.d_model
    assert c2.config.n_layers == c1.config.n_layers


def test_checkpoint_is_small(tmp_path):
    """The core stays tiny: a checkpoint must be well under 10 MB."""
    core = make_core(seed=1)
    path = tmp_path / "core.npz"
    save_core(core, path)
    assert path.stat().st_size < 10 * 1024 * 1024
