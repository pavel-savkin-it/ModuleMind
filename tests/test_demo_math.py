"""Mini-tests for the demo module demo_math training."""
import numpy as np

from conftest import make_core

from modules.demo_math.train import train_demo


def test_training_reduces_loss_and_learns(tmp_path):
    core = make_core(vocab_size=14, d_model=32, seed=0)
    res = train_demo(core=core, steps=300, batch=32, lr=1e-2, seed=0,
                     out_dir=tmp_path, log_to_console=False)
    assert res["last_loss"] < res["first_loss"] * 0.5
    assert res["answer_acc"] >= 0.8, f"module did not learn the facts: {res}"
    assert res["baseline_acc"] < 0.3


def test_weights_file_saved_and_loadable(tmp_path):
    core = make_core(vocab_size=14, d_model=32, seed=0)
    train_demo(core=core, steps=30, seed=0, out_dir=tmp_path, log_to_console=False)
    npz = np.load(tmp_path / "lora.npz")
    keys = set(npz.files)
    assert any(k.endswith(".A") for k in keys)
    assert any(k.endswith(".B") for k in keys)
    assert all(npz[k].dtype == np.float32 for k in keys)


def test_same_seed_same_loss(tmp_path):
    def run(out):
        core = make_core(vocab_size=14, d_model=32, seed=0)
        return train_demo(core=core, steps=20, seed=3, out_dir=out,
                          log_to_console=False)
    a = run(tmp_path / "a")
    b = run(tmp_path / "b")
    assert abs(a["last_loss"] - b["last_loss"]) < 1e-6
