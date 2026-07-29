import numpy as np
import pytest
import torch

from mrsim import experiment
from mrsim.data import load_array_dataset

SIZE = 8


def _cfg(tmp_path, **overrides) -> dict:
    cfg = {
        "experiment_name": "src_test",
        "seed": 0,
        "data": {
            "source": "file",
            "path": str(tmp_path / "stack.npy"),
            "n_images": 6,
            "image_size": SIZE,
            "n_train": 3,
            "n_val": 1,
            "n_test": 2,
        },
    }
    cfg["data"].update(overrides)
    return cfg


def _write(tmp_path, arr) -> str:
    path = tmp_path / "stack.npy"
    np.save(path, arr)
    return str(path)


def test_happy_path_npy(tmp_path):
    path = _write(tmp_path, np.random.default_rng(0).random((6, SIZE, SIZE)))
    images = load_array_dataset(path, 6, SIZE)
    assert images.shape == (6, SIZE, SIZE)
    assert images.dtype == torch.float32


def test_channel_dim_is_squeezed(tmp_path):
    path = _write(tmp_path, np.random.default_rng(0).random((6, 1, SIZE, SIZE)))
    assert load_array_dataset(path, 6, SIZE).shape == (6, SIZE, SIZE)


def test_wrapped_pt_payload(tmp_path):
    path = tmp_path / "stack.pt"
    torch.save({"metadata": {"x": 1}, "images": torch.rand(6, SIZE, SIZE)}, path)
    assert load_array_dataset(path, 6, SIZE).shape == (6, SIZE, SIZE)


@pytest.mark.parametrize(
    "arr, n_images, size",
    [
        (np.random.default_rng(0).random((6, SIZE + 2, SIZE + 2)), 6, SIZE),  # wrong size
        (np.random.default_rng(0).random((2, SIZE, SIZE)), 6, SIZE),          # too few images
        (np.random.default_rng(0).random((6, SIZE, SIZE)) * 5.0, 6, SIZE),    # out of [0, 1]
    ],
)
def test_guards_fire(tmp_path, arr, n_images, size):
    with pytest.raises(ValueError):
        load_array_dataset(_write(tmp_path, arr), n_images, size)


def test_unsupported_extension(tmp_path):
    path = tmp_path / "stack.txt"
    path.write_text("nope")
    with pytest.raises(ValueError):
        load_array_dataset(path, 6, SIZE)


def test_split_is_sequential_and_val_defaults_to_empty():
    images = torch.arange(10 * SIZE * SIZE, dtype=torch.float32).reshape(10, SIZE, SIZE)
    cfg = {"data": {"n_train": 4, "n_test": 3}, "seed": 0}
    train, val, test = experiment.train_val_test_split(images, cfg)
    assert val.shape[0] == 0
    # With no validation block the two- and three-way splits must agree exactly.
    train2, test2 = experiment.train_test_split(images, cfg)
    assert torch.equal(train, train2) and torch.equal(test, test2)
    assert torch.equal(test, images[4:7])


def test_validation_block_sits_between_train_and_test():
    images = torch.arange(10 * SIZE * SIZE, dtype=torch.float32).reshape(10, SIZE, SIZE)
    cfg = {"data": {"n_train": 4, "n_val": 2, "n_test": 3}, "seed": 0}
    train, val, test = experiment.train_val_test_split(images, cfg)
    assert torch.equal(train, images[:4])
    assert torch.equal(val, images[4:6])
    assert torch.equal(test, images[6:9])


def test_cache_invalidates_when_source_file_changes(tmp_path):
    run = tmp_path / "run"
    rng = np.random.default_rng(0)
    _write(tmp_path, rng.random((6, SIZE, SIZE)))
    cfg = _cfg(tmp_path)

    first = experiment.load_or_generate_dataset(cfg, run)
    assert torch.equal(experiment.load_or_generate_dataset(cfg, run), first)

    # Rewriting the array at the same path must not silently reuse the cache.
    _write(tmp_path, rng.random((6, SIZE, SIZE)))
    assert not torch.equal(experiment.load_or_generate_dataset(cfg, run), first)


def test_cache_invalidates_when_switching_to_synthetic(tmp_path):
    run = tmp_path / "run"
    _write(tmp_path, np.random.default_rng(0).random((6, SIZE, SIZE)))
    from_file = experiment.load_or_generate_dataset(_cfg(tmp_path), run)

    synthetic_cfg = _cfg(tmp_path, source="synthetic")
    synthetic = experiment.load_or_generate_dataset(synthetic_cfg, run)
    assert not torch.equal(synthetic, from_file)
