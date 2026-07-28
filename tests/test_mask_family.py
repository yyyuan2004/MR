"""The repo-wide convention is that every mask meets the budget exactly and
forces the center. The parameterized family multiplies the number of masks, so
the invariant is checked across the whole family and across budgets."""

import numpy as np
import pytest
import torch

from mrsim import experiment
from mrsim.data import generate_dataset

SIZE = 16


def _cfg(fraction: float) -> dict:
    return {
        "experiment_name": "family_test",
        "seed": 0,
        "data": {"image_size": SIZE, "n_images": 8, "n_train": 6, "n_test": 2},
        "measurement": {"noise_std": 0.005},
        "mask": {
            "sampling_fraction": fraction,
            "center_fraction": 0.05,
            "lines": {"n_center_lines": 1},
            # Trimmed grids keep the test fast while still covering every family.
            "family": {
                "seeds": [0, 1],
                "variable_density_decay": [2.0, 4.0],
                "variable_density_lines_decay": [2.0],
                "psf_penalized_beta": [1.0],
                "subspace_beta": [0.0],
                "multilevel": [{"n_levels": 3, "decay": 1.5}],
            },
        },
        "greedy": {"n_candidates": 8},
        "subspace": {"d": 4, "ridge": 1e-6},
        "recon": {"ridge_lambda": 0.05, "wavelet_ista": {"wavelet": "db2", "levels": 2}},
    }


@pytest.fixture(scope="module")
def train_images() -> torch.Tensor:
    return generate_dataset(6, SIZE, seed=0)


@pytest.mark.parametrize("fraction", [0.25, 0.125, 1.0 / 12.0])
def test_family_meets_budget_and_center_at_every_acceleration(fraction, train_images):
    cfg = _cfg(fraction)
    shape, n_samples, _ = experiment.mask_budgets(cfg)
    family = experiment.build_mask_family(cfg, train_images)
    assert len(family) >= 10

    for name, mask in family.items():
        assert mask.shape == shape, name
        assert set(np.unique(mask)).issubset({0.0, 1.0}), name
        assert int(mask.sum()) == n_samples, f"{name} missed the budget"


def test_family_names_are_unique_and_varied(train_images):
    family = experiment.build_mask_family(_cfg(0.25), train_images)
    assert len(set(family)) == len(family)
    # Distinct parameters must give distinct masks, or the sweep adds no power.
    signatures = {name: mask.tobytes() for name, mask in family.items()}
    assert len(set(signatures.values())) > len(family) // 2


def test_center_is_sampled_by_every_family_member(train_images):
    family = experiment.build_mask_family(_cfg(0.25), train_images)
    center = (SIZE // 2, SIZE // 2)
    for name, mask in family.items():
        assert mask[center] == 1.0, f"{name} left the center unmeasured"
