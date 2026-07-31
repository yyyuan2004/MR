"""Bias/variance split within the observed and null subspaces."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from mrsim.artifacts import bias_variance_decomposition
from mrsim.data import random_ellipse_phantom
from mrsim.masks import variable_density_mask
from mrsim.recon import ridge, simulate_measurements, zero_filled

SIZE = 32


def _setup(n_realizations=24, noise_std=0.05):
    truth = torch.from_numpy(random_ellipse_phantom(SIZE, np.random.default_rng(0)))
    mask = variable_density_mask(
        (SIZE, SIZE), (SIZE * SIZE) // 4, np.random.default_rng(1), n_center=8
    )
    stack = truth[None].repeat(n_realizations, 1, 1)
    generator = torch.Generator().manual_seed(0)
    y = simulate_measurements(stack, mask, noise_std=noise_std, generator=generator)
    return truth, mask, y


def test_four_pieces_sum_to_the_expected_error():
    truth, mask, y = _setup()
    pieces = bias_variance_decomposition(zero_filled(y), truth, mask)
    assert pieces["total"] == pytest.approx(pieces["expected_total_error"], rel=1e-5)
    for key in ("bias2_observed", "bias2_null", "var_observed", "var_null"):
        assert pieces[key] >= 0.0


def test_zero_filling_puts_all_its_bias_in_the_null_space():
    """Zero filling never invents null-space content, so the null term is
    exactly the truth's own null-space energy and carries no variance."""
    truth, mask, y = _setup()
    pieces = bias_variance_decomposition(zero_filled(y), truth, mask)
    assert pieces["var_null"] == pytest.approx(0.0, abs=1e-8)
    assert pieces["bias2_null"] > pieces["bias2_observed"]


def test_noise_free_measurements_have_no_variance():
    truth, mask, y = _setup(noise_std=0.0)
    pieces = bias_variance_decomposition(zero_filled(y), truth, mask)
    assert pieces["var_observed"] == pytest.approx(0.0, abs=1e-10)
    assert pieces["var_null"] == pytest.approx(0.0, abs=1e-10)


def test_shrinkage_trades_observed_variance_for_observed_bias():
    """The distinction the two-way split cannot make."""
    truth, mask, y = _setup(noise_std=0.2)
    weak = bias_variance_decomposition(ridge(y, mask, 1e-6), truth, mask)
    strong = bias_variance_decomposition(ridge(y, mask, 5.0), truth, mask)
    assert strong["var_observed"] < weak["var_observed"]
    assert strong["bias2_observed"] > weak["bias2_observed"]


def test_rejects_insufficient_realizations():
    truth, mask, y = _setup(n_realizations=2)
    with pytest.raises(ValueError):
        bias_variance_decomposition(zero_filled(y)[:1], truth, mask)
    with pytest.raises(ValueError):
        bias_variance_decomposition(zero_filled(y)[0], truth, mask)
